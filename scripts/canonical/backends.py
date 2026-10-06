"""Exact solver backends for the verification step.

A backend solves one (query, target) pair of LocalGraphs whose `labels` hold
sigma codes, and returns a SolveOutcome:
  FOUND    at least one embedding (local target ids, one array per embedding)
  NONE     the solver proved that the target holds no embedding
  TIMEOUT  the solver was stopped at the cap before deciding
  ERROR    the solver failed (always a bug to investigate, never a result)
Time limits are fractional seconds; the solver's process group is killed at the
cap, so no call runs past its limit by more than the kill latency.

Backends here: Glasgow (default verifier) and networkx VF2 (tests and CPU smoke
runs without solver binaries). SubgraphMatching backends (GQL, CFL, DP-iso)
live in classical.py and implement the same interface.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .store import LocalGraph

FOUND, NONE, TIMEOUT, ERROR = "FOUND", "NONE", "TIMEOUT", "ERROR"


@dataclass
class SolveOutcome:
    status: str
    embeddings: list[np.ndarray] = field(default_factory=list)
    elapsed: float = 0.0
    solver: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.status == FOUND


class SolverBackend(Protocol):
    name: str

    def solve(self, query: LocalGraph, target: LocalGraph, *, time_limit: float,
              max_solutions: int = 1) -> SolveOutcome: ...


def trivially_none(query: LocalGraph, target: LocalGraph) -> str | None:
    """A reason why no embedding can exist, or None. Sound, cheap checks only."""
    if query.n > target.n:
        return "query larger than target"
    if query.num_undirected_edges > target.num_undirected_edges:
        return "query has more edges than target"
    q_labels, q_counts = np.unique(query.labels, return_counts=True)
    t_labels, t_counts = np.unique(target.labels, return_counts=True)
    if q_labels.shape[0] == 0:
        return None
    if t_labels.shape[0] == 0:
        return "target is empty"
    pos = np.searchsorted(t_labels, q_labels)
    inside = pos < t_labels.shape[0]
    pos_c = np.where(inside, pos, 0)
    present = inside & (t_labels[pos_c] == q_labels)
    if not present.all() or (t_counts[pos_c] < q_counts).any():
        return "target lacks the query's label multiset"
    return None


# ----------------------------------------------------------------------------
# Glasgow
# ----------------------------------------------------------------------------
def write_vertexlabelled_lad(graph: LocalGraph, path: str | os.PathLike, chunk: int = 1 << 18) -> None:
    """Glasgow's vertexlabelledlad: first line n, then per vertex `label degree nbr...`."""
    labels = np.asarray(graph.labels, dtype=np.int64)
    indptr = np.asarray(graph.indptr, dtype=np.int64)
    indices = np.asarray(graph.indices, dtype=np.int64)
    degrees = np.diff(indptr)
    with open(path, "wb") as handle:
        handle.write(f"{graph.n}\n".encode("ascii"))
        start = 0
        while start < graph.n:
            stop = min(graph.n, start + chunk)
            lo, hi = int(indptr[start]), int(indptr[stop])
            nbr_text = indices[lo:hi].astype("S")
            head = np.char.add(np.char.add(labels[start:stop].astype("S"), b" "), degrees[start:stop].astype("S"))
            lines = []
            offset = 0
            local_deg = degrees[start:stop]
            nbr_list = nbr_text.tolist()
            head_list = head.tolist()
            for i in range(stop - start):
                d = int(local_deg[i])
                if d:
                    lines.append(head_list[i] + b" " + b" ".join(nbr_list[offset:offset + d]))
                else:
                    lines.append(head_list[i])
                offset += d
            handle.write(b"\n".join(lines))
            handle.write(b"\n")
            start = stop


_MAPPING = re.compile(r"\((\d+) -> (\d+)\)")


def parse_glasgow_output(text: str, query_n: int) -> tuple[str | None, list[np.ndarray], dict[str, Any]]:
    status = None
    embeddings: list[np.ndarray] = []
    detail: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("status ="):
            status = line.split("=", 1)[1].strip()
        elif line.startswith("mapping ="):
            phi = np.full(query_n, -1, dtype=np.int64)
            for q, t in _MAPPING.findall(line):
                phi[int(q)] = int(t)
            embeddings.append(phi)
        elif "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            if key in ("runtime", "nodes", "propagations", "solution_count", "search_time"):
                detail[key] = value
    return status, embeddings, detail


class GlasgowBackend:
    name = "glasgow"

    def __init__(self, binary: str | list[str] | tuple[str, ...] | None = None, workdir: str | None = None,
                 extra_args: tuple[str, ...] = ()) -> None:
        """`binary` is the solver path, or a command prefix such as [python, wrapper.py]."""
        binary = binary or os.environ.get("GLASGOW_SOLVER_BIN", "glasgow_subgraph_solver")
        self.command = [binary] if isinstance(binary, str) else list(binary)
        self.binary = self.command[-1]
        self.workdir = workdir
        self.extra_args = tuple(extra_args)

    def available(self) -> bool:
        head = self.command[0]
        return bool(shutil.which(head) or Path(head).exists())

    def solve(self, query: LocalGraph, target: LocalGraph, *, time_limit: float,
              max_solutions: int = 1) -> SolveOutcome:
        t0 = time.perf_counter()
        reason = trivially_none(query, target)
        if reason:
            return SolveOutcome(NONE, [], time.perf_counter() - t0, self.name, {"trivial": reason})
        if time_limit <= 0:
            return SolveOutcome(TIMEOUT, [], 0.0, self.name, {"reason": "no time left"})
        tmp = tempfile.mkdtemp(prefix="glasgow-", dir=self.workdir)
        try:
            pattern = os.path.join(tmp, "pattern.lad")
            data = os.path.join(tmp, "target.lad")
            write_vertexlabelled_lad(query, pattern)
            write_vertexlabelled_lad(target, data)
            io_seconds = time.perf_counter() - t0
            remaining = time_limit - io_seconds
            if remaining <= 0:
                return SolveOutcome(TIMEOUT, [], time.perf_counter() - t0, self.name,
                                    {"reason": "input writing used the whole cap", "io_seconds": io_seconds})
            cmd = self.command + ["--format", "vertexlabelledlad", "--timeout", str(max(1, math.ceil(remaining)))]
            if max_solutions > 1:
                cmd += ["--count-solutions", "--print-all-solutions", "--solution-limit", str(int(max_solutions))]
            cmd += list(self.extra_args) + [pattern, data]
            out_path = os.path.join(tmp, "stdout.txt")
            with open(out_path, "wb") as out:
                proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                        start_new_session=(os.name == "posix"))
                killed = threading.Event()

                def kill() -> None:
                    killed.set()
                    try:
                        if os.name == "posix":
                            os.killpg(proc.pid, signal.SIGKILL)
                        else:
                            proc.kill()
                    except (ProcessLookupError, PermissionError, OSError):
                        pass

                timer = threading.Timer(remaining, kill)
                timer.daemon = True
                timer.start()
                try:
                    proc.wait()
                finally:
                    timer.cancel()
            text = Path(out_path).read_text(encoding="utf-8", errors="replace")
            status, embeddings, detail = parse_glasgow_output(text, query.n)
            detail.update({"io_seconds": io_seconds, "returncode": proc.returncode, "killed_at_cap": killed.is_set(),
                           "target_nodes": target.n, "target_edges": target.num_undirected_edges})
            elapsed = time.perf_counter() - t0
            if embeddings:
                return SolveOutcome(FOUND, embeddings[:max_solutions], elapsed, self.name, detail)
            if killed.is_set() or status == "aborted":
                return SolveOutcome(TIMEOUT, [], elapsed, self.name, detail)
            if status == "false":
                return SolveOutcome(NONE, [], elapsed, self.name, detail)
            if status == "true" and max_solutions > 1:
                return SolveOutcome(ERROR, [], elapsed, self.name, {**detail, "error": "status true without mappings"})
            detail["output_tail"] = text[-2000:]
            return SolveOutcome(ERROR, [], elapsed, self.name, detail)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------
# networkx VF2 (tests and CPU smoke runs)
# ----------------------------------------------------------------------------
class _Deadline(Exception):
    pass


class NetworkXBackend:
    name = "networkx_vf2"

    def __init__(self, check_every: int = 256) -> None:
        self.check_every = int(check_every)

    @staticmethod
    def _to_nx(graph: LocalGraph):
        import networkx as nx

        g = nx.Graph()
        for i, label in enumerate(np.asarray(graph.labels).tolist()):
            g.add_node(i, label=int(label))
        edges = graph.edge_list()
        g.add_edges_from(map(tuple, edges.tolist()))
        return g

    def solve(self, query: LocalGraph, target: LocalGraph, *, time_limit: float,
              max_solutions: int = 1) -> SolveOutcome:
        from networkx.algorithms import isomorphism

        t0 = time.perf_counter()
        reason = trivially_none(query, target)
        if reason:
            return SolveOutcome(NONE, [], time.perf_counter() - t0, self.name, {"trivial": reason})
        if time_limit <= 0:
            return SolveOutcome(TIMEOUT, [], 0.0, self.name, {"reason": "no time left"})
        deadline = t0 + time_limit
        calls = [0]

        def node_match(a: dict, b: dict) -> bool:
            # VF2 calls this for every candidate pair, so the deadline also stops a search that finds nothing.
            calls[0] += 1
            if calls[0] % self.check_every == 0 and time.perf_counter() > deadline:
                raise _Deadline
            return a["label"] == b["label"]

        gt, gq = self._to_nx(target), self._to_nx(query)
        matcher = isomorphism.GraphMatcher(gt, gq, node_match=node_match)
        found: list[np.ndarray] = []
        stopped = False
        try:
            for mapping in matcher.subgraph_monomorphisms_iter():
                phi = np.full(query.n, -1, dtype=np.int64)
                for t_node, q_node in mapping.items():
                    phi[int(q_node)] = int(t_node)
                found.append(phi)
                if len(found) >= max_solutions:
                    break
                if time.perf_counter() > deadline:
                    stopped = True
                    break
        except _Deadline:
            stopped = True
        elapsed = time.perf_counter() - t0
        if found:
            return SolveOutcome(FOUND, found, elapsed, self.name, {"killed_at_cap": stopped})
        if stopped:
            return SolveOutcome(TIMEOUT, [], elapsed, self.name, {"killed_at_cap": True})
        return SolveOutcome(NONE, [], elapsed, self.name, {})


def make_backend(name: str, **kwargs: Any) -> SolverBackend:
    if name == "glasgow":
        return GlasgowBackend(**kwargs)
    if name in ("networkx", "networkx_vf2", "vf2"):
        return NetworkXBackend(**kwargs)
    if name in ("gql", "cfl", "dpiso"):
        from .classical import SubgraphMatchingBackend

        return SubgraphMatchingBackend(algorithm=name, **kwargs)
    raise ValueError(f"unknown solver backend {name!r}")
