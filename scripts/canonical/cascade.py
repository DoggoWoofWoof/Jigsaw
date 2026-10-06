"""The query pipeline: the anytime ladder, FilterAll and the direct solver.

Every method answers the find-first question: return one verified embedding of
the query, or prove that none exists, within T seconds.

  jigsaw*    The ranker orders the blocks and the policy turns the ranking into
             nested prefixes. Step i takes the first K_i blocks, builds the
             candidate G[C(S)], prunes it (prune.py) and runs the solver on the
             kept components, smallest first. Non-final steps get an absolute
             solver cap (TIME.step_cap_seconds); the final step takes every block
             and the remainder of T. The first verified embedding ends the query.
  filterall  One step: every block, the whole of T.
  direct     The data graph, sigma-labelled, goes to the solver as it is.

The candidate of every block is the label-filtered graph
G[{v : sigma(v) in sigma(V_Q)}], which the label index reads directly. A step
whose candidate already holds every node carrying a query label has that same
candidate; it is closed, gets the remainder of T like the final step, and a
NONE there proves that no embedding exists. Before any step, two global checks
prove NONE outright: a query label that the data graph holds fewer times than
the query needs, and a query edge whose endpoint types never meet on a data
edge (every sigma key contains the node type, so an embedding maps each query
edge onto a data edge between the same two types).

A step is skipped when its candidate, or its pruned targets, equal those of the
previous step and the previous answer carries over: NONE always carries over,
TIMEOUT carries over to a non-closed step (same cap, same input).

Operator ablations (E9) switch off one part: halo=False drops the one-hop
neighbours, prune_passes=0 skips the LDF/NLF passes, component_mode "none"
skips the component filter, component_cap keeps the smallest components only,
and label_filter=False hands the solver the unfiltered block candidate (the
final step is then the whole graph). A NONE from a capped component list proves
nothing, so it is reported as INCOMPLETE, a status that decides no query.

Sensitivity settings (E16) change one choice and keep every guarantee: halo=2
grows the halo a second hop, policy "topk" replaces the hybrid selector, the
hybrid policy_params (seed_count) change its seeds, and on a sensitivity
partition of B blocks a ladder step of K primary blocks (of P) takes
ceil(K * B / P) blocks, the same share of the graph.

Time. The clock starts before the query's labels are computed and counts the
stages query_labels, retrieval, candidate, prune, components and solve. The
independent witness check and per-step coverage bookkeeping run on a paused
clock. Because non-final caps do not depend on T, a run at T_max records the
same steps as a run at any T <= T_max up to time T, so solved@T is
time_found <= T for every T of the solved@t curve. Every step records t_start
and t_end and every solver call its start t, all on the query clock, so the
state of the query at any t <= T_max can be read from its record.

Failure localization. On a positive, each step records the coverage of the
planted witness by its blocks (homecov), by blocks plus halo (fullcov, covered)
and, once pruned, where the witness went (prune.locate: kept in which target,
dropped by the component cap, or lost, which soundness rules out).
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from . import backends, fastcand, features, protocol, prune, queries, rankers, verify
from .store import LocalGraph

FOUND, NONE, TIMEOUT, ERROR = backends.FOUND, backends.NONE, backends.TIMEOUT, backends.ERROR
INCOMPLETE = "INCOMPLETE"  # the solver found nothing in a capped component list
LADDER_OFF = "off"  # escalate / skip_below value that switches a planner setting off (the protocol's is ignored)
STAGES = tuple(protocol.TIME["stages_counted"])


# ----------------------------------------------------------------------------
# Label index: the nodes of every sigma code
# ----------------------------------------------------------------------------
def label_index_dir(store: Any, regime: str) -> Path:
    return Path(store.dir) / "label_index" / regime


def build_label_index(store: Any, regime: str) -> dict[str, Any]:
    """nodes[ptr[c]:ptr[c + 1]] are the nodes with code c, ascending."""
    table = store.sigma(regime)
    codes = np.asarray(table.codes, dtype=np.int64)
    order = np.argsort(codes, kind="stable").astype(np.int32)
    ptr = np.zeros(table.num_codes + 1, dtype=np.int64)
    np.cumsum(np.asarray(table.df, dtype=np.int64), out=ptr[1:])
    if int(ptr[-1]) != codes.shape[0]:
        raise AssertionError("sigma df does not sum to the node count")
    directory = label_index_dir(store, regime)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "ptr.npy", ptr, allow_pickle=False)
    np.save(directory / "nodes.npy", order, allow_pickle=False)
    info = {"version": "label-index-v1", "dataset": store.dataset, "regime": regime,
            "num_nodes": int(store.num_nodes), "num_codes": int(table.num_codes)}
    (directory / "info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return info


class LabelIndex:
    def __init__(self, store: Any, regime: str, *, mmap: bool | None = None) -> None:
        directory = label_index_dir(store, regime)
        path = directory / "info.json"
        if not path.exists():
            raise FileNotFoundError(f"{directory} missing; run python -m scripts.canonical.cascade build-label-index")
        info = json.loads(path.read_text(encoding="utf-8"))
        table = store.sigma(regime)
        if info.get("num_nodes") != int(store.num_nodes) or info.get("num_codes") != int(table.num_codes):
            raise ValueError(f"stale label index in {directory}: {info}")
        mode = "r" if (store.mmap if mmap is None else mmap) else None
        self.ptr = np.load(directory / "ptr.npy", mmap_mode=mode)
        self.nodes = np.load(directory / "nodes.npy", mmap_mode=mode)
        self.num_codes = int(table.num_codes)

    def nodes_of(self, codes: np.ndarray) -> np.ndarray:
        """Sorted nodes whose code is one of `codes` (fresh codes have none)."""
        labels = np.unique(np.asarray(codes, dtype=np.int64))
        labels = labels[(labels >= 0) & (labels < self.num_codes)]
        parts = [np.asarray(self.nodes[int(self.ptr[c]):int(self.ptr[c + 1])], dtype=np.int64) for c in labels]
        return np.sort(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)


# ----------------------------------------------------------------------------
# Clock
# ----------------------------------------------------------------------------
class Clock:
    """Stage timer with an absolute deadline; paused time moves the deadline."""

    def __init__(self, limit: float) -> None:
        self.t0 = time.perf_counter()
        self.limit = float(limit)
        self.paused = 0.0
        self.stages = {name: 0.0 for name in STAGES}

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0 - self.paused

    def remaining(self) -> float:
        return self.limit - self.elapsed()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - start

    @contextmanager
    def pause(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.paused += time.perf_counter() - start


# ----------------------------------------------------------------------------
# Context
# ----------------------------------------------------------------------------
@dataclass
class Cascade:
    """One method on one store and regime."""

    store: Any
    regime: str
    method: str
    backend: Any
    bank: rankers.RankerBank | None = None
    embed: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None
    rel_table: np.ndarray | None = None
    budgets: Sequence[int] | None = None
    step_cap: float | None = None
    component_mode: str | None = None
    component_cap: int | None = None
    halo: bool | int = True  # hops of the halo: True (1) canonical, False (0) E9, 2 in E16
    label_filter: bool = True
    prune_passes: int | None = None
    policy: str | None = None  # the method's policy unless set (E16: hybrid vs topk)
    policy_params: dict[str, Any] = field(default_factory=dict)
    # planner settings: None takes the protocol's value for the dataset and regime (none before the freeze),
    # LADDER_OFF switches the setting off
    escalate: float | str | None = None  # seconds after which the next step is the whole-graph step
    skip_below: int | str | None = None  # label nodes at or below which a query skips retrieval and the ladder
    record_coverage: bool = True
    _label_index: LabelIndex | None = field(default=None, repr=False)
    _full_target: LocalGraph | None = field(default=None, repr=False)
    _type_pairs: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        spec = protocol.METHODS.get(self.method)
        if spec is None or "external" in spec:
            raise ValueError(f"{self.method!r} is not a cascade method")
        self.spec = spec
        if spec.get("ranker") and self.bank is None:
            raise ValueError(f"{self.method} needs a RankerBank")
        if spec.get("ranker") in rankers.NEEDS_EMBEDDING and self.embed is None:
            raise ValueError(f"{self.method} needs a query encoder")
        if int(self.halo) < 0:
            raise ValueError(f"halo hops must be >= 0, got {self.halo!r}")
        if (self.policy is not None or self.policy_params) and not spec.get("ranker"):
            raise ValueError(f"{self.method} selects no blocks: it takes no policy settings")
        if self.policy not in (None, "hybrid", "topk"):
            raise ValueError(f"unknown selection policy {self.policy!r}")
        if self.policy_params and self.selection_policy != "hybrid":
            raise ValueError(f"policy settings {sorted(self.policy_params)} apply to the hybrid policy only")
        self.table = self.store.sigma(self.regime)
        self.num_blocks = int(self.store.num_blocks)
        if self.budgets is None:
            self.budgets = protocol.ladder_for(self.store.dataset) if spec.get("ladder") else [self.num_blocks]
            if spec.get("ladder") and getattr(self.store, "partition_name", "primary") != "primary":
                primary = int(protocol.DATASETS[self.store.dataset]["partitions"])
                # a sensitivity partition (E16) keeps the share of blocks of every step
                self.budgets = sorted({max(1, -(-int(k) * self.num_blocks // primary)) for k in self.budgets})
        self.budgets = [min(int(k), self.num_blocks) for k in self.budgets]
        if self.budgets[-1] != self.num_blocks:
            self.budgets.append(self.num_blocks)
        self.step_cap = float(protocol.TIME["step_cap_seconds"] if self.step_cap is None else self.step_cap)
        if not spec.get("ranker") and (self.escalate is not None or self.skip_below is not None):
            raise ValueError(f"{self.method} selects no blocks and has no ladder to plan: drop escalate and skip_below")
        self.escalate = self._planner_setting(self.escalate, "escalate_seconds")
        self.skip_below = self._planner_setting(self.skip_below, "skip_below_nodes")

    def _planner_setting(self, value: float | str | None, name: str) -> float | None:
        if value == LADDER_OFF or not self.spec.get("ranker"):
            return None
        if value is None:
            return protocol.ladder_setting(name, self.store.dataset, self.regime)
        return float(value)

    @property
    def label_index(self) -> LabelIndex:
        if self._label_index is None:
            self._label_index = LabelIndex(self.store, self.regime)
        return self._label_index

    @property
    def type_pairs(self) -> np.ndarray:
        """[T, T] bool: the ordered type pairs that occur on data edges (meta.json relation_pairs)."""
        if self._type_pairs is None:
            self._type_pairs = features.store_relation_table(self.store) >= 0
        return self._type_pairs

    def full_target(self) -> LocalGraph:
        """The whole data graph, sigma-labelled (the direct method's target)."""
        if self._full_target is None:
            n = int(self.store.num_nodes)
            self._full_target = LocalGraph(global_ids=np.arange(n, dtype=np.int64),
                                           indptr=np.asarray(self.store.indptr, dtype=np.int64),
                                           indices=np.asarray(self.store.indices, dtype=np.int32),
                                           labels=np.asarray(self.table.codes, dtype=np.int32))
        return self._full_target

    @property
    def selection_policy(self) -> str:
        return str(self.policy or self.spec["policy"])

    # ------------------------------------------------------------------ helpers
    def options(self) -> dict[str, Any]:
        """The selection, candidate and pruning settings of this run (the protocol's unless an ablation
        changes them)."""
        return {"policy": self.selection_policy, "policy_params": dict(self.policy_params),
                "halo": self.halo, "label_filter": self.label_filter,
                "prune_passes": int(protocol.CANDIDATE["fixpoint_max_iterations"] if self.prune_passes is None
                                    else self.prune_passes),
                "component_mode": str(protocol.CANDIDATE["component_mode"] if self.component_mode is None
                                      else self.component_mode),
                "component_cap": protocol.CANDIDATE["component_cap"] if self.component_cap is None
                else int(self.component_cap),
                "escalate": None if self.escalate is None else float(self.escalate),
                "skip_below": None if self.skip_below is None else int(self.skip_below)}

    def query_inputs(self, q: queries.Query, codes: np.ndarray) -> rankers.QueryInputs:
        name = self.spec["ranker"]
        qi = rankers.QueryInputs(q.uid, codes, np.asarray(q.payload["types"], dtype=np.int64), q.indptr, q.indices)
        if name in rankers.NEEDS_EMBEDDING or name == "mean_feature":
            inputs = features.query_graph_inputs(self.store, q, self.regime, codes, self.rel_table)
            qi.x = inputs["x"]
            if name in rankers.NEEDS_EMBEDDING:
                qi.embedding = np.asarray(self.embed(inputs), dtype=np.float64).reshape(-1)
        return qi

    def global_candidate(self, codes: np.ndarray, stats: dict[str, Any]) -> LocalGraph:
        nodes = self.label_index.nodes_of(codes)
        graph = fastcand.induced(self.store, nodes, self.regime)
        stats.update({"candidate_nodes": int(graph.n), "candidate_edges": int(graph.num_undirected_edges),
                      "adjacency_entries_read": int(self.store.degree(nodes).sum()) if nodes.shape[0] else 0,
                      "via": "label_index"})
        return graph

    def solve_targets(self, query: LocalGraph, targets: list[LocalGraph], budget: float,
                      clock: Clock) -> tuple[str, np.ndarray | None, list[dict[str, Any]]]:
        """Targets in order until a witness, an error or the budget runs out."""
        end = time.perf_counter() + budget
        calls: list[dict[str, Any]] = []
        for j, target in enumerate(targets):
            left = end - time.perf_counter()
            if left <= 0:
                calls.append({"target": j, "status": "not_started", "t": clock.elapsed()})
                return TIMEOUT, None, calls
            t = clock.elapsed()
            with clock.stage("solve"):
                out = self.backend.solve(query, target, time_limit=left)
            calls.append({"target": j, "status": out.status, "seconds": out.elapsed, "t": t, "nodes": target.n,
                          "edges": target.num_undirected_edges,
                          **{k: v for k, v in out.detail.items() if k in ("trivial", "killed_at_cap", "error")}})
            if out.status == FOUND:
                phi = np.asarray(out.embeddings[0], dtype=np.int64)
                return FOUND, np.asarray(target.global_ids, dtype=np.int64)[phi], calls
            if out.status in (TIMEOUT, ERROR):
                return out.status, None, calls
        return NONE, None, calls

    # ------------------------------------------------------------------ one query
    def run(self, q: queries.Query, *, time_limit: float | None = None) -> dict[str, Any]:
        T = float(protocol.TIME["primary_T_seconds"] if time_limit is None else time_limit)
        clock = Clock(T)
        rec: dict[str, Any] = {"uid": q.uid, "family": q.family, "size": q.size, "kind": q.kind,
                               "regime": self.regime, "method": self.method, "dataset": self.store.dataset,
                               "T": T, "step_cap": self.step_cap, "certified": q.certified,
                               "options": self.options(), "steps": []}
        with clock.stage("query_labels"):
            codes = self.table.query_codes(q.payload, self.store)
            query = q.local_graph(codes)
            req = prune.query_requirements(query)
            df = self.table.label_frequency(req.labels)
            global_nodes = int(df.sum())
            short = bool((df < req.counts).any())
            qtypes = np.asarray(q.payload["types"], dtype=np.int64)
            edges = np.asarray(q.edges, dtype=np.int64).reshape(-1, 2)
            foreign_edge = bool(edges.shape[0]) and not bool(self.type_pairs[qtypes[edges[:, 0]], qtypes[edges[:, 1]]].all())
        rec.update({"query_nodes": query.n, "query_edges": query.num_undirected_edges,
                    "query_labels": int(req.labels.shape[0]), "label_nodes_global": global_nodes})

        if self.spec.get("pruning") == "solver only":
            t_start = clock.elapsed()
            status, witness, calls = self.solve_targets(query, [self.full_target()], clock.remaining(), clock)
            rec["steps"].append({"step": 0, "budget": self.num_blocks, "final": True, "closed": True,
                                 "status": status, "solver_calls": calls, "t_start": t_start,
                                 "t_end": clock.elapsed()})
            return self._finish(rec, q, clock, status, witness, decided_step=0)

        if short:
            rec["certificate"] = "global label multiset"
            return self._finish(rec, q, clock, NONE, None, decided_step=None)
        if foreign_edge:
            rec["certificate"] = "edge type pair"
            return self._finish(rec, q, clock, NONE, None, decided_step=None)

        # A query whose label nodes are few reads them whole: the ladder would only add retrieval cost
        # to a graph the first step already covers (the global label frequency is known without a read).
        skipped = bool(self.spec.get("ranker")) and self.skip_below is not None and global_nodes <= self.skip_below
        budgets = [self.num_blocks] if skipped else self.budgets
        if skipped:
            rec["skipped_ladder"] = True
        selector = None
        if self.spec.get("ranker") and not skipped:
            with clock.stage("retrieval"):
                ranking, info = self.bank.rank(self.spec["ranker"], self.query_inputs(q, codes))
                selector = rankers.make_selector(self.selection_policy, ranking, self.store.part,
                                                 **self.policy_params)
            rec["ranker_info"] = info
        home = planted = None
        if self.record_coverage and q.kind == "positive":
            planted = np.asarray(q.planted, dtype=np.int64)  # query node j maps to planted[j]
            if selector is not None:
                with clock.pause():
                    home = np.asarray(self.store.part.node_block[planted], dtype=np.int64)

        prev_nodes: np.ndarray | None = None
        prev_targets: tuple | None = None
        prev_status: str | None = None
        for i, budget in enumerate(budgets):
            final = budget >= self.num_blocks
            escalated = (not final and selector is not None and self.escalate is not None
                         and clock.elapsed() >= self.escalate)
            if escalated:
                budget, final = self.num_blocks, True  # the whole-graph step is the ladder's last rung
            step: dict[str, Any] = {"step": i, "budget": int(budget), "final": final, "t_start": clock.elapsed()}
            if escalated:
                step["escalated"] = True
            rec["steps"].append(step)
            if clock.remaining() <= 0:
                step["status"] = "not_started"
                return self._finish(rec, q, clock, TIMEOUT, None, decided_step=None)
            cstats: dict[str, Any] = {}
            if (final or selector is None) and not self.label_filter:
                with clock.stage("candidate"):
                    cand = self.full_target()
                cstats.update({"candidate_nodes": int(cand.n), "candidate_edges": int(cand.num_undirected_edges),
                               "via": "whole graph"})
            elif final or selector is None:
                with clock.stage("candidate"):
                    cand = self.global_candidate(codes, cstats)
            else:
                with clock.stage("retrieval"):
                    blocks = selector.prefix(budget)
                with clock.stage("candidate"):
                    cand = fastcand.candidate(self.store, blocks, codes, self.regime, stats=cstats, halo=self.halo,
                                              label_filter=self.label_filter)
                if home is not None:
                    with clock.pause():
                        # fullcov: this step's candidate holds every planted node; homecov: the
                        # selected blocks alone do (failure localization separates the two)
                        in_home = np.isin(home, blocks)
                        covered = np.isin(planted, cand.global_ids)
                        step.update({"fullcov": bool(covered.all()), "covered": int(covered.sum()),
                                     "homecov": bool(in_home.all())})
            if self.label_filter:
                closed = final or cand.n == global_nodes
            else:
                with clock.stage("candidate"):
                    closed = final or int((req.label_index(np.asarray(cand.labels, dtype=np.int64)) >= 0).sum()) == global_nodes
            step.update({"closed": closed, "candidate": cstats})
            status, witness = self._step(step, cand, query, req, closed, clock, prev_status, prev_nodes, prev_targets,
                                         planted)
            step["t_end"] = clock.elapsed()
            prev_nodes = np.asarray(cand.global_ids)
            prev_targets = step.pop("_targets_key", prev_targets)
            prev_status = status
            if status == FOUND:
                return self._finish(rec, q, clock, FOUND, witness, decided_step=i)
            if status == ERROR:
                return self._finish(rec, q, clock, ERROR, None, decided_step=i)
            if status in (NONE, INCOMPLETE) and closed:
                # later steps hold the same label nodes, so they repeat this answer
                return self._finish(rec, q, clock, status, None, decided_step=i if status == NONE else None)
            if final:
                return self._finish(rec, q, clock, status, None, decided_step=None)
        return self._finish(rec, q, clock, TIMEOUT, None, decided_step=None)

    def _step(self, step: dict[str, Any], cand: LocalGraph, query: LocalGraph, req: prune.QueryRequirements,
              closed: bool, clock: Clock, prev_status: str | None, prev_nodes: np.ndarray | None,
              prev_targets: tuple | None, planted: np.ndarray | None = None) -> tuple[str, np.ndarray | None]:
        """Prune and solve one step, or carry the previous answer over to an identical input."""
        carries = prev_status in (NONE, INCOMPLETE) or (prev_status == TIMEOUT and not closed)
        if carries and prev_nodes is not None and np.array_equal(cand.global_ids, prev_nodes):
            step.update({"status": prev_status, "skipped": "identical candidate"})
            return str(prev_status), None
        if self.label_filter:
            t_prune = time.perf_counter()
            with clock.stage("prune"):
                result = prune.prune(cand, query, mode=self.component_mode, component_cap=self.component_cap,
                                     requirements=req, max_iterations=self.prune_passes)
            comp_seconds = float(result.stats.get("components_seconds", 0.0))
            clock.stages["prune"] -= comp_seconds
            clock.stages["components"] += comp_seconds
            step["prune"] = {k: v for k, v in result.stats.items() if k not in ("candidate_nodes", "candidate_edges")}
            step["prune"]["wall_seconds"] = time.perf_counter() - t_prune
        else:
            step["prune"] = {"skipped": "no label filter"}
            result = prune.PruneResult([cand] if cand.n else [], mode="unpruned")
        targets, capped = result.targets, int(result.stats.get("components_capped") or 0)
        if planted is not None:
            with clock.pause():
                # failure localization: did the witness reach a target, and which one
                step["witness"] = prune.locate(result, cand, planted)
        key = tuple(t.global_ids.tobytes() for t in targets)
        step["_targets_key"] = key
        if carries and prev_targets is not None and key == prev_targets:
            step.update({"status": prev_status, "skipped": "identical targets"})
            return str(prev_status), None
        solve_budget = clock.remaining() if closed else min(self.step_cap, clock.remaining())
        status, witness, calls = self.solve_targets(query, targets, solve_budget, clock)
        if status == NONE and capped:
            status = INCOMPLETE
        step.update({"status": status, "solve_budget": solve_budget, "solver_calls": calls})
        return status, witness

    def _finish(self, rec: dict[str, Any], q: queries.Query, clock: Clock, status: str,
                witness: np.ndarray | None, *, decided_step: int | None) -> dict[str, Any]:
        elapsed = clock.elapsed()
        rec.update({"status": status, "elapsed": elapsed, "stages": dict(clock.stages),
                    "decided_step": decided_step, "time_found": None, "time_decided": None})
        if status == FOUND:
            t0 = time.perf_counter()
            check = verify.check_embedding(self.store, q.payload, q.edges, witness, self.regime)
            rec["witness_check_seconds"] = time.perf_counter() - t0
            rec["witness"] = witness.astype(int).tolist()
            rec["witness_check"] = "ok" if check.ok else check.reason
            if check.ok:
                rec["time_found"] = rec["time_decided"] = elapsed
            else:
                rec["status"] = ERROR  # a rejected witness is a soundness failure, never a solve
                rec["error"] = f"witness rejected: {check.reason}"
        elif status == NONE:
            rec["time_decided"] = elapsed
        return rec


# ----------------------------------------------------------------------------
# Enumeration on the closed candidate
# ----------------------------------------------------------------------------
def enumerate_embeddings(cascade: Cascade, q: queries.Query, *, time_limit: float,
                         limit: int) -> dict[str, Any]:
    """Count embeddings, up to `limit`, on the label-filtered graph (the only candidate that holds them all)."""
    clock = Clock(time_limit)
    codes = cascade.table.query_codes(q.payload, cascade.store)
    query = q.local_graph(codes)
    stats: dict[str, Any] = {}
    cand = cascade.global_candidate(codes, stats)
    result = prune.prune(cand, query, mode="separate")
    found = 0
    complete = True
    for target in result.targets:
        left = clock.remaining()
        if left <= 0 or found >= limit:
            complete = False
            break
        out = cascade.backend.solve(query, target, time_limit=left, max_solutions=limit - found)
        if out.status == FOUND:
            found += len(out.embeddings)
            for phi in out.embeddings:
                check = verify.check_embedding(cascade.store, q.payload, q.edges,
                                               np.asarray(target.global_ids)[phi], cascade.regime)
                if not check.ok:
                    return {"uid": q.uid, "status": ERROR, "error": f"witness rejected: {check.reason}"}
            if found >= limit or out.detail.get("killed_at_cap"):
                complete = False
                break
        elif out.status != NONE:
            complete = False
            break
    return {"uid": q.uid, "status": "complete" if complete else "partial", "embeddings": found, "limit": limit,
            "elapsed": clock.elapsed(), "targets": len(result.targets), "candidate": stats}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from .store import GraphStore

    parser = argparse.ArgumentParser(description="Label index for the cascade.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-label-index")
    b.add_argument("--root", required=True)
    b.add_argument("--dataset", required=True)
    b.add_argument("--regimes", nargs="*", default=None)
    b.add_argument("--run-dir", default=None, help="record the build in a manifest there (offline_build.py, E17)")
    args = parser.parse_args(argv)
    from . import offline_build

    store = GraphStore(args.root, args.dataset, mmap=True)
    regimes = args.regimes or [r for r in protocol.regimes_for(args.dataset) if (store.dir / "sigma" / r).exists()]
    inputs = {"store_meta": store.dir / "meta.json", **{f"sigma/{r}": store.dir / "sigma" / r for r in regimes}}
    with offline_build.build_run(args.run_dir, stage="label_index", store_dir=store.dir, dataset=args.dataset,
                                 config={"regimes": regimes}, inputs=inputs, argv=argv) as written:
        for regime in regimes:
            print(json.dumps(build_label_index(store, regime), sort_keys=True))
            written.append(label_index_dir(store, regime))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
