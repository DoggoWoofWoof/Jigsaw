"""Synthetic stores for the canonical-package tests (CPU only, no solver binaries)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def synthetic_graph(num_nodes: int = 600, num_blocks: int = 6, seed: int = 7):
    """A planted-community graph with typed nodes, duplicate attribute rows and -0.0 values."""
    rng = np.random.default_rng(seed)
    community = rng.integers(0, num_blocks, size=num_nodes)
    edges = []
    for _ in range(num_nodes * 4):
        u = int(rng.integers(0, num_nodes))
        same = np.flatnonzero(community == community[u])
        v = int(rng.choice(same)) if rng.random() < 0.85 else int(rng.integers(0, num_nodes))
        edges.append((u, v))
    edges += [(3, 3), (5, 9), (9, 5), (5, 9)]  # self loop and duplicates the store must remove
    edges = np.asarray(edges, dtype=np.int64)
    node_type = rng.integers(0, 3, size=num_nodes).astype(np.int8)
    attrs = rng.normal(size=(num_nodes, 8)).astype(np.float32)
    prototypes = rng.normal(size=(5, 8)).astype(np.float32)
    shared = rng.random(num_nodes) < 0.3
    attrs[shared] = prototypes[rng.integers(0, 5, size=int(shared.sum()))]
    attrs[node_type == 2] = 0.0  # a type without attributes, like MAG authors
    zeros = np.flatnonzero(node_type == 2)
    attrs[zeros[: len(zeros) // 2], 0] = -0.0  # must be equal to +0.0
    classes = rng.integers(0, 5, size=num_nodes).astype(np.int32)
    classes[node_type == 2] = -1
    node_fine = community * 3 + (np.arange(num_nodes) % 3)
    return {
        "edges": edges,
        "num_nodes": num_nodes,
        "node_type": node_type,
        "attrs": attrs,
        "node_block": community,
        "node_fine": node_fine,
        "classes": classes,
    }


def clustered_graph(seed: int = 7):
    """synthetic_graph() with triadic closure and connected fine parts.

    Real graphs are clustered: 10-50% of random walks on CoraFull and Arxiv give a
    dense (average degree >= 3) induced query, against 0.3% on synthetic_graph().
    Fine parts are consecutive thirds of a BFS order inside each community, so
    they are connected pieces like METIS sub-partitions.
    """
    g = synthetic_graph(seed=seed)
    rng = np.random.default_rng(seed + 1)
    n = g["num_nodes"]
    e = g["edges"][g["edges"][:, 0] != g["edges"][:, 1]]
    nbrs = [set() for _ in range(n)]
    for u, v in e.tolist():
        nbrs[u].add(v)
        nbrs[v].add(u)
    closing = []
    for u in range(n):
        near = sorted(nbrs[u])
        for _ in range(2):
            if len(near) >= 2:
                a, b = rng.choice(near, size=2, replace=False)
                closing.append((int(a), int(b)))
    g["edges"] = np.concatenate([g["edges"], np.asarray(closing, dtype=np.int64)])
    for a, b in closing:
        nbrs[a].add(b)
        nbrs[b].add(a)
    community = g["node_block"]
    node_fine = np.empty(n, dtype=np.int64)
    for c in range(int(community.max()) + 1):
        members = set(np.flatnonzero(community == c).tolist())
        order, seen = [], set()
        for start in sorted(members):
            if start in seen:
                continue
            seen.add(start)
            queue = [start]
            while queue:
                u = queue.pop(0)
                order.append(u)
                for w in sorted(nbrs[u] & members):
                    if w not in seen:
                        seen.add(w)
                        queue.append(w)
        third = -(-len(order) // 3)
        for rank, u in enumerate(order):
            node_fine[u] = c * 3 + rank // third
    g["node_fine"] = node_fine
    return g


def _write_synthetic(root, name, g):
    from scripts.canonical import sigma, store

    store.write_store(
        root,
        name,
        edges=g["edges"],
        num_nodes=g["num_nodes"],
        node_type=g["node_type"],
        attrs=g["attrs"],
        node_block=g["node_block"],
        node_fine=g["node_fine"],
        classes=g["classes"],
        meta={"schema": "synthetic"},
        partition_info={"tool": "planted", "seed": 7},
    )
    sigma.build_store_regimes(root / name, ["R0", "R1", "R2-512", "R2-32"])


@pytest.fixture(scope="session")
def clustered_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("clustered")
    g = clustered_graph()
    _write_synthetic(root, "synth", g)
    return root, g


@pytest.fixture(scope="session")
def synthetic_store(tmp_path_factory):
    from scripts.canonical import sigma, store

    root = tmp_path_factory.mktemp("store")
    g = synthetic_graph()
    store.write_store(
        root,
        "synth",
        edges=g["edges"],
        num_nodes=g["num_nodes"],
        node_type=g["node_type"],
        attrs=g["attrs"],
        node_block=g["node_block"],
        node_fine=g["node_fine"],
        classes=g["classes"],
        meta={"schema": "synthetic"},
        partition_info={"tool": "planted", "seed": 7},
    )
    sigma.build_store_regimes(root / "synth", ["R0", "R1", "R2-512", "R2-32"])
    return root, g
