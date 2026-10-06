"""Independent witness check.

check_embedding() re-reads raw data from the store and confirms that a returned
mapping is an embedding of the query:
  1. it is total over the query nodes and injective;
  2. every query node's declared key equals its image's key under the regime
     (R0: node type and every attribute value; R1: type and class; R2-K: type and
     the hash bucket recomputed here from the raw attribute bytes);
  3. every query edge maps to a data edge.
It shares no code with the pipeline's sigma dictionaries, filters or solvers, so
a bug in those cannot hide a wrong answer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    reason: str = ""


def _values(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows)
    if np.issubdtype(rows.dtype, np.floating):
        return np.where(rows == 0, np.zeros((), dtype=rows.dtype), rows)
    return rows


def _bucket(node_type: int, row: np.ndarray, buckets: int) -> int:
    row = np.ascontiguousarray(_values(np.asarray(row).reshape(1, -1)))
    h = hashlib.blake2b(digest_size=16, person=b"jigsaw.sigma.v1")
    h.update(np.asarray([node_type], dtype="<i2").tobytes())
    h.update(row.tobytes())
    return int.from_bytes(h.digest()[:8], "big") % int(buckets)


def _has_edge(indptr: np.ndarray, indices: np.ndarray, u: int, v: int) -> bool:
    row = np.asarray(indices[int(indptr[u]):int(indptr[u + 1])])
    pos = int(np.searchsorted(row, v))
    return pos < row.shape[0] and int(row[pos]) == v


def check_embedding(store: Any, payload: Mapping[str, Any], edges: np.ndarray, mapping: Any,
                    regime: str) -> CheckResult:
    """Check `mapping` (query node i -> data node mapping[i]) against raw store data."""
    types_q = np.asarray(payload["types"], dtype=np.int64)
    n = int(types_q.shape[0])
    phi = np.asarray(mapping, dtype=np.int64).reshape(-1)
    if phi.shape[0] != n:
        return CheckResult(False, f"mapping covers {phi.shape[0]} of {n} query nodes")
    if (phi < 0).any() or (phi >= int(store.indptr.shape[0]) - 1).any():
        return CheckResult(False, "mapping points outside the data graph")
    if np.unique(phi).shape[0] != n:
        return CheckResult(False, "mapping is not injective")
    types_d = np.asarray(store.node_type[phi], dtype=np.int64)
    if not np.array_equal(types_d, types_q):
        return CheckResult(False, "node type differs")
    if regime == "R0":
        rows_d = _values(np.asarray(store.attrs[phi]))
        rows_q = _values(np.asarray(payload["attrs"]))
        if rows_q.shape != rows_d.shape or not bool(np.all(rows_q == rows_d)):
            return CheckResult(False, "attribute values differ")
    elif regime == "R1":
        if store.classes is None:
            return CheckResult(False, "store has no classes")
        if not np.array_equal(np.asarray(store.classes[phi], dtype=np.int64),
                              np.asarray(payload["classes"], dtype=np.int64)):
            return CheckResult(False, "class differs")
    elif regime.startswith("R2-"):
        buckets = int(regime.split("-", 1)[1])
        declared = np.asarray(payload["buckets"], dtype=np.int64)
        for i in range(n):
            if _bucket(int(types_d[i]), np.asarray(store.attrs[phi[i]]), buckets) != int(declared[i]):
                return CheckResult(False, f"hash bucket differs at query node {i}")
    else:
        return CheckResult(False, f"unknown regime {regime}")
    for a, b in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        if not _has_edge(store.indptr, store.indices, int(phi[a]), int(phi[b])):
            return CheckResult(False, f"query edge ({a},{b}) is not a data edge")
    return CheckResult(True)
