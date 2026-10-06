"""Encoder inputs for queries and blocks.

The trainer, the torch and numpy encoders and the cascade all build node inputs
here, so a query and a block are always encoded from the same kind of vector:

  x(v) = [ base(v) , log1p(number of neighbours of v of type t) for t in types ]

  base(v) = [ intrinsic attributes (float32), type one-hot ]        in R0
  base(v) = [ type one-hot, centroid(sigma(v)) ]                     in R1 and R2-K

Neighbour counts are taken inside the graph being encoded (the query, or the
block's induced subgraph), never from the full data graph. For RGCN the relation
of a directed message u -> v is the ordered type pair (type(u), type(v)), mapped
to a dense id by relation_table() over the pairs present in the data graph. In a
CSR the entry (row v, column u) carries the message u -> v.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def type_onehot(types: np.ndarray, num_types: int) -> np.ndarray:
    types = np.asarray(types, dtype=np.int64)
    out = np.zeros((types.shape[0], num_types), dtype=np.float32)
    out[np.arange(types.shape[0]), types] = 1.0
    return out


def neighbor_type_counts(indptr: np.ndarray, indices: np.ndarray, types: np.ndarray,
                         num_types: int) -> np.ndarray:
    """[n, T] counts of neighbours of each type, within the given (local) graph."""
    indptr = np.asarray(indptr, dtype=np.int64)
    n = indptr.shape[0] - 1
    lens = np.diff(indptr)
    rows = np.repeat(np.arange(n, dtype=np.int64), lens)
    nbr_types = np.asarray(types, dtype=np.int64)[np.asarray(indices, dtype=np.int64)]
    counts = np.bincount(rows * num_types + nbr_types, minlength=n * num_types)
    return counts.reshape(n, num_types).astype(np.float32)


def relation_pairs_present(indptr: np.ndarray, indices: np.ndarray, types: np.ndarray,
                           num_types: int, chunk: int = 1 << 24) -> list[tuple[int, int]]:
    """Ordered (source type, target type) pairs that occur on data-graph edges."""
    indptr = np.asarray(indptr)
    n = indptr.shape[0] - 1
    seen = np.zeros(num_types * num_types, dtype=bool)
    types = np.asarray(types, dtype=np.int64)
    start = 0
    while start < n:
        stop = n
        # keep chunks bounded by adjacency entries
        lo = int(indptr[start])
        hi_target = lo + chunk
        stop = int(np.searchsorted(indptr, hi_target, side="right")) - 1
        stop = min(max(stop, start + 1), n)
        lens = np.diff(np.asarray(indptr[start:stop + 1], dtype=np.int64))
        src = np.repeat(types[start:stop], lens)
        dst = types[np.asarray(indices[int(indptr[start]):int(indptr[stop])], dtype=np.int64)]
        seen[np.unique(src * num_types + dst)] = True
        start = stop
    return [(int(k // num_types), int(k % num_types)) for k in np.flatnonzero(seen)]


def relation_table(pairs: list[tuple[int, int]] | list[list[int]], num_types: int) -> np.ndarray:
    """[T, T] int table: relation id of the ordered pair, -1 when absent."""
    table = np.full((num_types, num_types), -1, dtype=np.int64)
    for rid, (a, b) in enumerate(sorted((int(a), int(b)) for a, b in pairs)):
        table[a, b] = rid
    return table


def edge_relations(indptr: np.ndarray, indices: np.ndarray, types: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Relation id of the message carried by every CSR entry (neighbour -> row)."""
    indptr = np.asarray(indptr, dtype=np.int64)
    lens = np.diff(indptr)
    rows = np.repeat(np.arange(lens.shape[0], dtype=np.int64), lens)
    types = np.asarray(types, dtype=np.int64)
    rel = table[types[np.asarray(indices, dtype=np.int64)], types[rows]]
    if (rel < 0).any():
        raise ValueError("an edge carries a type pair absent from the relation table")
    return rel


def store_relation_table(store: Any) -> np.ndarray:
    """The dataset's relation table from meta.json (written by prepare_data.py)."""
    num_types = int(store.meta["num_types"])
    pairs = store.meta.get("relation_pairs")
    if pairs is None:
        pairs = relation_pairs_present(store.indptr, store.indices, store.node_type, num_types)
    return relation_table(pairs, num_types)


def graph_inputs(regime: str, indptr: np.ndarray, indices: np.ndarray, types: np.ndarray, num_types: int, *,
                 attrs: np.ndarray | None = None, codes: np.ndarray | None = None, sigma_table: Any = None,
                 rel_table: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Everything an encoder needs for one graph: x, CSR and per-entry relations."""
    out = {
        "x": encoder_inputs(regime, indptr, indices, types, num_types, attrs=attrs, codes=codes,
                            sigma_table=sigma_table),
        "indptr": np.asarray(indptr, dtype=np.int64),
        "indices": np.asarray(indices, dtype=np.int64),
    }
    if rel_table is not None:
        out["rel"] = edge_relations(indptr, indices, types, rel_table)
    return out


def store_graph_inputs(store: Any, graph: Any, regime: str, rel_table: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Inputs of a data-graph LocalGraph (a block or fine part) from raw store arrays."""
    gids = np.asarray(graph.global_ids, dtype=np.int64)
    types = np.asarray(store.node_type[gids], dtype=np.int64)
    num_types = int(store.meta["num_types"])
    if regime == "R0":
        return graph_inputs(regime, graph.indptr, graph.indices, types, num_types,
                            attrs=np.asarray(store.attrs[gids]), rel_table=rel_table)
    table = store.sigma(regime)
    return graph_inputs(regime, graph.indptr, graph.indices, types, num_types,
                        codes=np.asarray(table.codes[gids], dtype=np.int64), sigma_table=table, rel_table=rel_table)


def query_graph_inputs(store: Any, query: Any, regime: str, codes: np.ndarray,
                       rel_table: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Inputs of a query from its payload only (a query never reads data-graph rows).

    `query` has .indptr, .indices and .payload (types, and attrs in R0); `codes`
    are the query's sigma codes from SigmaTable.query_codes().
    """
    types = np.asarray(query.payload["types"], dtype=np.int64)
    num_types = int(store.meta["num_types"])
    if regime == "R0":
        return graph_inputs(regime, query.indptr, query.indices, types, num_types,
                            attrs=np.asarray(query.payload["attrs"]), rel_table=rel_table)
    return graph_inputs(regime, query.indptr, query.indices, types, num_types, codes=codes,
                        sigma_table=store.sigma(regime), rel_table=rel_table)


def base_inputs(regime: str, types: np.ndarray, num_types: int, *, attrs: np.ndarray | None = None,
                codes: np.ndarray | None = None, sigma_table: Any = None) -> np.ndarray:
    if regime == "R0":
        if attrs is None:
            raise ValueError("R0 inputs need attributes")
        return np.concatenate([np.asarray(attrs, dtype=np.float32), type_onehot(types, num_types)], axis=1)
    if sigma_table is None or codes is None:
        raise ValueError(f"{regime} inputs need sigma codes and the sigma table")
    return sigma_table.encoder_inputs(types, codes)


def encoder_inputs(regime: str, indptr: np.ndarray, indices: np.ndarray, types: np.ndarray,
                   num_types: int, *, attrs: np.ndarray | None = None, codes: np.ndarray | None = None,
                   sigma_table: Any = None) -> np.ndarray:
    """float32 [n, d_in] inputs for a graph given as local CSR plus per-node data."""
    base = base_inputs(regime, types, num_types, attrs=attrs, codes=codes, sigma_table=sigma_table)
    struct = np.log1p(neighbor_type_counts(indptr, indices, types, num_types))
    return np.concatenate([base, struct], axis=1).astype(np.float32, copy=False)


def input_dim(regime: str, attr_dim: int, num_types: int) -> int:
    if regime == "R0":
        return attr_dim + num_types + num_types
    return num_types + attr_dim + num_types
