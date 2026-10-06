"""Collision-free node labels (sigma) for the regimes in protocol.LABEL_REGIMES.

R0 (exact):    sigma(v) = code of (type(v), intrinsic attribute values of v)
R1 (semantic): sigma(v) = code of (type(v), class(v)), class -1 when absent
R2-K (hashed): sigma(v) = code of (type(v), h(R0 key of v) mod K)

Codes are assigned in order of first appearance over canonical node ids, so a
store always yields the same codes. R0 codes come from 128-bit BLAKE2b row
digests and are then verified value-for-value inside every digest group, so two
nodes share a code exactly when their keys are equal. Floats are compared by
value: -0.0 equals 0.0, and NaN attributes are rejected.

A query label that no data node carries receives a fresh code >= num_codes, so
it matches nothing. For R1 and R2 the encoder input of a node is
[type one-hot, centroid(sigma(v))], where the centroid is the mean intrinsic
attribute vector of the data nodes with that code; unknown codes get zeros.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

DIGEST_BYTES = 16
PERSON = b"jigsaw.sigma.v1"
CHUNK = 1 << 16


# ----------------------------------------------------------------------------
# Row digests
# ----------------------------------------------------------------------------
def canonical_rows(attrs: np.ndarray) -> np.ndarray:
    """Contiguous copy whose bytes are equal exactly when the values are equal."""
    arr = np.ascontiguousarray(np.asarray(attrs))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if np.issubdtype(arr.dtype, np.floating):
        if np.isnan(arr).any():
            raise ValueError("NaN attributes cannot be labelled exactly")
        arr = np.where(arr == 0, np.zeros((), dtype=arr.dtype), arr)
    return arr


def row_digest(node_type: int, row: np.ndarray) -> bytes:
    data = np.asarray([node_type], dtype="<i2").tobytes() + canonical_rows(row.reshape(1, -1)).tobytes()
    return hashlib.blake2b(data, digest_size=DIGEST_BYTES, person=PERSON).digest()


def row_digests(types: np.ndarray, attrs: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    """[N, 16] uint8 digests of (type, attribute row) for every node."""
    n = int(types.shape[0])
    out = np.empty((n, DIGEST_BYTES), dtype=np.uint8)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        tbytes = np.asarray(types[start:stop], dtype="<i2")
        rows = canonical_rows(np.asarray(attrs[start:stop]))
        raw = rows.view(np.uint8).reshape(stop - start, -1)
        for i in range(stop - start):
            h = hashlib.blake2b(digest_size=DIGEST_BYTES, person=PERSON)
            h.update(tbytes[i].tobytes())
            h.update(raw[i].tobytes())
            out[start + i] = np.frombuffer(h.digest(), dtype=np.uint8)
    return out


def _void(digests: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(digests).view(np.dtype((np.void, digests.shape[1]))).ravel()


def _codes_by_first_appearance(first_index: np.ndarray, inverse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map np.unique group ids to codes ordered by first appearance."""
    order = np.argsort(first_index, kind="stable")
    code_of_group = np.empty(order.shape[0], dtype=np.int64)
    code_of_group[order] = np.arange(order.shape[0], dtype=np.int64)
    return code_of_group[inverse].astype(np.int32), code_of_group


# ----------------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------------
@dataclass
class ExactBuild:
    codes: np.ndarray  # int32 [N]
    digests_sorted: np.ndarray  # uint8 [C, 16]
    digest_code: np.ndarray  # int32 [C] code of each sorted digest
    rep: np.ndarray  # int64 [C] representative node of each code
    df: np.ndarray  # int64 [C]
    split_groups: int  # digest groups whose rows were not all equal (expected 0)
    node_digests: np.ndarray  # uint8 [N, 16], kept for R2


def build_exact(types: np.ndarray, attrs: np.ndarray, chunk: int = CHUNK) -> ExactBuild:
    types = np.asarray(types)
    n = int(types.shape[0])
    digests = row_digests(types, attrs, chunk)
    dv = _void(digests)
    _, first_index, inverse = np.unique(dv, return_index=True, return_inverse=True)
    inverse = inverse.astype(np.int64).ravel()
    group_size = np.bincount(inverse)
    # Verify every non-representative member of a multi-member group against its representative.
    check = np.flatnonzero((group_size[inverse] > 1) & (first_index[inverse] != np.arange(n)))
    bad: list[int] = []
    for start in range(0, check.shape[0], chunk):
        rows = check[start:start + chunk]
        reps = first_index[inverse[rows]]
        same = np.asarray(types[rows]) == np.asarray(types[reps])
        a = canonical_rows(np.asarray(attrs[rows]))
        b = canonical_rows(np.asarray(attrs[reps]))
        same &= np.all(a == b, axis=1)
        bad.extend(rows[~same].tolist())
    split_groups = 0
    if bad:
        # A 128-bit digest collision: give each distinct key inside the group its own group id.
        next_group = int(inverse.max()) + 1
        new_first = list(first_index)
        by_key: dict[tuple[int, bytes], int] = {}
        for node in bad:
            key = (int(types[node]), canonical_rows(np.asarray(attrs[node]).reshape(1, -1)).tobytes())
            if key not in by_key:
                by_key[key] = next_group
                new_first.append(node)
                next_group += 1
            inverse[node] = by_key[key]
        first_index = np.asarray(new_first, dtype=np.int64)
        split_groups = len(by_key)
    codes, code_of_group = _codes_by_first_appearance(first_index, inverse)
    num_codes = int(code_of_group.shape[0])
    rep = np.empty(num_codes, dtype=np.int64)
    rep[code_of_group] = first_index
    node_digest_of_code = digests[rep]
    sort_order = np.argsort(_void(node_digest_of_code), kind="stable")
    df = np.bincount(codes, minlength=num_codes).astype(np.int64)
    return ExactBuild(
        codes=codes,
        digests_sorted=node_digest_of_code[sort_order],
        digest_code=sort_order.astype(np.int32),
        rep=rep,
        df=df,
        split_groups=split_groups,
        node_digests=digests,
    )


@dataclass
class KeyBuild:
    codes: np.ndarray  # int32 [N]
    keys: np.ndarray  # int64 [C, 2] (type, value) in code order
    df: np.ndarray  # int64 [C]


def build_keyed(types: np.ndarray, values: np.ndarray) -> KeyBuild:
    """Codes for integer keys (type, value), value >= -1."""
    types = np.asarray(types, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)
    if values.min(initial=0) < -1:
        raise ValueError("key values must be >= -1")
    span = int(values.max(initial=0)) + 2
    combined = types * span + (values + 1)
    _, first_index, inverse = np.unique(combined, return_index=True, return_inverse=True)
    codes, code_of_group = _codes_by_first_appearance(first_index, inverse.ravel())
    num_codes = int(code_of_group.shape[0])
    rep = np.empty(num_codes, dtype=np.int64)
    rep[code_of_group] = first_index
    keys = np.stack([types[rep], values[rep]], axis=1)
    df = np.bincount(codes, minlength=num_codes).astype(np.int64)
    return KeyBuild(codes=codes, keys=keys, df=df)


def hashed_buckets(node_digests: np.ndarray, buckets: int) -> np.ndarray:
    """Bucket = first 8 digest bytes (big-endian) mod K."""
    head = np.ascontiguousarray(node_digests[:, :8]).view(">u8").ravel()
    return (head % np.uint64(buckets)).astype(np.int64)


def centroids(codes: np.ndarray, attrs: np.ndarray, num_codes: int, chunk: int = CHUNK) -> np.ndarray:
    """float32 [C, d]: mean attribute vector of the nodes carrying each code."""
    from scipy import sparse

    n = int(codes.shape[0])
    d = int(np.asarray(attrs[:1]).reshape(1, -1).shape[1])
    sums = np.zeros((num_codes, d), dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        block = np.asarray(attrs[start:stop], dtype=np.float64).reshape(stop - start, d)
        onehot = sparse.csr_matrix(
            (np.ones(stop - start), (np.asarray(codes[start:stop], dtype=np.int64), np.arange(stop - start))),
            shape=(num_codes, stop - start),
        )
        sums += onehot @ block
    counts = np.bincount(np.asarray(codes, dtype=np.int64), minlength=num_codes).astype(np.float64)
    return (sums / np.maximum(counts, 1.0)[:, None]).astype(np.float32)


# ----------------------------------------------------------------------------
# Writing all regimes of a store
# ----------------------------------------------------------------------------
def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array), allow_pickle=False)


def _df_summary(df: np.ndarray) -> dict[str, Any]:
    return {
        "num_codes": int(df.shape[0]),
        "singleton_codes": int((df == 1).sum()),
        "max_df": int(df.max(initial=0)),
        "median_df": float(np.median(df)) if df.shape[0] else 0.0,
    }


def build_store_regimes(store_dir: str | Path, regimes: list[str], *, chunk: int = CHUNK,
                        with_centroids: bool = True) -> dict[str, Any]:
    """Compute and write sigma/<regime>/ for every requested regime of a store."""
    from . import protocol

    store_dir = Path(store_dir)
    graph = store_dir / "graph"
    types = np.load(graph / "node_type.npy")
    attrs = np.load(graph / "attrs.npy", mmap_mode="r")
    class_path = graph / "class.npy"
    report: dict[str, Any] = {}
    exact: ExactBuild | None = None
    need_exact = any(r == "R0" or r.startswith("R2") for r in regimes)
    if need_exact:
        exact = build_exact(types, attrs, chunk)
    for regime in regimes:
        spec = protocol.LABEL_REGIMES[regime]
        out = store_dir / "sigma" / regime
        info: dict[str, Any] = {"regime": regime, "kind": spec["kind"]}
        if spec["kind"] == "exact":
            assert exact is not None
            _save(out / "codes.npy", exact.codes)
            _save(out / "digests_sorted.npy", exact.digests_sorted)
            _save(out / "digest_code.npy", exact.digest_code)
            _save(out / "rep.npy", exact.rep)
            _save(out / "df.npy", exact.df)
            info["digest_groups_split"] = exact.split_groups
            info.update(_df_summary(exact.df))
        else:
            if spec["kind"] == "semantic":
                if not class_path.exists():
                    raise FileNotFoundError(f"{regime} needs graph/class.npy")
                values = np.load(class_path).astype(np.int64)
            else:
                assert exact is not None
                values = hashed_buckets(exact.node_digests, int(spec["buckets"]))
                info["buckets"] = int(spec["buckets"])
            built = build_keyed(types, values)
            _save(out / "codes.npy", built.codes)
            _save(out / "keys.npy", built.keys)
            _save(out / "df.npy", built.df)
            info.update(_df_summary(built.df))
            if with_centroids:
                _save(out / "centroids.npy", centroids(built.codes, attrs, int(built.df.shape[0]), chunk))
        info["num_types"] = int(types.max()) + 1
        (out / "sigma.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report[regime] = info
    return report


# ----------------------------------------------------------------------------
# Loader and query-side lookup
# ----------------------------------------------------------------------------
class SigmaTable:
    def __init__(self, directory: Path, info: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        self.dir = directory
        self.info = info
        self.regime = info["regime"]
        self.kind = info["kind"]
        self.codes = arrays["codes"]
        self.df = np.asarray(arrays["df"])
        self.num_codes = int(self.df.shape[0])
        self.num_types = int(info.get("num_types", 1))
        self._arrays = arrays
        self.centroids = arrays.get("centroids")
        if self.kind == "exact":
            self.digests_sorted = arrays["digests_sorted"]
            self.digest_code = np.asarray(arrays["digest_code"])
            self.rep = np.asarray(arrays["rep"])
            self._digest_void = _void(np.asarray(self.digests_sorted))
        else:
            keys = np.asarray(arrays["keys"], dtype=np.int64)
            self.keys = keys
            self._span = int(keys[:, 1].max(initial=0)) + 2
            combined = keys[:, 0] * self._span + (keys[:, 1] + 1)
            self._key_order = np.argsort(combined, kind="stable")
            self._key_sorted = combined[self._key_order]

    @classmethod
    def load(cls, directory: str | Path, *, mmap: bool = False) -> "SigmaTable":
        directory = Path(directory)
        info = json.loads((directory / "sigma.json").read_text(encoding="utf-8"))
        mode = "r" if mmap else None
        arrays: dict[str, np.ndarray] = {}
        for path in directory.glob("*.npy"):
            big = path.stem in ("codes", "centroids")
            arrays[path.stem] = np.load(path, mmap_mode=mode if big else None)
        return cls(directory, info, arrays)

    def open_arrays(self) -> list[np.ndarray]:
        return [a for a in self._arrays.values() if isinstance(a, np.memmap)]

    # -------------------------------------------------------------- lookups
    def lookup_exact(self, types: np.ndarray, attrs: np.ndarray, store_types: np.ndarray,
                     store_attrs: np.ndarray) -> np.ndarray:
        """Codes of query rows; rows absent from the data get fresh codes >= num_codes."""
        types = np.asarray(types)
        digests = row_digests(types, np.asarray(attrs))
        dv = _void(digests)
        pos = np.searchsorted(self._digest_void, dv)
        pos_c = np.minimum(pos, self._digest_void.shape[0] - 1)
        hit = self._digest_void[pos_c] == dv
        codes = np.full(types.shape[0], -1, dtype=np.int64)
        query_rows = canonical_rows(np.asarray(attrs))
        for i in np.flatnonzero(hit):
            # Scan every code with this digest (more than one only after a digest collision).
            p = int(pos_c[i])
            while p < self._digest_void.shape[0] and self._digest_void[p] == dv[i]:
                code = int(self.digest_code[p])
                rep = int(self.rep[code])
                if int(store_types[rep]) == int(types[i]) and np.array_equal(
                    canonical_rows(np.asarray(store_attrs[rep]).reshape(1, -1))[0], query_rows[i]
                ):
                    codes[i] = code
                    break
                p += 1
        return self._fill_fresh(codes, [bytes(d) for d in digests])

    def lookup_keys(self, types: np.ndarray, values: np.ndarray) -> np.ndarray:
        types = np.asarray(types, dtype=np.int64)
        values = np.asarray(values, dtype=np.int64)
        combined = types * self._span + (values + 1)
        valid = (values >= -1) & (values + 1 < self._span)
        pos = np.minimum(np.searchsorted(self._key_sorted, combined), self._key_sorted.shape[0] - 1)
        hit = valid & (self._key_sorted[pos] == combined)
        codes = np.where(hit, self._key_order[pos], -1).astype(np.int64)
        return self._fill_fresh(codes, [(int(t), int(v)) for t, v in zip(types, values)])

    def _fill_fresh(self, codes: np.ndarray, keys: list[Any]) -> np.ndarray:
        fresh: dict[Any, int] = {}
        for i in np.flatnonzero(codes < 0):
            codes[i] = fresh.setdefault(keys[i], self.num_codes + len(fresh))
        return codes.astype(np.int64)

    def query_codes(self, payload: Mapping[str, Any], store: Any) -> np.ndarray:
        """Codes for a query payload (see queries.py for the payload fields)."""
        types = np.asarray(payload["types"])
        if self.kind == "exact":
            return self.lookup_exact(types, np.asarray(payload["attrs"]), store.node_type, store.attrs)
        if self.kind == "semantic":
            return self.lookup_keys(types, np.asarray(payload["classes"]))
        return self.lookup_keys(types, np.asarray(payload["buckets"]))

    def label_frequency(self, codes: np.ndarray) -> np.ndarray:
        """df of each code; fresh codes have frequency 0."""
        codes = np.asarray(codes, dtype=np.int64)
        out = np.zeros(codes.shape[0], dtype=np.int64)
        known = codes < self.num_codes
        out[known] = self.df[codes[known]]
        return out

    def encoder_inputs(self, types: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """[type one-hot, centroid(code)] for coarse regimes; zeros for fresh codes."""
        if self.centroids is None:
            raise ValueError(f"{self.regime} has no centroids")
        types = np.asarray(types, dtype=np.int64)
        codes = np.asarray(codes, dtype=np.int64)
        onehot = np.zeros((types.shape[0], self.num_types), dtype=np.float32)
        onehot[np.arange(types.shape[0]), types] = 1.0
        cent = np.zeros((codes.shape[0], self.centroids.shape[1]), dtype=np.float32)
        known = codes < self.num_codes
        if known.any():
            cent[known] = np.asarray(self.centroids[codes[known]], dtype=np.float32)
        return np.concatenate([onehot, cent], axis=1)
