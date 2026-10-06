"""Prepare the canonical store of one dataset: load, normalise, partition, relabel, write.

    python -m scripts.canonical.prepare_data --dataset arxiv --data-root data \\
        --store-root "$JIGSAW_STORE_ROOT" [--regimes R0 R1] [--alternatives] \\
        [--import-partition PATH] [--skip-sigma] [--force]

Stages run inside one RunManifest (<store>/_prepare_run/manifest.json). Each
stage writes a done-marker (_prepare_run/stages/<stage>.json) keyed by its
configuration and its upstream keys; a rerun skips finished stages, a rerun
with nothing left to do touches no file (not even the manifest), and --force
redoes everything. Code changes do not invalidate markers: use --force.

  load          offline source files only (nothing is downloaded), then the graph
                normalisation of protocol.GRAPH_NORMALIZATION: symmetrise, drop
                self loops, merge duplicates, int64 indptr, int32 indices
  partition     primary METIS k-way partition: P = DATASETS[d]['partitions'],
                PARTITIONING seed and ufactor (or --import-partition). Parts
                METIS leaves empty are refilled by partition.EMPTY_PART_RULE
                (ogbn-mag: 1 of 2000 with pymetis 2025.2.1), so the store has
                exactly P blocks
  fine          METIS inside every block, fine_per_block parts
  relabel       canonical ids sorted by (block, fine, source id)
  store         store.write_store, then relation pairs of the final graph in meta.json
  sigma         sigma.build_store_regimes(protocol.regimes_for(d))
  alternatives  kahip_fast (when kaffpa exists), bfs_random, metis_x0.5 and metis_x2
                on the stored graph, each with a METIS-within-block fine level
  report        <store>/prepare_report.json

Intermediate arrays live in <store>/_prepare_work/ until the store is written.

Loaders (source id = type offset + local id, types in schema order):
  corafull    CoraFull/cora/raw/cora.npz (its pickled members are never loaded;
              attributes stored as word presence, x > 0 -> 1, uint8)
  arxiv       ogbn_arxiv/ogbn_arxiv/raw/*.csv.gz
  mag         ogbn_mag/ogbn_mag/raw/ (paper, author, institution, field_of_study;
              128 paper features, zeros for other types; class = venue of papers)
  products    ogbn_products/ogbn_products/raw/*.csv.gz (not run yet)
  papers100m  ogbn_papers100M/ogbn_papers100M/raw/{data.npz, node-label.npz},
              out of core (not run yet)

Peak memory of the whole process tree including the pymetis worker (memprobe
over prepare_all.sh; prepare_report.json has the per-stage peaks and the run
manifest the OS peak commit). Measured with pymetis 2025.2.1 on a 16 GB Windows
laptop whose RAM was mostly taken by other jobs, so the working set may have
been trimmed; the commit peaks are given where they are larger:
  corafull    1.5 GiB (commit 1.75 GiB), 31 s. The peak is in sigma; loading
              densifies cora.npz as float32 (0.64 GiB) before binarising it to
              uint8 word presence (0.16 GiB). Float32 builds: 2.0 GiB, 58-82 s
  arxiv       0.4 GiB, 46-53 s over two builds (main process commit 0.7 GiB,
              METIS worker 0.5 GiB)
  mag         at most 4.6 GiB of commit: 4.1 GiB in the METIS worker plus
              0.5 GiB in the driver. The driver alone peaks at 3.6 GiB while
              loading and 4.1 GiB in the later stages; working set 3.6 GiB.
              57 min here (load 15, METIS 19, store 16); load and METIS kept
              the CPU busy about a third of their wall time, paging the rest
  products    about 10 GB (estimate for the in-RAM path, 124M adjacency entries)
  papers100m  out of core. Loading streams data.npz into memmaps and builds the
              CSR by a chunked counting sort (about 5 GB plus three [N] int64
              vectors, 2.7 GB). METIS dominates: pymetis or gpmetis built with
              64-bit idx_t holds xadj/adjncy/adjwgt of 3.2 billion adjacency
              entries plus the coarsening hierarchy, roughly 100-150 GB. On a
              smaller machine partition elsewhere and pass --import-partition.
              The other stages stay near 10 GB of RSS plus page cache. Disk:
              work files peak near 70 GB (edge_index 26 GB while the CSR is
              built, then attributes 57 GB and CSR 13 GB) and the store takes
              about 80 GB, so keep about 150 GB free besides the raw files.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import features, manifest, ooc, protocol, sigma
from . import partition as part_mod
from . import store as store_mod

PREPARE_VERSION = 1
STAGES = ("load", "partition", "fine", "relabel", "store", "sigma", "alternatives", "report")
DEPENDS = {
    "load": (),
    "partition": ("load",),
    "fine": ("partition",),
    "relabel": ("fine",),
    "store": ("relabel",),
    "sigma": ("store",),
    "alternatives": ("store",),
    "report": ("store", "sigma", "alternatives"),
}
NEEDS = {  # stages whose intermediate files a stage reads
    "partition": ("load",),
    "fine": ("load", "partition"),
    "relabel": ("partition", "fine"),
    "store": ("load", "partition", "fine", "relabel"),
}
WORK_FILES = {
    "load": ("indptr.npy", "indices.npy", "node_type.npy", "attrs.npy", "load.json"),
    "partition": ("node_block.npy", "partition.json"),
    "fine": ("node_fine.npy", "fine.json"),
    "relabel": ("new_to_old.npy",),
}
ALTERNATIVES = ("kahip_fast", "bfs_random", "metis_x0.5", "metis_x2")
OUT_OF_CORE = ("papers100m",)
RUN_DIR = "_prepare_run"
WORK_DIR = "_prepare_work"
REPORT_NAME = "prepare_report.json"
STORE_OUTPUTS = ("meta.json", REPORT_NAME, "graph", "part", "sigma")
MAG_RELATIONS = (("author", "affiliated_with", "institution"), ("author", "writes", "paper"),
                 ("paper", "cites", "paper"), ("paper", "has_topic", "field_of_study"))
CLASS_SEMANTICS = {
    "corafull": "class",
    "arxiv": "class (arXiv subject area)",
    "mag": "venue for papers, -1 for other types",
    "products": "class (product category)",
    "papers100m": "arXiv subject area for labelled papers, -1 otherwise (no R1 regime)",
}
FEATURE_ROWS_PER_CHUNK = 100_000


def log(dataset: str, message: str) -> None:
    stamp = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp} prepare {dataset}] {message}", file=sys.stderr, flush=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=manifest._json_default) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _key(payload: Any) -> str:
    return hashlib.sha256(protocol.canonical_json(payload).encode("ascii")).hexdigest()


# ----------------------------------------------------------------------------
# Source files
# ----------------------------------------------------------------------------
def ogb_raw_dir(data_root: Path, name: str) -> Path:
    """<root>/<name>/<name>/raw as PyG's OGB datasets lay it out ('-' becomes '_')."""
    folder = name.replace("-", "_")
    for candidate in (data_root / folder / folder / "raw", data_root / folder / "raw"):
        if candidate.is_dir():
            return candidate
    return data_root / folder / folder / "raw"


def _require(dataset: str, files: dict[str, Path]) -> dict[str, Path]:
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{dataset}: missing source file(s): {', '.join(missing)}. prepare_data reads "
                                "offline copies only; put the dataset files there first (see "
                                "scripts/canonical/launch/README.md)")
    return files


def _read_triplets(path: Path) -> list[tuple[str, str, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [line.strip().split(",") for line in handle if line.strip()]
    if not rows or any(len(row) != 3 for row in rows):
        raise ValueError(f"{path}: expected 'source,relation,target' rows")
    return [(a, b, c) for a, b, c in rows]


def source_files(dataset: str, data_root: str | os.PathLike) -> dict[str, Path]:
    """Every raw file a loader reads, by name; raises FileNotFoundError naming missing ones."""
    root = Path(data_root)
    if dataset == "corafull":
        return _require(dataset, {"cora.npz": root / "CoraFull" / "cora" / "raw" / "cora.npz"})
    if dataset in ("arxiv", "products"):
        raw = ogb_raw_dir(root, f"ogbn-{dataset}")
        names = ("num-node-list", "num-edge-list", "edge", "node-feat", "node-label")
        return _require(dataset, {f"{name}.csv.gz": raw / f"{name}.csv.gz" for name in names})
    if dataset == "mag":
        raw = ogb_raw_dir(root, "ogbn-mag")
        files = _require(dataset, {"num-node-dict.csv.gz": raw / "num-node-dict.csv.gz",
                                   "triplet-type-list.csv.gz": raw / "triplet-type-list.csv.gz"})
        for s, r, t in _read_triplets(files["triplet-type-list.csv.gz"]):
            rel = f"{s}___{r}___{t}"
            for name in ("edge.csv.gz", "num-edge-list.csv.gz"):
                files[f"relations/{rel}/{name}"] = raw / "relations" / rel / name
        files["node-feat/paper/node-feat.csv.gz"] = raw / "node-feat" / "paper" / "node-feat.csv.gz"
        files["node-label/paper/node-label.csv.gz"] = raw / "node-label" / "paper" / "node-label.csv.gz"
        return _require(dataset, files)
    if dataset == "papers100m":
        raw = ogb_raw_dir(root, "ogbn-papers100M")
        return _require(dataset, {"data.npz": raw / "data.npz", "node-label.npz": raw / "node-label.npz"})
    raise ValueError(f"unknown dataset {dataset!r}")


def _fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


# ----------------------------------------------------------------------------
# CSV readers (the OGB conventions: pandas, header=None, float64 then float32)
# ----------------------------------------------------------------------------
def _csv(path: Path, **kwargs: Any) -> Any:
    import pandas as pd

    return pd.read_csv(path, compression="gzip", header=None, **kwargs)


def read_single_int(path: Path) -> int:
    return int(np.asarray(_csv(path).values).reshape(-1)[0])


def read_edge_csv(path: Path) -> np.ndarray:
    """[E, 2] int64 from an OGB edge.csv.gz ('source,target' per line)."""
    edges = np.asarray(_csv(path, dtype=np.int64).values)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"{path}: expected two columns, got shape {edges.shape}")
    return edges


def csv_columns(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return len(handle.readline().strip().split(","))


def read_float_csv_into(path: Path, out: np.ndarray, *, chunk_rows: int = FEATURE_ROWS_PER_CHUNK) -> int:
    """Parse a float CSV into out[0:rows] as OGB does (pandas float64, then float32)."""
    pos = 0
    for chunk in _csv(path, chunksize=chunk_rows):
        values = np.asarray(chunk.values)
        if values.shape[1] != out.shape[1]:
            raise ValueError(f"{path}: {values.shape[1]} columns, expected {out.shape[1]}")
        if pos + values.shape[0] > out.shape[0]:
            raise ValueError(f"{path}: more than {out.shape[0]} rows")
        out[pos:pos + values.shape[0]] = values.astype(np.float32)
        pos += values.shape[0]
    return pos


def read_labels(path: Path, num_rows: int) -> np.ndarray:
    """int32 labels, -1 where the file has no label (NaN)."""
    values = np.asarray(_csv(path).values).reshape(-1)
    if values.shape[0] != num_rows:
        raise ValueError(f"{path}: {values.shape[0]} labels for {num_rows} nodes")
    return labels_to_int32(values, str(path))


def labels_to_int32(values: np.ndarray, source: str) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.int32)
    missing = np.isnan(values)
    known = values[~missing]
    if known.shape[0] and not np.array_equal(known, np.round(known)):
        raise ValueError(f"{source}: non-integer labels")
    out = np.full(values.shape[0], -1, dtype=np.int32)
    out[~missing] = known.astype(np.int64)
    return out


# ----------------------------------------------------------------------------
# Loaders (in RAM)
# ----------------------------------------------------------------------------
@dataclass
class Loaded:
    num_nodes: int
    src: np.ndarray
    dst: np.ndarray
    node_type: np.ndarray
    attrs: np.ndarray
    classes: np.ndarray | None
    type_names: list[str]
    type_counts: list[int]
    notes: dict[str, Any] = field(default_factory=dict)
    attribute_notes: dict[str, Any] = field(default_factory=dict)


def load_corafull(files: dict[str, Path]) -> Loaded:
    """CoraFull from cora.npz (directed citations, sparse attributes, 70 classes)."""
    from scipy import sparse

    path = files["cora.npz"]
    with np.load(path, allow_pickle=False) as npz:  # idx_to_* are pickled object arrays: never read
        adj_data, adj_indices, adj_indptr = npz["adj_data"], npz["adj_indices"], npz["adj_indptr"]
        adj_shape = tuple(int(s) for s in npz["adj_shape"])
        attr = sparse.csr_matrix((npz["attr_data"], npz["attr_indices"], npz["attr_indptr"]),
                                 shape=tuple(int(s) for s in npz["attr_shape"]))
        labels = np.asarray(npz["labels"])
    n = adj_shape[0]
    if adj_shape != (n, n) or attr.shape[0] != n or labels.shape[0] != n:
        raise ValueError(f"{path}: inconsistent shapes adj {adj_shape}, attr {attr.shape}, labels {labels.shape}")
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(adj_indptr.astype(np.int64)))
    edge = adj_data != 0
    attrs = attr.toarray()  # sums duplicate entries, as scipy (and PyG) read the matrix
    norms = np.empty(n, dtype=np.float64)
    parts = []
    for a in range(0, n, 1024):  # statistics of the dense values, 1024 rows at a time
        block = attrs[a:a + 1024]
        wide = block.astype(np.float64)
        norms[a:a + 1024] = np.sqrt(np.einsum("ij,ij->i", wide, wide))
        parts.append(block[block != 0])
    nonzero = np.concatenate(parts) if parts else np.zeros(0, dtype=attrs.dtype)
    binary = bool(np.all(nonzero == 1))
    notes = {
        "schema_dtype": protocol.ATTRIBUTE_SCHEMAS["corafull_bow_v1"]["dtype"],
        "attr_values_binary": binary,
        "attr_nonzero_entries": int(nonzero.shape[0]),
        "attr_distinct_nonzero_values": int(np.unique(nonzero).shape[0]),
        "attr_nonzero_min": float(nonzero.min()) if nonzero.shape[0] else 0.0,
        "attr_nonzero_max": float(nonzero.max()) if nonzero.shape[0] else 0.0,
        "attr_row_l2_norm_min": float(norms.min()),
        "attr_row_l2_norm_max": float(norms.max()),
        "attr_empty_rows": int(np.count_nonzero(norms == 0)),
    }
    if nonzero.shape[0] and float(nonzero.min()) < 0:
        raise ValueError(f"{path}: negative attribute values; corafull_bow_v1 stores word presence")
    if not binary:
        notes["attr_note"] = ("cora.npz holds L2-normalised term weights; corafull_bow_v1 stores word presence "
                              "(x > 0 -> 1), as PyG's CitationFull('cora') and the legacy pipeline read them")
    attrs = (attrs > 0).astype(np.uint8)
    notes["stored_dtype"] = str(attrs.dtype)
    return Loaded(
        num_nodes=n,
        src=rows[edge],
        dst=adj_indices.astype(np.int64)[edge],
        node_type=np.zeros(n, dtype=np.int8),
        attrs=attrs,
        classes=labels.astype(np.int32),
        type_names=list(protocol.ATTRIBUTE_SCHEMAS["corafull_bow_v1"]["node_types"]),
        type_counts=[n],
        notes={"adjacency_entries": int(adj_data.shape[0]), "adjacency_explicit_zeros": int((~edge).sum()),
               "adjacency_directed": True},
        attribute_notes=notes,
    )


def load_ogb_homogeneous(files: dict[str, Path], dataset: str) -> Loaded:
    """ogbn-arxiv and ogbn-products: one node type, CSV edges, features and labels."""
    n = read_single_int(files["num-node-list.csv.gz"])
    m = read_single_int(files["num-edge-list.csv.gz"])
    edges = read_edge_csv(files["edge.csv.gz"])
    if edges.shape[0] != m:
        raise ValueError(f"{files['edge.csv.gz']}: {edges.shape[0]} edges, num-edge-list says {m}")
    feat_path = files["node-feat.csv.gz"]
    attrs = np.empty((n, csv_columns(feat_path)), dtype=np.float32)
    rows = read_float_csv_into(feat_path, attrs)
    if rows != n:
        raise ValueError(f"{feat_path}: {rows} rows for {n} nodes")
    schema = protocol.ATTRIBUTE_SCHEMAS[protocol.DATASETS[dataset]["schema"]]
    return Loaded(
        num_nodes=n,
        src=edges[:, 0],
        dst=edges[:, 1],
        node_type=np.zeros(n, dtype=np.int8),
        attrs=attrs,
        classes=read_labels(files["node-label.csv.gz"], n),
        type_names=list(schema["node_types"]),
        type_counts=[n],
        notes={"edges_directed_in_source": True},
    )


def load_mag(files: dict[str, Path]) -> Loaded:
    """ogbn-mag as one typed graph: global id = type offset + local id, schema type order."""
    import pandas as pd

    schema = protocol.ATTRIBUTE_SCHEMAS["mag_intrinsic_v1"]
    type_names = list(schema["node_types"])
    header = pd.read_csv(files["num-node-dict.csv.gz"], compression="gzip")
    counts_by_name = {str(name): int(header[name].iloc[0]) for name in header.columns}
    if sorted(counts_by_name) != sorted(type_names):
        raise ValueError(f"num-node-dict types {sorted(counts_by_name)} differ from the schema {type_names}")
    counts = np.asarray([counts_by_name[name] for name in type_names], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    n = int(counts.sum())
    src_parts, dst_parts, relations = [], [], {}
    for s, r, t in _read_triplets(files["triplet-type-list.csv.gz"]):
        rel = f"{s}___{r}___{t}"
        edges = read_edge_csv(files[f"relations/{rel}/edge.csv.gz"])
        expected = read_single_int(files[f"relations/{rel}/num-edge-list.csv.gz"])
        if edges.shape[0] != expected:
            raise ValueError(f"{rel}: {edges.shape[0]} edges, num-edge-list says {expected}")
        si, ti = type_names.index(s), type_names.index(t)
        if edges.shape[0] and (edges.min() < 0 or edges[:, 0].max() >= counts[si] or edges[:, 1].max() >= counts[ti]):
            raise ValueError(f"{rel}: local ids outside the node counts")
        src_parts.append(edges[:, 0] + offsets[si])
        dst_parts.append(edges[:, 1] + offsets[ti])
        relations[rel] = int(edges.shape[0])
    paper = type_names.index("paper")
    lo, hi = int(offsets[paper]), int(offsets[paper] + counts[paper])
    feat_path = files["node-feat/paper/node-feat.csv.gz"]
    attrs = np.zeros((n, csv_columns(feat_path)), dtype=np.float32)
    rows = read_float_csv_into(feat_path, attrs[lo:hi])
    if rows != hi - lo:
        raise ValueError(f"{feat_path}: {rows} rows for {hi - lo} papers")
    classes = np.full(n, -1, dtype=np.int32)
    classes[lo:hi] = read_labels(files["node-label/paper/node-label.csv.gz"], hi - lo)
    return Loaded(
        num_nodes=n,
        src=np.concatenate(src_parts),
        dst=np.concatenate(dst_parts),
        node_type=np.repeat(np.arange(len(type_names), dtype=np.int8), counts),
        attrs=attrs,
        classes=classes,
        type_names=type_names,
        type_counts=[int(c) for c in counts],
        notes={"type_offsets": [int(o) for o in offsets], "relations": relations,
               "non_paper_attrs": "zeros", "dropped_from_legacy": schema.get("dropped_from_legacy")},
    )


LOADERS: dict[str, Callable[[dict[str, Path]], Loaded]] = {
    "corafull": load_corafull,
    "arxiv": lambda files: load_ogb_homogeneous(files, "arxiv"),
    "mag": load_mag,
    "products": lambda files: load_ogb_homogeneous(files, "products"),
}


# ----------------------------------------------------------------------------
# Normalisation (in RAM; papers100M uses ooc.csr_from_edge_index)
# ----------------------------------------------------------------------------
def csr_from_pairs(lo: np.ndarray, hi: np.ndarray, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric CSR with sorted rows from unique pairs lo < hi sorted by (lo, hi)."""
    n = int(num_nodes)
    m = int(lo.shape[0])
    below = np.bincount(hi, minlength=n).astype(np.int64)  # neighbours smaller than the row
    above = np.bincount(lo, minlength=n).astype(np.int64)  # neighbours larger than the row
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(below + above, out=indptr[1:])
    indices = np.empty(2 * m, dtype=np.int32 if n < 2**31 else np.int64)
    rank = np.arange(m, dtype=np.int64) - np.repeat(np.cumsum(above) - above, above)
    indices[indptr[lo] + below[lo] + rank] = hi
    order = np.argsort(hi, kind="stable")
    rank = np.arange(m, dtype=np.int64) - np.repeat(np.cumsum(below) - below, below)
    indices[indptr[hi[order]] + rank] = lo[order]
    return indptr, indices


def normalize_edges(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """GRAPH_NORMALIZATION: the CSR of store.csr_from_edges and a count of what was removed."""
    n = int(num_nodes)
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    if src.shape != dst.shape:
        raise ValueError("src and dst differ in length")
    if src.shape[0] and (min(int(src.min()), int(dst.min())) < 0 or max(int(src.max()), int(dst.max())) >= n):
        raise ValueError(f"edge endpoints outside [0, {n})")
    loop = src == dst
    s, d = src[~loop], dst[~loop]
    directed = int(np.unique(s * n + d).shape[0])
    keys = np.unique(np.minimum(s, d) * n + np.maximum(s, d))
    del s, d
    lo = keys // n
    hi = keys - lo * n
    del keys
    indptr, indices = csr_from_pairs(lo, hi, n)
    nonloop = int(src.shape[0] - int(loop.sum()))
    m = int(lo.shape[0])
    report = {
        "input_edges": int(src.shape[0]),
        "self_loops_dropped": int(loop.sum()),
        "duplicates_merged": nonloop - m,
        "exact_duplicates": nonloop - directed,
        "reciprocal_pairs_merged": directed - m,
        "undirected_edges": m,
        "adjacency_entries": 2 * m,
        "isolated_nodes": int(np.count_nonzero(np.diff(indptr) == 0)),
    }
    return indptr, indices, report


# ----------------------------------------------------------------------------
# Context, stage keys and the plan
# ----------------------------------------------------------------------------
@dataclass
class Context:
    dataset: str
    data_root: Path
    store_root: Path
    sources: dict[str, Path]
    regimes: list[str]
    partitions: int
    fine_per_block: int
    tool: str = "auto"
    binary: str | Sequence[str] | None = None
    timeout: float | None = None
    import_partition: Path | None = None
    trusted_pickle: bool = False
    alternatives: bool = False
    skip_sigma: bool = False
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> dict[str, Any]:
        return protocol.DATASETS[self.dataset]

    @property
    def store_dir(self) -> Path:
        return self.store_root / self.dataset

    @property
    def run_dir(self) -> Path:
        return self.store_dir / RUN_DIR

    @property
    def work(self) -> Path:
        return self.store_dir / WORK_DIR

    @property
    def ooc(self) -> bool:
        return self.dataset in OUT_OF_CORE

    def stages(self) -> list[str]:
        skip = set()
        if self.skip_sigma:
            skip.add("sigma")
        if not self.alternatives:
            skip.add("alternatives")
        return [stage for stage in STAGES if stage not in skip]

    def marker(self, stage: str) -> Path:
        return self.run_dir / "stages" / f"{stage}.json"

    def log(self, message: str) -> None:
        log(self.dataset, message)


def metis_identity(tool: str, binary: str | Sequence[str] | None) -> dict[str, Any]:
    name, cmd = part_mod.resolve_metis_tool(tool, binary)
    if name == "gpmetis":
        assert cmd is not None
        return {"tool": "gpmetis", **part_mod._binary_info(cmd)}
    import importlib.metadata as md

    return {"tool": "pymetis", "version": md.version("pymetis")}


def kahip_identity() -> dict[str, Any]:
    try:
        cmd = part_mod._command(None, "KAHIP_BIN", "kaffpa")
    except part_mod.PartitionerUnavailable as exc:
        return {"available": False, "reason": str(exc)}
    return {"available": False} if cmd is None else {"available": True, **part_mod._binary_info(cmd)}


def stage_configs(ctx: Context) -> dict[str, dict[str, Any]]:
    schema_name = ctx.spec["schema"]
    metis = metis_identity(ctx.tool, ctx.binary)
    configs: dict[str, dict[str, Any]] = {
        "load": {
            "prepare_version": PREPARE_VERSION,
            "dataset": ctx.dataset,
            "loader": ctx.spec["loader"],
            "schema": {schema_name: protocol.ATTRIBUTE_SCHEMAS[schema_name]},
            "normalization": protocol.GRAPH_NORMALIZATION,
            "sources": {name: _fingerprint(path) for name, path in sorted(ctx.sources.items())},
        },
        "partition": {
            "partitioning": {k: protocol.PARTITIONING[k] for k in ("tool", "objective", "ufactor", "seed")},
            "partitions": ctx.partitions,
            "import": None if ctx.import_partition is None else part_mod.file_sha256(ctx.import_partition),
            "metis": None if ctx.import_partition is not None else metis,
            "empty_parts": None if ctx.import_partition is not None else part_mod.EMPTY_PART_RULE,
        },
        "fine": {"fine_level": protocol.PARTITIONING["fine_level"], "fine_per_block": ctx.fine_per_block,
                 "metis": metis},
        "relabel": {"node_order": "block, fine, source id"},
        "store": {"store_version": store_mod.STORE_VERSION},
        "sigma": {"regimes": {r: protocol.LABEL_REGIMES[r] for r in ctx.regimes}},
        "alternatives": {"names": list(ALTERNATIVES), "metis": metis, "kahip": kahip_identity(),
                         "scales": protocol.PARTITIONING["partition_scale_for_sensitivity"],
                         "seed": protocol.PARTITIONING["seed"], "fine_per_block": ctx.fine_per_block,
                         "empty_parts": part_mod.EMPTY_PART_RULE},
        "report": {"report_version": 1},
    }
    return configs


def stage_keys(ctx: Context, stages: Sequence[str]) -> dict[str, str]:
    configs = stage_configs(ctx)
    keys: dict[str, str] = {}
    for stage in STAGES:
        if stage not in stages:
            continue
        upstream = {dep: keys[dep] for dep in DEPENDS[stage] if dep in keys}
        keys[stage] = _key({"stage": stage, "config": configs[stage], "upstream": upstream})
    return keys


def persistent_outputs(ctx: Context, stage: str, marker: dict[str, Any]) -> list[Path]:
    d = ctx.store_dir
    if stage == "store":
        names = ["meta.json", "graph/indptr.npy", "graph/indices.npy", "graph/node_type.npy", "graph/attrs.npy",
                 "graph/orig_id.npy", "part/primary/partition.json", "part/primary/node_fine.npy"]
        if marker.get("info", {}).get("has_class"):
            names.append("graph/class.npy")
        return [d / name for name in names]
    if stage == "sigma":
        return [d / "sigma" / regime / name for regime in ctx.regimes for name in ("sigma.json", "codes.npy")]
    if stage == "alternatives":
        built = marker.get("info", {}).get("built", [])
        return [d / "part" / name / "partition.json" for name in built]
    if stage == "report":
        return [d / REPORT_NAME]
    return []


def work_outputs(ctx: Context, stage: str) -> list[Path]:
    files = [ctx.work / name for name in WORK_FILES.get(stage, ())]
    if stage == "load" and (ctx.work / "load.json").exists() and _read_json(ctx.work / "load.json").get("has_class"):
        files.append(ctx.work / "class.npy")
    return files


def plan(ctx: Context, keys: dict[str, str], force: bool) -> list[str]:
    """Stages to run: forced, stale (key or outputs), downstream of a rerun, or needed for inputs."""
    stages = list(keys)
    run: dict[str, bool] = {}
    for stage in stages:
        marker_path = ctx.marker(stage)
        marker = _read_json(marker_path) if marker_path.exists() else None
        stale = marker is None or marker.get("key") != keys[stage] or not all(
            path.exists() for path in persistent_outputs(ctx, stage, marker))
        run[stage] = force or stale or any(run.get(dep, False) for dep in DEPENDS[stage])
    changed = True
    while changed:
        changed = False
        for stage in stages:
            if run[stage]:
                for need in NEEDS.get(stage, ()):
                    if not run[need] and not all(path.exists() for path in work_outputs(ctx, need)):
                        run[need] = changed = True
            elif any(run.get(dep, False) for dep in DEPENDS[stage]):
                run[stage] = changed = True
    return [stage for stage in stages if run[stage]]


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------
def _load_work_csr(ctx: Context, mmap: bool) -> tuple[np.ndarray, np.ndarray]:
    mode = "r" if mmap else None
    return np.load(ctx.work / "indptr.npy", mmap_mode=mode), np.load(ctx.work / "indices.npy", mmap_mode=mode)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def stage_load(ctx: Context, rm: manifest.RunManifest) -> dict[str, Any]:
    if ctx.ooc:
        return stage_load_ooc(ctx)
    loaded = LOADERS[ctx.dataset](ctx.sources)
    ctx.log(f"loaded {loaded.num_nodes} nodes, {loaded.src.shape[0]} source edges")
    if "attr_note" in loaded.attribute_notes:
        ctx.log(f"note: {loaded.attribute_notes['attr_note']}")
    indptr, indices, report = normalize_edges(loaded.src, loaded.dst, loaded.num_nodes)
    loaded.src = loaded.dst = np.zeros(0, dtype=np.int64)
    ctx.work.mkdir(parents=True, exist_ok=True)
    np.save(ctx.work / "indptr.npy", indptr)
    np.save(ctx.work / "indices.npy", indices)
    np.save(ctx.work / "node_type.npy", loaded.node_type)
    np.save(ctx.work / "attrs.npy", loaded.attrs)
    if loaded.classes is not None:
        np.save(ctx.work / "class.npy", loaded.classes)
    info = {
        "num_nodes": loaded.num_nodes,
        "type_names": loaded.type_names,
        "type_counts": loaded.type_counts,
        "attr_dim": int(loaded.attrs.shape[1]),
        "attr_dtype": str(loaded.attrs.dtype),
        "has_class": loaded.classes is not None,
        "class_semantics": CLASS_SEMANTICS[ctx.dataset],
        "normalization": report,
        "notes": loaded.notes,
        "attribute_notes": loaded.attribute_notes,
    }
    _write_json(ctx.work / "load.json", info)
    return info


def stage_load_ooc(ctx: Context) -> dict[str, Any]:
    """papers100M: stream data.npz into memmaps and build the CSR by a chunked counting sort."""
    ctx.work.mkdir(parents=True, exist_ok=True)
    data_npz, label_npz = ctx.sources["data.npz"], ctx.sources["node-label.npz"]
    with np.load(data_npz, allow_pickle=False) as npz:
        num_nodes = int(np.asarray(npz["num_nodes_list"]).reshape(-1)[0])
        num_edges = int(np.asarray(npz["num_edges_list"]).reshape(-1)[0])
    ctx.log(f"streaming edge_index ({num_edges} edges) out of data.npz")
    edge_path = ctx.work / "edge_index.npy"
    edge_index = ooc.npz_member_to_npy(data_npz, "edge_index", edge_path)
    if tuple(edge_index.shape) != (2, num_edges):
        raise ValueError(f"{data_npz}: edge_index shape {edge_index.shape}, expected (2, {num_edges})")
    indptr, indices, report = ooc.csr_from_edge_index(edge_index, num_nodes, ctx.work)
    ooc.close_memmap(edge_index)
    ooc.close_memmap(indices)
    del edge_index, indices
    edge_path.unlink()
    ctx.log("streaming node_feat out of data.npz (cast to float32)")
    member = ooc.npz_member_info(data_npz, "node_feat")
    attrs = ooc.npz_member_to_npy(data_npz, "node_feat", ctx.work / "attrs.npy", dtype=np.float32)
    if attrs.shape[0] != num_nodes or attrs.ndim != 2:
        raise ValueError(f"{data_npz}: node_feat shape {attrs.shape} for {num_nodes} nodes")
    attr_dim = int(attrs.shape[1])
    ooc.close_memmap(attrs)
    del attrs
    with np.load(label_npz, allow_pickle=False) as npz:
        classes = labels_to_int32(np.asarray(npz["node_label"]), str(label_npz))
    if classes.shape[0] != num_nodes:
        raise ValueError(f"{label_npz}: {classes.shape[0]} labels for {num_nodes} nodes")
    np.save(ctx.work / "class.npy", classes)
    np.save(ctx.work / "node_type.npy", np.zeros(num_nodes, dtype=np.int8))
    schema = protocol.ATTRIBUTE_SCHEMAS[ctx.spec["schema"]]
    info = {
        "num_nodes": num_nodes,
        "type_names": list(schema["node_types"]),
        "type_counts": [num_nodes],
        "attr_dim": attr_dim,
        "attr_dtype": "float32",
        "has_class": True,
        "class_semantics": CLASS_SEMANTICS[ctx.dataset],
        "normalization": report,
        "notes": {"node_feat_source_dtype": member["dtype"], "labelled_nodes": int(np.count_nonzero(classes >= 0)),
                  "out_of_core": True},
        "attribute_notes": {},
    }
    _write_json(ctx.work / "load.json", info)
    return info


def stage_partition(ctx: Context, rm: manifest.RunManifest) -> dict[str, Any]:
    indptr, indices = _load_work_csr(ctx, mmap=True)
    n = int(indptr.shape[0]) - 1
    seed, ufactor = int(protocol.PARTITIONING["seed"]), int(protocol.PARTITIONING["ufactor"])
    if ctx.import_partition is not None:
        node_block, info = part_mod.import_partition(ctx.import_partition, n, nparts=ctx.partitions,
                                                     trusted_pickle=ctx.trusted_pickle)
        info.update(part_mod.partition_stats(indptr, indices, node_block))
    else:
        node_block, info = part_mod.metis_partition(indptr, indices, ctx.partitions, seed=seed, ufactor=ufactor,
                                                    tool=ctx.tool, timeout=ctx.timeout, binary=ctx.binary,
                                                    workdir=ctx.work / "tmp", fill_empty=True)
        for fill in info["empty_parts_filled"]:
            ctx.log(f"partition: METIS left part {fill['empty_part']} empty; it takes {fill['sizes_after'][1]} "
                    f"of the {fill['donor_size']} nodes of part {fill['donor_part']} ({fill['method']})")
    if info["num_parts"] != ctx.partitions:
        raise part_mod.PartitionError(f"primary partition has {info['num_parts']} non-empty blocks, "
                                      f"expected {ctx.partitions}")
    np.save(ctx.work / "node_block.npy", node_block)
    _write_json(ctx.work / "partition.json", info)
    ctx.log(f"partition: {info['num_parts']} blocks, edge cut {info['edge_cut']}, imbalance {info['imbalance']:.3f}")
    return info


def stage_fine(ctx: Context, rm: manifest.RunManifest) -> dict[str, Any]:
    indptr, indices = _load_work_csr(ctx, mmap=True)
    node_block = np.load(ctx.work / "node_block.npy")
    info: dict[str, Any] = {}
    node_fine, _ = part_mod.fine_partition(indptr, indices, node_block, ctx.fine_per_block,
                                           seed=int(protocol.PARTITIONING["seed"]),
                                           ufactor=int(protocol.PARTITIONING["ufactor"]), tool=ctx.tool,
                                           timeout=ctx.timeout, binary=ctx.binary, workdir=ctx.work / "tmp",
                                           info=info)
    np.save(ctx.work / "node_fine.npy", node_fine)
    _write_json(ctx.work / "fine.json", info)
    ctx.log(f"fine: {info['num_fine']} parts, {info['empty_parts_renumbered']} empty parts renumbered")
    return info


def stage_relabel(ctx: Context, rm: manifest.RunManifest) -> dict[str, Any]:
    node_block = np.load(ctx.work / "node_block.npy")
    node_fine = np.load(ctx.work / "node_fine.npy")
    perm = part_mod.block_contiguous_order(node_block, node_fine)
    if np.any(np.diff(node_block[perm]) < 0) or np.any(np.diff(node_fine[perm]) < 0):
        raise AssertionError("canonical order is not block- and fine-contiguous")
    np.save(ctx.work / "new_to_old.npy", perm)
    moved = int(np.count_nonzero(perm != np.arange(perm.shape[0])))
    return {"nodes_moved": moved, "node_order": "block, fine, source id"}


def _clear_store_outputs(ctx: Context) -> list[str]:
    for name in STORE_OUTPUTS:
        path = ctx.store_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    others = sorted(p.name for p in ctx.store_dir.iterdir() if p.name not in {RUN_DIR, WORK_DIR})
    if others:
        ctx.log(f"warning: {', '.join(others)} were built on the previous store; rebuild them")
    return others


def _source_entries(ctx: Context, rm: manifest.RunManifest) -> dict[str, Any]:
    inputs = (rm.manifest or {}).get("inputs", {})
    out = {}
    for name, path in sorted(ctx.sources.items()):
        entry = inputs.get(name) or manifest.sha256_file(path)
        out[name] = {"path": _relative(path, ctx.data_root), "bytes": entry.get("bytes"),
                     "sha256": entry.get("sha256") or entry.get("fingerprint")}
    return out


def stage_store(ctx: Context, rm: manifest.RunManifest) -> dict[str, Any]:
    load = _read_json(ctx.work / "load.json")
    pinfo = _read_json(ctx.work / "partition.json")
    finfo = _read_json(ctx.work / "fine.json")
    perm = np.load(ctx.work / "new_to_old.npy")
    node_block = np.load(ctx.work / "node_block.npy")
    node_fine = np.load(ctx.work / "node_fine.npy")
    has_class = bool(load["has_class"])
    stale = _clear_store_outputs(ctx)
    partition_info = dict(pinfo)
    partition_info.update({"contiguous": True, "fine": finfo, "computed_on": "source ids",
                           "partitions_override": ctx.overrides.get("partitions") is not None,
                           "fine_per_block_override": ctx.overrides.get("fine_per_block") is not None})
    meta = {
        "schema": ctx.spec["schema"],
        "loader": ctx.spec["loader"],
        "num_types": len(load["type_names"]),
        "type_names": load["type_names"],
        "type_counts": load["type_counts"],
        "type_offsets": [int(x) for x in np.concatenate([[0], np.cumsum(load["type_counts"])[:-1]])],
        "attr_dim": load["attr_dim"],
        "attr_dtype": load["attr_dtype"],
        "has_class": has_class,
        "class_semantics": load["class_semantics"],
        "source_files": _source_entries(ctx, rm),
        "normalization": {"rules": protocol.GRAPH_NORMALIZATION, **load["normalization"]},
        "attribute_notes": load["attribute_notes"],
        "loader_notes": load["notes"],
        "primary_partition": {"tool": pinfo.get("tool"), "version": pinfo.get("version"),
                              "num_blocks": ctx.partitions, "fine_per_block": ctx.fine_per_block,
                              "seed": protocol.PARTITIONING["seed"], "ufactor": protocol.PARTITIONING["ufactor"]},
        "node_order": "canonical id = rank of (primary block, fine part, source id)",
        "orig_id": "graph/orig_id.npy holds the source id (type offset + local id in the source files)",
        "prepare_version": PREPARE_VERSION,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_hash": protocol.protocol_hash(),
    }
    n = int(load["num_nodes"])
    if ctx.ooc:
        indptr = np.load(ctx.work / "indptr.npy")
        indices = np.load(ctx.work / "indices.npy", mmap_mode="r")
        attrs = np.load(ctx.work / "attrs.npy", mmap_mode="r")
        ooc.write_store_ooc(ctx.store_root, ctx.dataset, indptr=indptr, indices=indices,
                            node_type=np.load(ctx.work / "node_type.npy"), attrs=attrs, node_block=node_block,
                            node_fine=node_fine,
                            classes=np.load(ctx.work / "class.npy") if has_class else None,
                            meta=meta, partition_info=partition_info, new_to_old=perm)
        del indptr, indices, attrs
    else:
        indptr, indices = _load_work_csr(ctx, mmap=False)
        old_to_new = np.empty(n, dtype=np.int64)
        old_to_new[perm] = np.arange(n, dtype=np.int64)
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        upper = rows < indices
        edges = np.stack([old_to_new[rows[upper]], old_to_new[indices[upper].astype(np.int64)]], axis=1)
        del rows, upper, indices, indptr, old_to_new
        attrs = np.load(ctx.work / "attrs.npy", mmap_mode="r")[perm]
        store_mod.write_store(
            ctx.store_root, ctx.dataset, edges=edges, num_nodes=n,
            node_type=np.load(ctx.work / "node_type.npy")[perm], attrs=attrs,
            node_block=node_block[perm], node_fine=node_fine[perm],
            classes=np.load(ctx.work / "class.npy")[perm] if has_class else None,
            orig_id=perm, meta=meta, partition_info=partition_info)
        del edges, attrs
    graph = ctx.store_dir / "graph"
    block_nodes = np.load(ctx.store_dir / "part" / "primary" / "block_nodes.npy")
    if not np.array_equal(block_nodes, np.arange(n)):
        raise AssertionError("the stored primary partition is not block-contiguous")
    del block_nodes
    indptr = np.load(graph / "indptr.npy", mmap_mode="r" if ctx.ooc else None)
    indices = np.load(graph / "indices.npy", mmap_mode="r" if ctx.ooc else None)
    node_type = np.load(graph / "node_type.npy")
    pairs = features.relation_pairs_present(indptr, indices, node_type, len(load["type_names"]))
    m = int(indices.shape[0]) // 2
    ooc.close_memmap(indptr)
    ooc.close_memmap(indices)
    del indptr, indices
    meta_path = ctx.store_dir / "meta.json"
    stored = _read_json(meta_path)
    stored["relation_pairs"] = [[a, b] for a, b in pairs]
    meta_path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if m != int(load["normalization"]["undirected_edges"]):
        raise AssertionError(f"store has {m} edges, normalisation produced {load['normalization']['undirected_edges']}")
    shutil.rmtree(ctx.work, ignore_errors=True)
    ctx.log(f"store written: {n} nodes, {m} edges, relation pairs {stored['relation_pairs']}")
    return {"num_nodes": n, "undirected_edges": m, "has_class": has_class, "relation_pairs": stored["relation_pairs"],
            "stale_directories_left": stale}


def stage_sigma(ctx: Context, rm: manifest.RunManifest) -> dict[str, Any]:
    shutil.rmtree(ctx.store_dir / "sigma", ignore_errors=True)  # the store holds exactly the requested regimes
    report = sigma.build_store_regimes(ctx.store_dir, ctx.regimes)
    for regime, info in report.items():
        ctx.log(f"sigma {regime}: {info['num_codes']} codes, {info['singleton_codes']} singletons")
    return report


def stage_alternatives(ctx: Context, rm: manifest.RunManifest) -> dict[str, Any]:
    mode = "r" if ctx.ooc else None
    graph = ctx.store_dir / "graph"
    indptr = np.load(graph / "indptr.npy", mmap_mode=mode)
    indices = np.load(graph / "indices.npy", mmap_mode=mode)
    seed, ufactor = int(protocol.PARTITIONING["seed"]), int(protocol.PARTITIONING["ufactor"])
    tmp = ctx.store_dir / WORK_DIR / "tmp"
    meta_path = ctx.store_dir / "meta.json"
    meta = _read_json(meta_path)
    for name in ALTERNATIVES:  # drop earlier builds, including ones that cannot be rebuilt now
        shutil.rmtree(ctx.store_dir / "part" / name, ignore_errors=True)
    meta["partitions"] = [name for name in meta["partitions"] if name not in ALTERNATIVES]
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    built, results = [], {}
    for name in ALTERNATIVES:
        t0 = time.perf_counter()
        try:
            if name == "kahip_fast":
                node_block, info = part_mod.kahip_partition(indptr, indices, ctx.partitions, seed=seed,
                                                            preconfiguration="fast", timeout=ctx.timeout,
                                                            workdir=tmp)
            elif name == "bfs_random":
                node_block, info = part_mod.bfs_random_partition(indptr, indices, ctx.partitions, seed=seed)
            else:
                scale = float(name.split("_x", 1)[1])
                nparts = part_mod.scaled_nparts(ctx.partitions, scale)
                node_block, info = part_mod.metis_partition(indptr, indices, nparts, seed=seed, ufactor=ufactor,
                                                            tool=ctx.tool, timeout=ctx.timeout, binary=ctx.binary,
                                                            workdir=tmp, fill_empty=True)
                info["scale"] = scale
        except part_mod.PartitionerUnavailable as exc:
            results[name] = {"skipped": str(exc)}
            ctx.log(f"alternative {name} skipped: {exc}")
            continue
        fine_info: dict[str, Any] = {}
        node_fine, _ = part_mod.fine_partition(indptr, indices, node_block, ctx.fine_per_block, seed=seed,
                                               ufactor=ufactor, tool=ctx.tool, timeout=ctx.timeout,
                                               binary=ctx.binary, workdir=tmp, info=fine_info)
        info.update({"name": name, "contiguous": False, "fine": fine_info, "computed_on": "canonical store ids"})
        add = ooc.add_partition_ooc if ctx.ooc else store_mod.add_partition
        add(ctx.store_dir, name, node_block=node_block, node_fine=node_fine, info=info)
        built.append(name)
        results[name] = {"num_parts": info["num_parts"], "edge_cut": info["edge_cut"],
                         "imbalance": info["imbalance"], "num_fine": fine_info["num_fine"],
                         "wall_seconds": time.perf_counter() - t0}
        ctx.log(f"alternative {name}: {info['num_parts']} blocks, edge cut {info['edge_cut']}")
    ooc.close_memmap(indptr)
    ooc.close_memmap(indices)
    shutil.rmtree(ctx.store_dir / WORK_DIR, ignore_errors=True)
    return {"built": built, "results": results}


def _quantiles(values: np.ndarray, qs: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1.0)) -> dict[str, float]:
    if values.shape[0] == 0:
        return {}
    return {f"p{q * 100:g}": float(v) for q, v in zip(qs, np.quantile(values, qs))}


def _size_stats(ptr: np.ndarray) -> dict[str, Any]:
    sizes = np.diff(np.asarray(ptr, dtype=np.int64))
    return {"count": int(sizes.shape[0]), "min": int(sizes.min()), "max": int(sizes.max()),
            "mean": float(sizes.mean()), "quantiles": _quantiles(sizes.astype(np.float64))}


def build_report(ctx: Context) -> dict[str, Any]:
    d = ctx.store_dir
    meta = _read_json(d / "meta.json")
    indptr = np.load(d / "graph" / "indptr.npy")
    degree = np.diff(indptr).astype(np.float64)
    del indptr
    node_type = np.load(d / "graph" / "node_type.npy")
    n = int(node_type.shape[0])
    m = int(meta["num_undirected_edges"])
    classes_path = d / "graph" / "class.npy"
    report: dict[str, Any] = {
        "dataset": ctx.dataset,
        "prepare_version": PREPARE_VERSION,
        "protocol_hash": meta.get("protocol_hash"),
        "store_dir": str(d),
        "nodes": n,
        "undirected_edges": m,
        "adjacency_entries": 2 * m,
        "type_names": meta.get("type_names"),
        "type_counts": [int(c) for c in np.bincount(node_type, minlength=int(meta["num_types"]))],
        "attr_dim": meta.get("attr_dim"),
        "attr_dtype": meta.get("attr_dtype"),
        "labelled_nodes": int(np.count_nonzero(np.load(classes_path) >= 0)) if classes_path.exists() else 0,
        "degree": {"mean": float(degree.mean()) if n else 0.0, "isolated": int(np.count_nonzero(degree == 0)),
                   "quantiles": _quantiles(degree)},
        "normalization": meta.get("normalization"),
        "relation_pairs": meta.get("relation_pairs"),
        "partitions": {},
        "sigma": {},
        "stages": {},
    }
    del degree, node_type
    for name in meta.get("partitions", []):
        part_dir = d / "part" / name
        info = _read_json(part_dir / "partition.json")
        entry = {
            "tool": info.get("tool"),
            "version": info.get("version"),
            "num_blocks": info.get("num_blocks"),
            "num_fine": info.get("num_fine"),
            "cut_edges": info.get("cut_edges"),
            "cut_fraction": (info.get("cut_edges") or 0) / m if m else 0.0,
            "imbalance": info.get("imbalance"),
            "metis_empty_parts": info.get("metis_empty_parts"),
            "empty_parts_filled": len(info.get("empty_parts_filled") or []),
            "empty_parts_renumbered": info.get("empty_parts_renumbered"),
            "fine_empty_parts_renumbered": (info.get("fine") or {}).get("empty_parts_renumbered"),
            "blocks": _size_stats(np.load(part_dir / "block_ptr.npy")),
        }
        if (part_dir / "fine_ptr.npy").exists():
            entry["fine"] = _size_stats(np.load(part_dir / "fine_ptr.npy"))
        report["partitions"][name] = entry
    sigma_dir = d / "sigma"
    if sigma_dir.is_dir():
        for regime_dir in sorted(p for p in sigma_dir.iterdir() if (p / "sigma.json").exists()):
            info = _read_json(regime_dir / "sigma.json")
            report["sigma"][regime_dir.name] = {k: info.get(k) for k in
                                                ("num_codes", "singleton_codes", "max_df", "median_df")}
    for stage in STAGES:
        path = ctx.marker(stage)
        if path.exists():
            marker = _read_json(path)
            report["stages"][stage] = {k: marker.get(k) for k in ("wall_seconds", "peak_rss_bytes", "finished_utc")}
    return report


def stage_report(ctx: Context, rm: manifest.RunManifest) -> dict[str, Any]:
    report = build_report(ctx)
    _write_json(ctx.store_dir / REPORT_NAME, report)
    return {"path": str(ctx.store_dir / REPORT_NAME)}


STAGE_FUNCS: dict[str, Callable[[Context, manifest.RunManifest], dict[str, Any]]] = {
    "load": stage_load,
    "partition": stage_partition,
    "fine": stage_fine,
    "relabel": stage_relabel,
    "store": stage_store,
    "sigma": stage_sigma,
    "alternatives": stage_alternatives,
    "report": stage_report,
}


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def os_peak_rss() -> dict[str, int]:
    """OS peak memory of this process, and on POSIX of its finished children (the METIS workers)."""
    out = part_mod.self_peak_memory()
    try:
        import resource

        scale = 1 if sys.platform == "darwin" else 1024
        out["children_maxrss_bytes"] = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * scale
    except ImportError:
        pass
    return out


class _Sampler:
    """memprobe.TreeSampler over this process tree, or a no-op without psutil."""

    def __init__(self, interval: float = 0.1) -> None:
        try:
            from .memprobe import TreeSampler

            self.sampler: Any = TreeSampler(os.getpid(), interval=interval, max_timeline=200).start()
        except RuntimeError:
            self.sampler = None

    def stop(self) -> int | None:
        return None if self.sampler is None else int(self.sampler.stop().peak_rss_bytes)


def store_file_hashes(store_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in STORE_OUTPUTS:
        path = store_dir / name
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else ([path] if path.exists() else [])
        for file in files:
            entry = manifest.sha256_file(file, fast=file.stat().st_size > manifest.FULL_HASH_LIMIT_BYTES)
            entry.pop("path", None)
            out[file.relative_to(store_dir).as_posix()] = entry
    return out


def make_context(dataset: str, *, data_root: str | os.PathLike, store_root: str | os.PathLike,
                 regimes: Sequence[str] | None = None, alternatives: bool = False,
                 import_partition: str | os.PathLike | None = None, skip_sigma: bool = False,
                 partitions: int | None = None, fine_per_block: int | None = None, tool: str = "auto",
                 binary: str | Sequence[str] | None = None, timeout: float | None = None,
                 trusted_pickle: bool = False) -> Context:
    if dataset not in protocol.DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(protocol.DATASETS)}")
    spec = protocol.DATASETS[dataset]
    chosen = list(regimes) if regimes else protocol.regimes_for(dataset)
    unknown = [r for r in chosen if r not in protocol.LABEL_REGIMES]
    if unknown:
        raise ValueError(f"unknown regimes {unknown}")
    data_root = Path(data_root)
    return Context(
        dataset=dataset,
        data_root=data_root,
        store_root=Path(store_root),
        sources=source_files(dataset, data_root),
        regimes=chosen,
        partitions=int(partitions if partitions is not None else spec["partitions"]),
        fine_per_block=int(fine_per_block if fine_per_block is not None else spec["fine_per_block"]),
        tool=tool,
        binary=binary,
        timeout=timeout,
        import_partition=None if import_partition is None else Path(import_partition),
        trusted_pickle=trusted_pickle,
        alternatives=alternatives,
        skip_sigma=skip_sigma,
        overrides={"partitions": partitions, "fine_per_block": fine_per_block},
    )


def prepare(dataset: str, *, data_root: str | os.PathLike, store_root: str | os.PathLike,
            regimes: Sequence[str] | None = None, alternatives: bool = False,
            import_partition: str | os.PathLike | None = None, skip_sigma: bool = False, force: bool = False,
            partitions: int | None = None, fine_per_block: int | None = None, tool: str = "auto",
            binary: str | Sequence[str] | None = None, timeout: float | None = None,
            trusted_pickle: bool = False, argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Run the pending stages for one dataset and return the preparation report.

    partitions and fine_per_block override the protocol values (tests and smoke
    runs only; the override is recorded in partition.json).
    """
    ctx = make_context(dataset, data_root=data_root, store_root=store_root, regimes=regimes,
                       alternatives=alternatives, import_partition=import_partition, skip_sigma=skip_sigma,
                       partitions=partitions, fine_per_block=fine_per_block, tool=tool, binary=binary,
                       timeout=timeout, trusted_pickle=trusted_pickle)
    stages = ctx.stages()
    keys = stage_keys(ctx, stages)
    todo = plan(ctx, keys, force)
    report_path = ctx.store_dir / REPORT_NAME
    if not todo:
        ctx.log(f"every stage is done ({', '.join(stages)}); nothing to do")
        return _read_json(report_path)
    ctx.log(f"stages to run: {', '.join(todo)}")
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    old = ctx.run_dir / manifest.MANIFEST_NAME
    if old.exists():
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        (ctx.run_dir / "attempts").mkdir(exist_ok=True)
        os.replace(old, ctx.run_dir / "attempts" / f"manifest-{stamp}.json")
    for stage in todo:
        ctx.marker(stage).unlink(missing_ok=True)
    inputs: dict[str, Any] = dict(ctx.sources)
    if ctx.import_partition is not None:
        inputs["import_partition"] = ctx.import_partition
    config = {
        "dataset": dataset,
        "data_root": str(ctx.data_root),
        "store_root": str(ctx.store_root),
        "regimes": ctx.regimes,
        "alternatives": alternatives,
        "import_partition": None if ctx.import_partition is None else str(ctx.import_partition),
        "skip_sigma": skip_sigma,
        "force": force,
        "partitions": ctx.partitions,
        "fine_per_block": ctx.fine_per_block,
        "overrides": ctx.overrides,
        "metis_tool": tool,
        "metis_timeout": timeout,
        "prepare_version": PREPARE_VERSION,
        "stages": stages,
        "stages_run": todo,
        "stage_keys": keys,
    }
    solvers = {}
    name, cmd = part_mod.resolve_metis_tool(tool, binary)
    if name == "gpmetis" and cmd:
        solvers["gpmetis"] = cmd[-1]
    with manifest.RunManifest(ctx.run_dir, experiment="prepare_data", config=config, inputs=inputs,
                              solvers=solvers, argv=argv) as rm:
        log_entries: dict[str, Any] = {}
        for stage in stages:
            if stage not in todo:
                ctx.log(f"{stage}: done earlier, skipped")
                log_entries[stage] = {"skipped": True}
                continue
            ctx.log(f"{stage}: running")
            sampler = _Sampler()
            t0 = time.perf_counter()
            info = STAGE_FUNCS[stage](ctx, rm)
            wall = time.perf_counter() - t0
            peak = sampler.stop()
            marker = {"stage": stage, "key": keys[stage], "finished_utc": manifest.utc_now(), "wall_seconds": wall,
                      "peak_rss_bytes": peak, "info": info}
            _write_json(ctx.marker(stage), marker)
            log_entries[stage] = {"wall_seconds": wall, "peak_rss_bytes": peak}
            peak_text = "n/a" if peak is None else f"{peak / 2**30:.2f} GiB"
            ctx.log(f"{stage}: {wall:.1f} s, peak RSS {peak_text}")
        peaks = [entry["peak_rss_bytes"] for entry in log_entries.values() if entry.get("peak_rss_bytes")]
        rm.note(stages=log_entries, peak_rss_bytes=max(peaks) if peaks else None, os_peak=os_peak_rss(),
                store_files=store_file_hashes(ctx.store_dir))
    return _read_json(report_path) if report_path.exists() else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(protocol.DATASETS))
    parser.add_argument("--data-root", default=os.environ.get("JIGSAW_DATA_ROOT") or str(manifest.REPO_ROOT / "data"))
    parser.add_argument("--store-root", default=os.environ.get("JIGSAW_STORE_ROOT"))
    parser.add_argument("--regimes", nargs="+", default=None, choices=list(protocol.LABEL_REGIMES),
                        help="default: protocol.regimes_for(dataset)")
    parser.add_argument("--alternatives", action="store_true", help="also build the sensitivity partitions")
    parser.add_argument("--import-partition", default=None,
                        help="node-to-block file (.npy, .txt/.part, .pt) in source ids instead of running METIS")
    parser.add_argument("--trusted-pickle", action="store_true",
                        help="allow a full unpickle of your own legacy .pt partition file")
    parser.add_argument("--skip-sigma", action="store_true")
    parser.add_argument("--force", action="store_true", help="redo every stage")
    parser.add_argument("--metis-tool", default="auto", choices=["auto", "gpmetis", "pymetis"])
    parser.add_argument("--metis-timeout", type=float, default=None, help="seconds per METIS call")
    args = parser.parse_args(argv)
    if not args.store_root:
        parser.error("--store-root (or JIGSAW_STORE_ROOT) is required")
    report = prepare(args.dataset, data_root=args.data_root, store_root=args.store_root, regimes=args.regimes,
                     alternatives=args.alternatives, import_partition=args.import_partition,
                     skip_sigma=args.skip_sigma, force=args.force, tool=args.metis_tool,
                     timeout=args.metis_timeout, trusted_pickle=args.trusted_pickle,
                     argv=sys.argv if argv is None else ["prepare_data", *argv])
    summary = {k: report.get(k) for k in ("dataset", "nodes", "undirected_edges", "store_dir")}
    summary["partitions"] = {name: {k: entry.get(k) for k in ("num_blocks", "num_fine", "cut_edges")}
                             for name, entry in report.get("partitions", {}).items()}
    summary["sigma"] = {regime: entry.get("num_codes") for regime, entry in report.get("sigma", {}).items()}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
