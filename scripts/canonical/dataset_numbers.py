r"""Dataset statistics as paper numbers, read from the prepared stores.

    python -m scripts.canonical.dataset_numbers --store-root ROOT [--datasets D ...] --out DIR

A store is ROOT/<dataset>, written by prepare_data.py. Its prepare run (_prepare_run/manifest.json) must
be complete and match its outputs (manifest.verify_manifest), and prepare_report.json must be the file
that run recorded (extra.store_files). The run ran before the protocol took its final hash (the router
threshold is fit after training), so its hash is not checked and the numbers carry the current one, as
the reducer's E17 does. Without --datasets every protocol dataset whose store has a prepare run is read
and the others are listed on stderr; a dataset named with --datasets must have one.

Keys, from prepare_report.json with the prepare run as their source (d the dataset):
  data.<d>.nodes, .edges (undirected, after normalization), .types, .attr_dim, .labelled_nodes
  data.<d>.type.<name>.nodes                  nodes of each node type
  data.<d>.degree.{mean, median, p99, max, isolated}
  data.<d>.normalization.{input_edges, reciprocal_pairs_merged, duplicates_merged, self_loops_dropped}
  data.<d>.sigma.<r>.{codes, singletons, singleton_share, median_df, max_df}
                                               regime r: distinct labels, labels of one node, the share
                                               of nodes whose label no other node holds (pct), median
                                               and largest nodes per label
  data.<d>.partition.<p>.{blocks, fine, block_nodes_min, block_nodes_median, block_nodes_max,
                          cut_edges, cut_share, imbalance, metis_empty_parts, empty_parts_filled, tool}
                                               p primary or a sensitivity partition (metis_x0.5 is
                                               metis_x0p5, as in the reducer)
A median or quantile that is a whole number prints as a count. Table data.datasets: dataset, nodes,
edges, node types, attribute dimension, blocks and the label count of each regime (-- when the dataset
has no such regime).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import manifest, numbers_json, protocol

PREPARE_RUN = "_prepare_run"  # prepare_data.RUN_DIR
REPORT = "prepare_report.json"  # prepare_data.REPORT_NAME
TABLE = "data.datasets"
NAMES = {"corafull": "CoraFull", "arxiv": "ogbn-arxiv", "mag": "ogbn-mag", "products": "ogbn-products",
         "papers100m": "ogbn-papers100M"}
TABLE_REGIMES = ("R0", "R1", "R2-512", "R2-128", "R2-32")


class DatasetNumbersError(ValueError):
    pass


def partition_segment(name: str) -> str:
    return numbers_json.check_key(name.replace(".", "p"), "partition")


def read_store(store: str | os.PathLike) -> tuple[dict[str, Any], dict[str, Any], str]:
    """(prepare report, prepare manifest, the manifest's sha256) of a store, checked."""
    store = Path(store)
    run = store / PREPARE_RUN
    mpath = run / manifest.MANIFEST_NAME
    if not mpath.is_file():
        raise DatasetNumbersError(f"{store}: no prepare run ({mpath} is missing)")
    raw = mpath.read_bytes()
    doc = json.loads(raw)
    if doc.get("status") != "complete":
        raise DatasetNumbersError(f"{run}: prepare run status {doc.get('status')!r}, not 'complete'")
    problems = manifest.verify_manifest(run)
    if problems:
        raise DatasetNumbersError(f"{run}: differs from its manifest: {'; '.join(problems[:5])}")
    report_path = store / REPORT
    if not report_path.is_file():
        raise DatasetNumbersError(f"{store}: no {REPORT}")
    recorded = ((doc.get("extra") or {}).get("store_files") or {}).get(REPORT) or {}
    if recorded.get("sha256") != hashlib.sha256(report_path.read_bytes()).hexdigest():
        raise DatasetNumbersError(f"{report_path}: not the report its prepare run recorded")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("dataset") != store.name:
        raise DatasetNumbersError(f"{report_path}: dataset {report.get('dataset')!r} in the store {store.name!r}")
    return report, doc, hashlib.sha256(raw).hexdigest()


def add_dataset(nums: numbers_json.Numbers, store: str | os.PathLike) -> str:
    """Adds the keys of one store; returns the dataset."""
    store = Path(store)
    report, doc, sha = read_store(store)
    d = numbers_json.check_key(store.name, "dataset")
    src = nums.source(f"data.{d}.prepare", path=str(store / PREPARE_RUN), manifest_sha256=sha,
                      experiment=str(doc.get("experiment")), dataset=d)

    def put(key: str, value: Any, unit: str, digits: int | None = None) -> None:
        if value is None:
            return
        if unit == "number" and float(value).is_integer():
            unit, value, digits = "count", int(value), None
        nums.number(f"data.{d}.{key}", value, unit, digits=digits, source=[src])

    put("nodes", report["nodes"], "count")
    put("edges", report["undirected_edges"], "count")
    put("types", len(report["type_names"]), "count")
    for name, count in zip(report["type_names"], report["type_counts"]):
        put(f"type.{numbers_json.check_key(name, 'node type')}.nodes", count, "count")
    put("attr_dim", report.get("attr_dim"), "count")
    put("labelled_nodes", report.get("labelled_nodes"), "count")
    degree = report.get("degree") or {}
    quantiles = degree.get("quantiles") or {}
    put("degree.mean", degree.get("mean"), "number", 1)
    put("degree.median", quantiles.get("p50"), "number", 1)
    put("degree.p99", quantiles.get("p99"), "number", 1)
    put("degree.max", quantiles.get("p100"), "number", 1)
    put("degree.isolated", degree.get("isolated"), "count")
    norm = report.get("normalization") or {}
    for name in ("input_edges", "reciprocal_pairs_merged", "duplicates_merged", "self_loops_dropped"):
        put(f"normalization.{name}", norm.get(name), "count")
    for regime, s in sorted((report.get("sigma") or {}).items()):
        r = numbers_json.check_key(regime, "regime")
        put(f"sigma.{r}.codes", s.get("num_codes"), "count")
        put(f"sigma.{r}.singletons", s.get("singleton_codes"), "count")
        if s.get("singleton_codes") is not None and report["nodes"]:
            put(f"sigma.{r}.singleton_share", 100.0 * s["singleton_codes"] / report["nodes"], "pct")
        put(f"sigma.{r}.median_df", s.get("median_df"), "number", 1)
        put(f"sigma.{r}.max_df", s.get("max_df"), "count")
    for name, p in sorted((report.get("partitions") or {}).items()):
        seg = f"partition.{partition_segment(name)}"
        blocks = p.get("blocks") or {}
        put(f"{seg}.blocks", p.get("num_blocks"), "count")
        put(f"{seg}.fine", p.get("num_fine"), "count")
        put(f"{seg}.block_nodes_min", blocks.get("min"), "count")
        put(f"{seg}.block_nodes_median", (blocks.get("quantiles") or {}).get("p50"), "number", 1)
        put(f"{seg}.block_nodes_max", blocks.get("max"), "count")
        put(f"{seg}.cut_edges", p.get("cut_edges"), "count")
        if p.get("cut_fraction") is not None:
            put(f"{seg}.cut_share", 100.0 * p["cut_fraction"], "pct")
        put(f"{seg}.imbalance", p.get("imbalance"), "number", 3)
        put(f"{seg}.metis_empty_parts", p.get("metis_empty_parts"), "count")
        put(f"{seg}.empty_parts_filled", p.get("empty_parts_filled"), "count")
        if p.get("tool"):
            put(f"{seg}.tool", str(p["tool"]), "text")
    return d


def _order(d: str) -> tuple[int, str]:
    names = list(protocol.DATASETS)
    return (names.index(d) if d in names else len(names), d)


def dataset_numbers(stores: Sequence[str | os.PathLike]) -> numbers_json.Numbers:
    nums = numbers_json.Numbers(split="test")
    datasets = []
    for store in stores:
        d = Path(store).name
        if d in datasets:
            raise DatasetNumbersError(f"{d}: two stores of one dataset")
        datasets.append(add_dataset(nums, store))
    regimes = [r for r in TABLE_REGIMES if any(f"data.{d}.sigma.{r}.codes" in nums.numbers for d in datasets)]
    rows = []
    for d in sorted(datasets, key=_order):
        cells: list[Any] = [NAMES.get(d, d)]
        for key in ("nodes", "edges", "types", "attr_dim", "partition.primary.blocks",
                    *(f"sigma.{r}.codes" for r in regimes)):
            full = f"data.{d}.{key}"
            cells.append({"key": full} if full in nums.numbers else "--")
        rows.append(cells)
    nums.table(TABLE, ["Dataset", "Nodes", "Edges", "Node types", "Attribute dim.", "Blocks",
                       *(f"{r} labels" for r in regimes)], rows)
    return nums


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", default=os.environ.get("JIGSAW_STORE_ROOT"))
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="default: every protocol dataset whose store has a prepare run")
    parser.add_argument("--out", required=True, help="directory for numbers.json")
    args = parser.parse_args(argv)
    if not args.store_root:
        parser.error("give --store-root or set JIGSAW_STORE_ROOT")
    root = Path(args.store_root)
    if args.datasets is None:
        stores = []
        for d in protocol.DATASETS:
            if (root / d / PREPARE_RUN / manifest.MANIFEST_NAME).is_file():
                stores.append(root / d)
            else:
                print(f"absent: {root / d}", file=sys.stderr)
    else:
        stores = [root / d for d in args.datasets]
    try:
        if not stores:
            raise DatasetNumbersError(f"{root}: no store with a prepare run")
        nums = dataset_numbers(stores)
    except (DatasetNumbersError, numbers_json.NumbersError, OSError, ValueError, KeyError) as exc:
        print(f"dataset_numbers: {exc}", file=sys.stderr)
        return 2
    path = nums.write(args.out)
    print(f"{len(nums.numbers)} dataset numbers from {len(stores)} stores -> {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
