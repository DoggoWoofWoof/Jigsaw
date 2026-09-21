"""Validate the exact split MAG production query caches from query payloads.

The check is deliberately independent of planted-ID pruning. Planted target IDs
are used only as audit witnesses: each cached query must carry the same raw
features, feature-derived labels, and serving-safe ``type_feat32`` tokens as its
known target nodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_overlap_glasgow_cascade as cascade
from benchmark_retrieval import _feature_bit_hash, _node_type_tensor, load_named_data
from src.utils import feature_to_label


EXPECTED_KEYS = {
    "bca702f664e58aad",
    "86d76b2687707326",
    "44e026ffd70eb787",
    "ec622a8592da29ec",
}
EXPECTED_FAMILIES = {
    "single",
    "k_hop",
    "degree_k_hop",
    "multi_fine",
    "random_walk",
    "multi_coarse",
}
EXPECTED_SIZES = {20, 50, 100}


def type_feat32(x: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
    return node_type.detach().cpu().long() * (2**32) + _feature_bit_hash(
        x.detach().cpu().float(), width=32
    )


def query_fingerprint(item: dict) -> str:
    query = item["query"]
    digest = hashlib.sha256()
    digest.update(str(item["query_id"]).encode("utf-8"))
    digest.update(query.x.detach().cpu().contiguous().numpy().tobytes())
    digest.update(query.edge_index.detach().cpu().contiguous().numpy().tobytes())
    edge_type = getattr(query, "edge_type", None)
    if isinstance(edge_type, torch.Tensor):
        digest.update(edge_type.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache_files = sorted(args.cache_dir.glob("*_queries.pt"))
    cache_keys = {path.name.removesuffix("_queries.pt") for path in cache_files}
    if cache_keys != EXPECTED_KEYS:
        raise ValueError(
            f"split cache keys mismatch: expected={sorted(EXPECTED_KEYS)} "
            f"actual={sorted(cache_keys)}"
        )

    data = load_named_data("mag", args.data_root)
    target_node_type = _node_type_tensor(data)

    family_size_counts: Counter[tuple[str, int]] = Counter()
    family_counts: Counter[str] = Counter()
    file_rows: dict[str, int] = {}
    file_families: dict[str, list[str]] = {}
    raw_pairs = 0
    label_pairs = 0
    signature_pairs = 0
    fingerprints_by_shape: defaultdict[tuple[str, ...], list[set[str]]] = defaultdict(list)

    for path in cache_files:
        queries = cascade.torch_load_any(path)
        file_rows[path.name] = len(queries)
        families = sorted({str(item["query_type"]) for item in queries})
        file_families[path.name] = families
        expected_rows = len(families) * len(EXPECTED_SIZES) * 50
        if len(queries) != expected_rows:
            raise ValueError(
                f"{path.name}: expected {expected_rows} rows for {families}, got {len(queries)}"
            )
        file_fingerprints: set[str] = set()

        for item in queries:
            if bool(item.get("is_negative")) or not bool(item.get("expected_match", True)):
                raise ValueError(f"{path.name}: positive cache contains a negative query")
            family = str(item["query_type"])
            size = int(item["target_query_size"])
            if family not in EXPECTED_FAMILIES or size not in EXPECTED_SIZES:
                raise ValueError(f"{path.name}: unexpected query cell {(family, size)}")

            query = item["query"]
            nodes = item["query_nodes"].detach().cpu().long()
            if int(query.num_nodes) != int(nodes.numel()):
                raise ValueError(f"{path.name}: query/node witness length mismatch")
            global_id = getattr(query, "global_id", None)
            if not isinstance(global_id, torch.Tensor) or not torch.equal(
                global_id.detach().cpu().long(), nodes
            ):
                raise ValueError(f"{path.name}: query.global_id does not match audit witness")

            target_x = data.x[nodes].detach().cpu()
            query_x = query.x.detach().cpu()
            if not torch.equal(query_x, target_x):
                raise ValueError(f"{path.name}: raw feature mismatch in {item['query_id']}")
            raw_pairs += int(nodes.numel())

            query_labels = cascade.derive_query_labels(query, label_source="feature")
            target_labels = [feature_to_label(target_x[i]) for i in range(target_x.size(0))]
            if query_labels != target_labels:
                raise ValueError(f"{path.name}: feature-label mismatch in {item['query_id']}")
            label_pairs += int(nodes.numel())

            query_type = _node_type_tensor(query)
            target_type = target_node_type[nodes]
            if not torch.equal(query_type, target_type):
                raise ValueError(f"{path.name}: node-type mismatch in {item['query_id']}")
            query_tokens = type_feat32(query_x, query_type)
            target_tokens = type_feat32(target_x, target_type)
            if not torch.equal(query_tokens, target_tokens):
                raise ValueError(f"{path.name}: type_feat32 mismatch in {item['query_id']}")
            signature_pairs += int(nodes.numel())

            family_counts[family] += 1
            family_size_counts[(family, size)] += 1
            fingerprint = query_fingerprint(item)
            if fingerprint in file_fingerprints:
                raise ValueError(f"{path.name}: duplicate query payload {item['query_id']}")
            file_fingerprints.add(fingerprint)

        fingerprints_by_shape[tuple(families)].append(file_fingerprints)

    if sum(file_rows.values()) != 1800:
        raise ValueError(f"expected 1,800 MAG positives, got {sum(file_rows.values())}")
    if set(family_counts) != EXPECTED_FAMILIES or any(
        family_counts[family] != 300 for family in EXPECTED_FAMILIES
    ):
        raise ValueError(f"family denominators mismatch: {dict(family_counts)}")
    if any(
        family_size_counts[(family, size)] != 100
        for family in EXPECTED_FAMILIES
        for size in EXPECTED_SIZES
    ):
        raise ValueError("family/size cells must each contain 100 queries across two seeds")
    for shape, seed_payloads in fingerprints_by_shape.items():
        if len(seed_payloads) != 2:
            raise ValueError(f"expected two seed caches for family group {shape}")
        if seed_payloads[0] == seed_payloads[1]:
            raise ValueError(f"two seed caches are identical for family group {shape}")

    result = {
        "status": "pass",
        "cache_keys": sorted(cache_keys),
        "cache_files": file_rows,
        "cache_families": file_families,
        "unique_positive_queries": 1800,
        "families": {key: family_counts[key] for key in sorted(family_counts)},
        "sizes": sorted(EXPECTED_SIZES),
        "queries_per_family_size": 100,
        "raw_feature_pairs_checked": raw_pairs,
        "feature_label_pairs_checked": label_pairs,
        "type_feat32_pairs_checked": signature_pairs,
        "raw_feature_mismatches": 0,
        "feature_label_mismatches": 0,
        "type_feat32_mismatches": 0,
        "pruning_source": "query_payload_v1",
        "signature": "type_feat32",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
