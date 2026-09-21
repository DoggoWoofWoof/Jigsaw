"""Validate deterministic MAG negative-query caches before serving-safe reruns."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_overlap_glasgow_cascade as cascade
from benchmark_retrieval import load_named_data


FAMILIES = {"negative_label", "negative_structure"}
SIZES = {20, 50, 100}


def fingerprint(item: dict) -> str:
    query = item["query"]
    digest = hashlib.sha256()
    digest.update(str(item["query_id"]).encode())
    digest.update(query.x.detach().cpu().contiguous().numpy().tobytes())
    digest.update(query.edge_index.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.cache_dir.glob("*_queries.pt"))
    if len(files) != 2:
        raise ValueError(f"Expected two negative cache files, found {len(files)}")
    data = load_named_data("mag", args.data_root)
    counts: Counter[tuple[str, int]] = Counter()
    file_rows: dict[str, int] = {}
    seed_payloads: list[set[str]] = []
    raw_equal_pairs = 0
    altered_label_queries = 0

    for path in files:
        rows = cascade.torch_load_any(path)
        if len(rows) != 300:
            raise ValueError(f"{path.name}: expected 300 rows, found {len(rows)}")
        file_rows[path.name] = len(rows)
        seen: set[str] = set()
        for item in rows:
            family = str(item["query_type"])
            size = int(item["target_query_size"])
            if family not in FAMILIES or size not in SIZES:
                raise ValueError(f"Unexpected cell {(family, size)}")
            if not bool(item.get("is_negative")) or bool(item.get("expected_match", True)):
                raise ValueError(f"Negative flags missing for {item['query_id']}")
            query = item["query"]
            nodes = item["query_nodes"].detach().cpu().long()
            if int(query.num_nodes) != int(nodes.numel()):
                raise ValueError(f"Witness length mismatch for {item['query_id']}")
            target_x = data.x[nodes].detach().cpu()
            query_x = query.x.detach().cpu()
            equal_rows = torch.all(query_x == target_x, dim=1)
            if family == "negative_structure":
                if not bool(equal_rows.all()):
                    raise ValueError(f"Structure negative changed attributes: {item['query_id']}")
                raw_equal_pairs += int(nodes.numel())
            else:
                if int((~equal_rows).sum()) != 1:
                    raise ValueError(f"Label negative must alter one node: {item['query_id']}")
                altered_label_queries += 1
            counts[(family, size)] += 1
            token = fingerprint(item)
            if token in seen:
                raise ValueError(f"Duplicate payload in {path.name}: {item['query_id']}")
            seen.add(token)
        seed_payloads.append(seen)

    if seed_payloads[0] == seed_payloads[1]:
        raise ValueError("The two negative-query seed caches are identical")
    if any(counts[(family, size)] != 100 for family in FAMILIES for size in SIZES):
        raise ValueError(f"Negative family/size denominators mismatch: {dict(counts)}")

    result = {
        "status": "pass",
        "cache_files": file_rows,
        "unique_negative_queries": 600,
        "queries_per_family_size": 100,
        "altered_label_queries": altered_label_queries,
        "structure_negative_raw_equal_pairs": raw_equal_pairs,
        "pruning_source": "query_payload_v1",
        "signature": "type_feat32",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
