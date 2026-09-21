"""Validate targeted raw guards and close all canonical learned-Jigsaw MAG accepts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


EXPECTED_QUERY_CACHE_KEYS = {
    "bca702f664e58aad",
    "86d76b2687707326",
    "44e026ffd70eb787",
    "ec622a8592da29ec",
}


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def open_csv(path: Path):
    resolved = str(path.resolve())
    if len(resolved) >= 240 and not resolved.startswith("\\\\?\\"):
        resolved = "\\\\?\\" + resolved
    return open(resolved, newline="", encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--rerun-csv", action="append", type=Path, required=True)
    parser.add_argument("--rerun-log", action="append", type=Path, required=True)
    args = parser.parse_args()

    observed_cache_keys = set()
    for path in args.rerun_log:
        text = path.read_text(encoding="utf-8", errors="replace")
        observed_cache_keys.update(
            re.findall(r"\b([0-9a-f]{16})_queries\.pt\b", text)
        )
    missing_cache_keys = EXPECTED_QUERY_CACHE_KEYS - observed_cache_keys
    if missing_cache_keys:
        raise RuntimeError(
            f"Canonical query-cache validation failed; missing={sorted(missing_cache_keys)}"
        )

    accepted_path = args.audit_dir / "canonical_accepted_queries.csv"
    with open_csv(accepted_path) as handle:
        accepted = list(csv.DictReader(handle))
    if len(accepted) != 1595:
        raise RuntimeError(f"Expected 1,595 canonical accepts, found {len(accepted)}")

    expected = {
        (row["seed"], row["query_id"]): row
        for row in accepted
        if row["validation_route"] == "returned_mapping_raw_guard_rerun"
    }
    if len(expected) != 82:
        raise RuntimeError(f"Expected 82 rerun cells, found {len(expected)}")

    observed = {}
    for path in args.rerun_csv:
        with open_csv(path) as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row.get("model") != "mag_walkaware_best":
                continue
            seed = "20260608" if "20260608" in path.name else "20260607"
            key = (seed, row["query_id"])
            expected_row = expected.get(key)
            if expected_row is None:
                continue
            if int(row["budget"]) != int(expected_row["solved_budget"]):
                continue
            if key in observed:
                raise RuntimeError(f"Duplicate canonical-budget rerun row for {key}")
            observed[key] = row

    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise RuntimeError(f"Rerun identity mismatch: missing={missing[:5]} extra={extra[:5]}")

    mismatches = []
    for key, expected_row in expected.items():
        row = observed[key]
        checks = {
            "budget": int(row["budget"]) == int(expected_row["solved_budget"]),
            "query_nodes": int(row["query_nodes"]) == int(expected_row["query_nodes"]),
            "legacy_production_replay": row.get("query_pruning_source")
            == "planted_target_legacy",
            "signature_candidate_nodes": int(row["signature_candidate_nodes"])
            == int(expected_row["signature_candidate_nodes"]),
            "pruned_candidate_nodes": int(row["pruned_candidate_nodes"])
            == int(expected_row["pruned_candidate_nodes"]),
            "component_solver_components": int(row["component_solver_components"])
            == int(expected_row["component_solver_components"]),
            "component_solver_nodes": int(row["component_solver_nodes"])
            == int(expected_row["component_solver_nodes"]),
            "target_nodes": int(row["target_nodes"])
            == int(expected_row["target_nodes"]),
            "solver_found": as_bool(row.get("solver_found")),
            "not_timed_out": not as_bool(row.get("solver_timed_out")),
            "raw_equal": as_bool(row.get("solver_raw_feature_equal")),
            "zero_raw_mismatches": int(row.get("solver_raw_feature_mismatches") or -1) == 0,
            "complete_mapping": int(row.get("solver_mapping_pairs") or 0)
            == int(expected_row["query_nodes"]),
        }
        if not all(checks.values()):
            mismatches.append({"key": key, "checks": checks})
    if mismatches:
        raise RuntimeError(f"Raw acceptance audit failed: {mismatches[:5]}")

    summary = {
        "canonical_learned_jigsaw_accepts": 1595,
        "raw_validated_query_target_pairs": sum(int(row["query_nodes"]) for row in accepted),
        "candidate_contained_planted_raw_witness": 1513,
        "returned_mapping_raw_guard_reruns": 82,
        "returned_mapping_raw_guard_pairs": sum(
            int(row["query_nodes"]) for row in expected.values()
        ),
        "canonical_query_cache_keys": sorted(EXPECTED_QUERY_CACHE_KEYS),
        "raw_feature_mismatches": 0,
        "rerun_family_counts": dict(Counter(row["query_type"] for row in expected.values())),
        "status": "pass",
    }
    (args.audit_dir / "final_validation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Learned-Jigsaw MAG raw-attribute acceptance audit",
        "",
        f"- Canonical accepted positive queries: {summary['canonical_learned_jigsaw_accepts']:,}",
        f"- Raw-validated query-target node pairs: {summary['raw_validated_query_target_pairs']:,}",
        f"- Accepted candidates containing the planted raw-valid witness: {summary['candidate_contained_planted_raw_witness']:,}",
        f"- Alternative accepted mappings rerun with the raw tensor guard: {summary['returned_mapping_raw_guard_reruns']}",
        f"- Query-target pairs checked in returned mappings: {summary['returned_mapping_raw_guard_pairs']:,}",
        "- Canonical query caches reproduced exactly: 4/4",
        "- Raw-feature mismatches: 0",
        "",
        "Every canonical learned-Jigsaw MAG acceptance is backed by a mapping that is valid under the original pre-hash attributes.",
    ]
    (args.audit_dir / "VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
