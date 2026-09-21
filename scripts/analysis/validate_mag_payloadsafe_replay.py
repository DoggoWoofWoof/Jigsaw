"""Validate the serving-safe replay of the 82 non-planted MAG acceptances."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    args = parser.parse_args()

    accepted = read_rows(args.audit_dir / "canonical_accepted_queries.csv")
    expected = {
        (row["seed"], row["query_id"]): row
        for row in accepted
        if row["validation_route"] == "returned_mapping_raw_guard_rerun"
    }
    if len(expected) != 82:
        raise RuntimeError(f"Expected 82 replay cases, found {len(expected)}")

    observed: dict[tuple[str, str], dict[str, str]] = {}
    replay_files = sorted(args.replay_dir.glob("payloadsafe_20*.csv"))
    if len(replay_files) != 4:
        raise RuntimeError(f"Expected four replay CSVs, found {len(replay_files)}")
    for path in replay_files:
        match = re.search(r"(2026060[78])", path.name)
        if not match:
            raise RuntimeError(f"Seed missing from replay filename: {path.name}")
        seed = match.group(1)
        for row in read_rows(path):
            key = (seed, row["query_id"])
            if key in observed:
                raise RuntimeError(f"Duplicate replay row: {key}")
            observed[key] = row

    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise RuntimeError(f"Replay identity mismatch: missing={missing[:5]} extra={extra[:5]}")

    pair_count = 0
    for key, row in observed.items():
        source = expected[key]
        pairs = int(row["solver_mapping_pairs"])
        checks = {
            "budget": int(row["budget"]) == int(source["solved_budget"]),
            "query_nodes": int(row["query_nodes"]) == int(source["query_nodes"]),
            "payload_pruning": row["query_pruning_source"] == "query_payload_v1",
            "serving_safe_signature": row["signature"] == "type_feat32",
            "found": as_bool(row["solver_found"]),
            "not_timed_out": not as_bool(row["solver_timed_out"]),
            "raw_equal": as_bool(row["solver_raw_feature_equal"]),
            "zero_mismatches": int(row["solver_raw_feature_mismatches"]) == 0,
            "complete_mapping": pairs == int(source["query_nodes"]),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Replay validation failed for {key}: {checks}")
        pair_count += pairs

    summary = {
        "replayed_acceptances": len(observed),
        "returned_mapping_pairs": pair_count,
        "query_pruning_source": "query_payload_v1",
        "signature": "type_feat32",
        "raw_feature_mismatches": 0,
        "status": "pass",
    }
    output = args.replay_dir / "payloadsafe_replay_validation.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
