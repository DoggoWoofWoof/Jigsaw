"""Build the targeted raw-attribute audit for canonical learned-Jigsaw MAG accepts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "benchmarks" / "rebuttal" / "mag_raw_acceptance_audit"
SOURCES = (
    ROOT
    / "runs"
    / "lightning_completion"
    / "mag_walkaware_remaining_v2"
    / "final_per_query",
    ROOT
    / "runs"
    / "lightning_completion"
    / "mag_targeted_v1_final"
    / "results",
)
EXPECTED_FAMILY_SOLVES = {
    "single": 296,
    "k_hop": 280,
    "degree_k_hop": 278,
    "multi_fine": 300,
    "random_walk": 213,
    "multi_coarse": 228,
}
QUERY_TYPE_GROUPS = (
    "single,k_hop,degree_k_hop,multi_fine",
    "random_walk,multi_coarse",
)


def clean_tag(text: str) -> str:
    return str(text).replace(",", "_").replace(" ", "").replace("/", "_")


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def io_path(path: Path) -> str:
    resolved = str(path.resolve())
    if len(resolved) >= 240 and not resolved.startswith("\\\\?\\"):
        resolved = "\\\\?\\" + resolved
    return resolved


def open_csv(path: Path):
    return open(io_path(path), newline="", encoding="utf-8-sig")


def source_files() -> list[Path]:
    easy = sorted(SOURCES[0].glob("*.csv"))
    hard = sorted(SOURCES[1].glob("*neural_component*_per_query.csv"))
    files = easy + hard
    if len(files) != 4:
        raise RuntimeError(f"Expected four canonical source files, found {len(files)}")
    return files


def seed_from_path(path: Path) -> int:
    match = re.search(r"2026060[78]", path.name)
    if not match:
        raise ValueError(f"Cannot recover seed from {path}")
    return int(match.group(0))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(io_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    grouped: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    provenance = []
    for path in source_files():
        seed = seed_from_path(path)
        with open_csv(path) as handle:
            rows = list(csv.DictReader(handle))
        selected = [row for row in rows if row.get("model") == "mag_walkaware_best"]
        if not selected:
            raise RuntimeError(f"No mag_walkaware_best rows in {path}")
        for row in selected:
            key = (seed, row["query_type"], row["query_id"])
            grouped[key].append(row)
        provenance.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
                "selected_rows": len(selected),
            }
        )

    if len(grouped) != 1800:
        raise RuntimeError(f"Expected 1,800 canonical queries, found {len(grouped)}")

    accepted = []
    for (seed, query_type, query_id), rows in sorted(grouped.items()):
        solved = [row for row in rows if as_bool(row.get("cascade_first_solved"))]
        if len(solved) > 1:
            raise RuntimeError(f"Multiple first-solved rows for {seed}:{query_id}")
        if not solved:
            continue
        row = solved[0]
        accepted.append(
            {
                "seed": seed,
                "query_id": query_id,
                "query_type": query_type,
                "target_query_size": int(row["target_query_size"]),
                "query_nodes": int(row["query_nodes"]),
                "solved_budget": int(row["budget"]),
                "pruned_node_fullcov": as_bool(row.get("pruned_node_fullcov")),
                "pruned_missed_node_count": int(row.get("pruned_missed_node_count") or 0),
                "signature_candidate_nodes": int(row["signature_candidate_nodes"]),
                "pruned_candidate_nodes": int(row["pruned_candidate_nodes"]),
                "component_solver_components": int(row["component_solver_components"]),
                "component_solver_nodes": int(row["component_solver_nodes"]),
                "target_nodes": int(row["target_nodes"]),
                "validation_route": (
                    "candidate_contains_planted_raw_witness"
                    if as_bool(row.get("pruned_node_fullcov"))
                    else "returned_mapping_raw_guard_rerun"
                ),
            }
        )

    family_counts = Counter(row["query_type"] for row in accepted)
    if dict(family_counts) != EXPECTED_FAMILY_SOLVES:
        raise RuntimeError(
            f"Canonical family solves changed: got={dict(family_counts)} "
            f"expected={EXPECTED_FAMILY_SOLVES}"
        )
    if len(accepted) != 1595:
        raise RuntimeError(f"Expected 1,595 accepts, found {len(accepted)}")

    planted = [row for row in accepted if row["pruned_node_fullcov"]]
    rerun = [row for row in accepted if not row["pruned_node_fullcov"]]
    if len(planted) != 1513 or len(rerun) != 82:
        raise RuntimeError(
            f"Unexpected audit split: planted={len(planted)} rerun={len(rerun)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(accepted[0])
    with (OUTPUT_DIR / "canonical_accepted_queries.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(accepted)

    for seed in (20260607, 20260608):
        selected = [row for row in rerun if row["seed"] == seed]
        with (OUTPUT_DIR / f"rerun_query_ids_seed{seed}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "query_id",
                    "query_type",
                    "target_query_size",
                    "solved_budget",
                    "query_nodes",
                ],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(selected)
        for query_types in QUERY_TYPE_GROUPS:
            families = set(query_types.split(","))
            grouped_rows = [row for row in selected if row["query_type"] in families]
            with (OUTPUT_DIR / f"rerun_query_ids_seed{seed}_{clean_tag(query_types)}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "query_id",
                        "query_type",
                        "target_query_size",
                        "solved_budget",
                        "query_nodes",
                    ],
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(grouped_rows)

    summary = {
        "canonical_queries": len(grouped),
        "accepted_queries": len(accepted),
        "accepted_query_node_pairs": sum(row["query_nodes"] for row in accepted),
        "candidate_contains_planted_raw_witness": len(planted),
        "candidate_contains_planted_raw_witness_pairs": sum(
            row["query_nodes"] for row in planted
        ),
        "requires_returned_mapping_rerun": len(rerun),
        "requires_returned_mapping_rerun_pairs": sum(row["query_nodes"] for row in rerun),
        "rerun_by_seed": dict(Counter(str(row["seed"]) for row in rerun)),
        "rerun_by_family": dict(Counter(row["query_type"] for row in rerun)),
        "rerun_by_size": dict(Counter(str(row["target_query_size"]) for row in rerun)),
        "canonical_family_accepts": dict(family_counts),
        "sources": provenance,
    }
    (OUTPUT_DIR / "preflight_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
