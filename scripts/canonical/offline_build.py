"""Manifests of the offline builds that write into a store (plan E17, offline cost).

The label index (cascade build-label-index), the ranker inputs (rankers build) and the block embeddings
(serving embed-blocks) write their artifacts next to the store's data. Given --run-dir, a build runs inside
a manifest.RunManifest (experiment "offline_build") in that directory:

  extra.wall_seconds     the build's wall time (RunManifest)
  extra.peak_memory      the OS peak memory of the process (partition.self_peak_memory)
  extra.artifact_bytes   the bytes of every file the build wrote
  artifacts.json         an output of the run: each such file (path relative to the store directory, bytes,
                         SHA-256, or a fingerprint above manifest.FULL_HASH_LIMIT_BYTES)

The reducer's E17 reads these manifests. A build without --run-dir writes no manifest.

offline_cost() reads the cost of any offline manifest by its stage, including the two that are not built
here: prepare_data's run (the store) and train_encoder's run (an encoder). The names it needs from those
modules are repeated below and kept equal by test_offline_build, so the reducer imports neither.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from . import manifest

EXPERIMENT = "offline_build"
ARTIFACTS = "artifacts.json"
STAGES = ("label_index", "rankers", "embeddings")
PREPARE_STAGES = ("load", "partition", "fine", "relabel", "store", "sigma", "alternatives", "report")
TRAIN_LOGS = ("train_log.jsonl", "val_metrics.jsonl")  # train_encoder.TRAIN_LOG and VAL_METRICS
LATER_BUILDS = ("rankers", "embeddings")  # part/<p>/<dir>: built after prepare, counted by their own manifests
# An epoch is quiet when other work kept fewer than QUIET_OTHER_CPUS logical CPUs busy on average beside it
# (train_encoder's host_busy_cpus minus its own cpu_seconds / seconds). A run with fewer than
# QUIET_MIN_EPOCHS quiet epochs has no quiet estimate.
QUIET_OTHER_CPUS = 1.0
QUIET_MIN_EPOCHS = 5


def artifact_entries(root: str | os.PathLike, paths: Iterable[str | os.PathLike]) -> dict[str, dict[str, Any]]:
    """Every file under `paths` (files or directories), keyed by its path relative to `root`."""
    root = Path(root).resolve()
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{path}: the build wrote no such artifact")
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for file in files:
            entry = manifest.sha256_file(file, fast=file.stat().st_size > manifest.FULL_HASH_LIMIT_BYTES)
            entry.pop("path", None)
            out[file.relative_to(root).as_posix()] = entry
    return out


@contextmanager
def build_run(run_dir: str | os.PathLike | None, *, stage: str, store_dir: str | os.PathLike, dataset: str,
              config: Mapping[str, Any], inputs: Mapping[str, str | os.PathLike | None] | None = None,
              argv: Sequence[str] | None = None) -> Iterator[list[Path]]:
    """Yields the list the build appends its artifact files and directories to."""
    written: list[Path] = []
    if run_dir is None:
        yield written
        return
    if stage not in STAGES:
        raise ValueError(f"unknown offline build stage {stage!r}; known: {', '.join(STAGES)}")
    from .partition import self_peak_memory

    run_dir = Path(run_dir)
    (run_dir / ARTIFACTS).unlink(missing_ok=True)  # a failed rebuild must not keep the last build's list
    config = {"stage": stage, "dataset": dataset, "store_dir": str(store_dir), **dict(config)}
    with manifest.RunManifest(run_dir, experiment=EXPERIMENT, config=config, inputs=inputs or {},
                              argv=argv) as run:
        yield written
        entries = artifact_entries(store_dir, written)
        (run_dir / ARTIFACTS).write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        run.note(stage=stage, artifacts=len(entries), artifact_bytes=sum(int(e["bytes"]) for e in entries.values()),
                 peak_memory=self_peak_memory())


def peak_bytes(value: Any) -> int | None:
    """The largest byte count of a peak-memory record: a number, or a dict such as self_peak_memory()'s."""
    if isinstance(value, Mapping):
        found = [int(v) for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return max(found) if found else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return None


def _logged_seconds(path: Path) -> float:
    return sum(float(r.get("seconds") or 0.0) for r in _jsonl(path))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quiet_training(run_dir: str | os.PathLike) -> dict[str, Any]:
    """The training time a run would take without other work beside it.

    epochs      the lines of train_log.jsonl; measured: those with cpu_seconds and host_busy_cpus
    quiet       the measured epochs whose other load (host_busy_cpus - cpu_seconds / seconds) is below
                QUIET_OTHER_CPUS; other_cpus_median: the median other load of the measured epochs
    wall_seconds    the median quiet epoch times the epochs, plus the validation seconds, or None with
                fewer than QUIET_MIN_EPOCHS quiet epochs
    """
    run_dir = Path(run_dir)
    epochs = _jsonl(run_dir / TRAIN_LOGS[0])
    other: list[tuple[float, float]] = []
    for r in epochs:
        seconds, cpu, busy = r.get("seconds"), r.get("cpu_seconds"), r.get("host_busy_cpus")
        if seconds and cpu is not None and busy is not None:
            other.append((float(seconds), float(busy) - float(cpu) / float(seconds)))
    quiet = sorted(s for s, load in other if load < QUIET_OTHER_CPUS)
    loads = sorted(load for _, load in other)
    out: dict[str, Any] = {"epochs": len(epochs), "measured": len(other), "quiet": len(quiet),
                           "other_cpus_median": _median(loads), "epoch_seconds_median": _median(quiet),
                           "wall_seconds": None}
    if len(quiet) >= QUIET_MIN_EPOCHS:
        out["wall_seconds"] = out["epoch_seconds_median"] * len(epochs) + _logged_seconds(run_dir / TRAIN_LOGS[1])
    return out


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    mid = len(values) // 2
    return float(values[mid]) if len(values) % 2 else (float(values[mid - 1]) + float(values[mid])) / 2.0


def offline_cost(stage: str, path: str | os.PathLike, doc: Mapping[str, Any]) -> dict[str, Any]:
    """wall_seconds, bytes, peak_bytes, parts and disk of the complete offline manifest `doc` of run dir `path`.

    prepare     parts: each stage's marker (stages/<stage>.json, an output of the run), since a resumed
                prepare's manifest times only its last call; wall_seconds is their sum. disk: the store files
                the manifest lists (extra.store_files) by graph, sigma, meta and partition/<p>, without the
                ranker inputs and block embeddings built later under part/<p>/; bytes is their sum.
    training    the epoch and validation seconds of the kept trajectory (TRAIN_LOGS), which a resume keeps;
                no bytes (the export lives in the models directory); quiet: quiet_training(path).
    label_index, rankers, embeddings    extra.wall_seconds, artifact_bytes and peak_memory (build_run).
    any other   extra.wall_seconds and the bytes of the manifest's outputs.
    """
    path = Path(path)
    extra = doc.get("extra") or {}
    cost: dict[str, Any] = {"wall_seconds": float(extra.get("wall_seconds") or 0.0), "bytes": None,
                            "peak_bytes": None, "parts": {}, "disk": {}}
    if stage == "prepare":
        for name in PREPARE_STAGES:
            marker = path / "stages" / f"{name}.json"
            if marker.is_file():
                entry = json.loads(marker.read_text(encoding="utf-8"))
                cost["parts"][name] = {"wall_seconds": float(entry.get("wall_seconds") or 0.0),
                                       "peak_bytes": peak_bytes(entry.get("peak_rss_bytes"))}
        if not cost["parts"]:
            raise ValueError(f"{path}: a prepare run without stage markers")
        for rel, entry in sorted((extra.get("store_files") or {}).items()):
            parts = rel.split("/")
            if parts[0] == "part" and len(parts) > 2 and parts[2] in LATER_BUILDS:
                continue
            component = (parts[0] if parts[0] in ("graph", "sigma") else
                         f"partition/{parts[1]}" if parts[0] == "part" and len(parts) > 2 else "meta")
            cost["disk"][component] = cost["disk"].get(component, 0) + int(entry.get("bytes") or 0)
        cost["wall_seconds"] = sum(p["wall_seconds"] for p in cost["parts"].values())
        cost["bytes"] = sum(cost["disk"].values()) if cost["disk"] else None
        peaks = [p["peak_bytes"] for p in cost["parts"].values() if p["peak_bytes"] is not None]
        cost["peak_bytes"] = max(peaks) if peaks else None
    elif stage == "training":
        if not all((path / name).is_file() for name in TRAIN_LOGS):
            raise ValueError(f"{path}: a training run without {' and '.join(TRAIN_LOGS)}")
        cost["wall_seconds"] = sum(_logged_seconds(path / name) for name in TRAIN_LOGS)
        cost["quiet"] = quiet_training(path)
    elif stage in STAGES:
        if doc.get("experiment") != EXPERIMENT:
            raise ValueError(f"{path}: {stage} is read from an {EXPERIMENT} manifest, not {doc.get('experiment')!r}")
        cost["bytes"] = int(extra["artifact_bytes"])
        cost["peak_bytes"] = peak_bytes(extra.get("peak_memory"))
    else:
        cost["bytes"] = sum(int(o.get("bytes") or 0) for o in (doc.get("outputs") or {}).values())
    return cost
