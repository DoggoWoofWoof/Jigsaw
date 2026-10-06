"""E3x input: the union of the queries E1 left open for the extension arms, derived from the records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.canonical import extension_uids as ex
from scripts.canonical import manifest, protocol
from scripts.canonical import run_experiment as rx

FIRST = int(protocol.SEEDS["training"][0])
JIG, FA = f"jigsaw@{FIRST}", "filterall"


@pytest.fixture(autouse=True)
def synth(monkeypatch):
    monkeypatch.setitem(protocol.DATASETS, "synth", dict(protocol.DATASETS["corafull"], partitions=6))


def _record(uid: str, status: str, **kw) -> dict:
    rec = {"uid": uid, "status": status, "time_decided": None, "witness_check": None}
    if status == "FOUND":
        rec.update(witness_check="ok", time_decided=1.0)
    if status == "NONE":
        rec["time_decided"] = 2.0
    rec.update(kw)
    return rec


def _run(root: Path, name: str, records: dict, *, T: float = 120.0, status: str = "complete",
         split: str = "test", shard=None, experiment: str = "E1") -> Path:
    d = root / name
    for (regime, arm), recs in records.items():
        folder = d / "records" / regime
        folder.mkdir(parents=True, exist_ok=True)
        suffix = f".shard{shard[0]}of{shard[1]}" if shard else ""
        with open(folder / f"{arm}{suffix}.jsonl", "w", encoding="utf-8") as handle:
            for rec in recs:
                handle.write(json.dumps(rec) + "\n")
    (d / manifest.MANIFEST_NAME).write_text(json.dumps({"status": status, "split": split, "config": {
        "experiment": experiment, "dataset": "synth", "T": T, "shard": list(shard) if shard else None}}),
        encoding="utf-8")
    return d


def test_the_union_of_open_queries_over_both_arms(tmp_path):
    r0 = {("R0", JIG): [_record("a", "FOUND"), _record("b", "TIMEOUT"), _record("c", "NONE"), _record("d", "ERROR"),
                        _record("e", "FOUND", witness_check="node type differs")],
          ("R0", FA): [_record("a", "TIMEOUT"), _record("b", "TIMEOUT"), _record("c", "NONE"), _record("d", "NONE"),
                       _record("e", "FOUND")]}
    r1 = {("R1", JIG): [_record("f", "INCOMPLETE")], ("R1", FA): [_record("f", "FOUND")]}
    one = _run(tmp_path, "shard0", {**r0}, shard=(0, 2))
    two = _run(tmp_path, "shard1", {**r1}, shard=(1, 2))
    out = tmp_path / "e3x" / "uids.txt"
    assert ex.main(["--e1", str(one), str(two), "--dataset", "synth", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").split() == ["a", "b", "f"]
    doc = json.loads(Path(f"{out}.json").read_text(encoding="utf-8"))
    assert doc["schema"] == ex.SCHEMA and doc["arms"] == [JIG, FA] and doc["split"] == "test"
    assert doc["e1_T"] == 120.0 and doc["extension_T"] == float(rx.experiment_plan("E3x", "synth")["T"])
    assert doc["counts"]["R0"] == {"queries": 5, "extension": 2, "open": {JIG: 1, FA: 2},
                                   "errors": {JIG: 1, FA: 0}, "decided": {JIG: 2, FA: 3}}
    assert doc["uids_sha256"] == manifest.sha256_file(out)["sha256"] and len(doc["e1_runs"]) == 2
    assert doc["uids"] == ["a", "b", "f"] and doc["counts"]["R1"]["extension"] == 1


def test_incomplete_or_foreign_runs_are_refused(tmp_path):
    good = {("R0", JIG): [_record("a", "TIMEOUT")], ("R0", FA): [_record("a", "NONE")]}
    cases = {
        "not complete": _run(tmp_path, "running", good, status="running"),
        "not E1's": _run(tmp_path, "t600", good, T=600.0),
        "not E1 on synth": _run(tmp_path, "e9", good, experiment="E9"),
        "different queries": _run(tmp_path, "unpaired", {("R0", JIG): [_record("a", "TIMEOUT")],
                                                         ("R0", FA): [_record("b", "NONE")]}),
    }
    for match, d in cases.items():
        with pytest.raises(SystemExit, match=match):
            ex.main(["--e1", str(d), "--dataset", "synth", "--out", str(tmp_path / "u.txt")])
    dev = _run(tmp_path, "dev", good, split="dev")
    test = _run(tmp_path, "test", {("R1", JIG): [_record("x", "TIMEOUT")], ("R1", FA): [_record("x", "NONE")]})
    with pytest.raises(SystemExit, match="different splits"):
        ex.main(["--e1", str(dev), str(test), "--dataset", "synth", "--out", str(tmp_path / "u.txt")])
    twice = _run(tmp_path, "twice", good)
    with pytest.raises(SystemExit, match="appears twice"):
        ex.main(["--e1", str(twice), str(twice), "--dataset", "synth", "--out", str(tmp_path / "u.txt")])
    assert not (tmp_path / "u.txt").exists()
