"""E2 driver: one measured process per (system, mode, cache, cap, repeat), resumable, sealed rules."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from scripts.canonical import cascade, memprobe, protocol, queries, rankers, seal
from scripts.canonical import run_experiment as rx
from scripts.canonical import run_external as rex
from scripts.canonical import store as store_mod
from scripts.canonical import streaming as st

FIRST = int(protocol.SEEDS["training"][0])
JIGSAW_SIDE = [(f"jigsaw@{FIRST}", "streaming"), ("jigsaw_index", "streaming"), ("filterall", "streaming"),
               ("filterall", "resident"), ("direct", "resident")]
# the child sees the test dataset the way the in-process tests do
WRAPPER = ("import sys; from scripts.canonical import protocol; "
           "protocol.DATASETS['synth'] = dict(protocol.DATASETS['corafull'], partitions=6); "
           "from scripts.canonical import run_experiment as rx; sys.exit(rx.main(sys.argv[1:]))")


@pytest.fixture(scope="module")
def e2store(clustered_store, tmp_path_factory):
    root, _ = clustered_store
    copy = tmp_path_factory.mktemp("streaming")
    shutil.copytree(root / "synth", copy / "synth")
    gstore = store_mod.GraphStore(copy, "synth", mmap=True)
    rankers.build_artifacts(gstore, ["R0"])
    cascade.build_label_index(gstore, "R0")
    for split, master in (("dev", "20260607"), ("test", "sealed-for-tests")):
        queries.SplitBuilder(gstore, "synth", split, master, sizes=[8], max_per_cell=1).build(
            ["R0"], copy / "queries" / split, master_tag=master)
    return copy


@pytest.fixture
def synth(monkeypatch):
    monkeypatch.setitem(protocol.DATASETS, "synth", dict(protocol.DATASETS["corafull"], partitions=6))


def _argv(copy: Path, e2: Path, *extra: str, split: str = "dev") -> list[str]:
    return ["--store-root", str(copy), "--dataset", "synth", "--queries", str(copy / "queries" / split),
            "--e2-root", str(e2), *extra]


def _args(copy: Path, e2: Path, *extra: str, split: str = "dev"):
    return st.build_parser().parse_args(_argv(copy, e2, *extra, split=split))


# ----------------------------------------------------------------------------
# The grid
# ----------------------------------------------------------------------------
def test_cells_cover_the_plan_in_rounds(synth, tmp_path):
    assert st.parse_caps(None) == [None, *(float(c) for c in protocol.SERVING["memory_caps_gb"])]
    assert st.parse_caps(["none", "4"]) == [None, 4.0]
    for bad, match in ((["0"], "positive"), (["2", "2.0"], "twice")):
        with pytest.raises(SystemExit, match=match):
            st.parse_caps(bad)
    cells = st.plan_cells("synth", caps=[None, 2.0], caches=None, repeats=2, external=True)
    external = [(m, "resident") for m in rex.experiment_plan("E2", "synth")["methods"]]
    members = JIGSAW_SIDE + external
    assert len(cells) == 2 * 2 * 2 * len(members)
    assert {(c.system, c.mode) for c in cells} == set(members)
    assert {c.runner for c in cells if (c.system, c.mode) in external} == {"run_external"}
    reps = [c.repeat for c in cells]
    assert reps == sorted(reps)  # every cell once before any cell twice
    assert len(set(cells)) == len(cells)
    cell = st.Cell("filterall", "run_experiment", "resident", "warm", None, 1)
    assert cell.directory(tmp_path, "synth") == tmp_path / "synth" / "filterall" / "resident-warm-capnone" / "rep1"
    assert st.Cell("direct", "run_experiment", "resident", "cold", 4.0, 3).directory(tmp_path, "d").parts[-2:] == \
        ("resident-cold-cap4", "rep3")
    only = st.plan_cells("synth", caps=[None], caches=["warm"], repeats=1, external=False, systems=["filterall"])
    assert [(c.system, c.mode) for c in only] == [("filterall", "streaming"), ("filterall", "resident")]
    with pytest.raises(SystemExit, match="systems outside E2"):
        st.plan_cells("synth", caps=[None], caches=None, repeats=1, external=False, systems=["classical_gql"])
    with pytest.raises(SystemExit, match="caches outside E2"):
        st.plan_cells("synth", caps=[None], caches=["none"], repeats=1, external=False)


def test_e10_measures_its_own_plan_in_its_own_root(synth, e2store, tmp_path, monkeypatch):
    copy = e2store
    monkeypatch.setitem(protocol.SERVING, "scale_out", dict(protocol.SERVING["scale_out"], datasets=["synth"]))
    e10 = rx.experiment_plan("E10", "synth")
    caps, repeats = st.plan_caps(e10), st.plan_repeats(e10)
    assert caps == [None, *(float(c) for c in protocol.SERVING["scale_out"]["memory_caps_gb"])]
    cells = st.plan_cells("synth", caps=caps, caches=None, repeats=repeats, external=True, experiment="E10")
    members = [(a, "streaming") for a in (f"jigsaw@{FIRST}", "jigsaw_index", "filterall")] + \
        [(m, "resident") for m in rex.experiment_plan("E10", "synth")["methods"]]
    assert {(c.system, c.mode) for c in cells} == set(members) and {c.cache for c in cells} == {"cold"}
    assert len(cells) == len(members) * len(caps) * repeats
    with pytest.raises(SystemExit, match="caches outside E10"):
        st.plan_cells("synth", caps=[None], caches=["warm"], repeats=1, external=False, experiment="E10")
    monkeypatch.setattr(os, "posix_fadvise", lambda *a: None, raising=False)
    monkeypatch.setattr(memprobe, "detect_cap_mechanism", lambda: "cgroup")
    calls: list = []
    monkeypatch.setattr(memprobe, "run_measured", _fake_measured(calls))
    root = tmp_path / "e10"
    argv = _argv(copy, root, "--experiment", "E10", "--external")
    planned = st.execute(st.build_parser().parse_args([*argv, "--dry-run"]))
    assert planned["cells"] == len(cells) and planned["experiment"] == "E10" and not calls
    summary = st.execute(st.build_parser().parse_args(argv))
    assert summary["measured"] == len(cells) and summary["failed"] == 0
    assert {c["memory_cap"] for c in calls} == {None, *(f"{c:g}G" for c in caps if c is not None)}
    assert all(c["cmd"][c["cmd"].index("--experiment") + 1] == "E10" for c in calls)
    assert {json.loads(p.read_text("utf-8"))["experiment"] for p in root.rglob("measure.json")} == {"E10"}
    calls.clear()
    with pytest.raises(SystemExit, match="holds E10 cells"):
        st.execute(st.build_parser().parse_args(_argv(copy, root, "--caps", "none", "--caches", "warm")))
    assert not calls
    # a sealed E10 sweep takes the plan's repeats and caps, not E2's
    monkeypatch.setattr(seal, "UNSEAL_LOG", tmp_path / "UNSEAL_LOG.jsonl")
    for extra, match in ((("--repeats", str(protocol.SERVING["repeats"])), "the plan's is"),
                         (("--caps", "none", "8"), "outside the protocol")):
        with pytest.raises(SystemExit, match=match):
            st.main(_argv(copy, tmp_path / "sealed", "--experiment", "E10", *extra, "--unseal", split="test"))
    with pytest.raises(SystemExit, match="not frozen"):
        st.main(_argv(copy, tmp_path / "sealed", "--experiment", "E10", "--unseal", split="test"))
    monkeypatch.setitem(protocol.SERVING, "scale_out", dict(protocol.SERVING["scale_out"], datasets=["papers100m"]))
    with pytest.raises(SystemExit, match="E10 runs on papers100m"):
        st.main(_argv(copy, tmp_path / "other", "--experiment", "E10", "--caps", "none"))
    assert not (tmp_path / "UNSEAL_LOG.jsonl").exists()


def test_child_commands_name_the_cell(synth, e2store, tmp_path):
    copy = e2store
    xcache = tmp_path / "xcache"
    xcache.mkdir()
    args = _args(copy, tmp_path, "--solver-bin", "glasgow", "--external-binary", "sm", "--external-cache-dir",
                 str(xcache), "--limit", "3", "--per-cell", "2", "--fast-hashes")
    run_dir = tmp_path / "run"
    jig = st.child_command(st.Cell("jigsaw_index", "run_experiment", "streaming", "warm", 2.0, 1), args, run_dir)
    assert jig[:3] == [sys.executable, "-m", "scripts.canonical.run_experiment"]
    pairs = dict(zip(jig, jig[1:]))
    for flag, value in (("--experiment", "E2"), ("--arms", "jigsaw_index"), ("--mode", "streaming"),
                        ("--cache", "warm"), ("--out", str(run_dir)), ("--solver-bin", "glasgow"),
                        ("--limit", "3"), ("--per-cell", "2"), ("--dataset", "synth")):
        assert pairs[flag] == value
    assert "--fast-hashes" in jig and "--unseal" not in jig and "--method" not in jig
    ext = st.child_command(st.Cell("classical_gql", "run_external", "resident", "cold", None, 2), args, run_dir)
    assert ext[:3] == [sys.executable, "-m", "scripts.canonical.run_external"]
    assert ext[ext.index("--method") + 1] == "classical_gql" and ext[ext.index("--binary") + 1] == "sm"
    assert "--mode" not in ext and "--arms" not in ext and "--solver-bin" not in ext
    assert ext[ext.index("--per-cell") + 1] == "2"
    cold = st.Cell("classical_gql", "run_external", "resident", "cold", None, 1)
    assert st.evict_paths(cold, args) == [copy / "synth", xcache]
    assert st.evict_paths(st.Cell("filterall", "run_experiment", "resident", "cold", None, 1), args) == [copy / "synth"]
    assert st.evict_paths(st.Cell("filterall", "run_experiment", "resident", "warm", None, 1), args) == []
    sealed = _args(copy, tmp_path, "--unseal", split="test")
    cmd = st.child_command(st.Cell("direct", "run_experiment", "resident", "cold", 8.0, 3), sealed, run_dir)
    assert "--unseal" in cmd and cmd[cmd.index("--reason") + 1] == "E2 direct resident cold cap 8 rep 3"


def _fake_measured(calls: list, oom_caps: tuple = ()):
    def run(cmd, **kw):
        out = Path(cmd[cmd.index("--out") + 1])
        calls.append({"cmd": cmd, **kw})
        oom = kw["memory_cap"] in oom_caps
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps({"status": "running" if oom else "complete"}), encoding="utf-8")
        return {"cmd": cmd, "exit_code": -9 if oom else 0, "timed_out": False, "wall_seconds": 1.5,
                "peak_rss_bytes": 3 * 2**20, "oom_killed": oom if kw["memory_cap"] else None,
                "memory_cap": {"bytes": None, "mechanism": kw["cap_mechanism"]}}
    return run


def test_sweep_measures_each_cell_once_and_resumes(synth, e2store, tmp_path, monkeypatch):
    copy = e2store
    calls: list = []
    monkeypatch.setattr(os, "posix_fadvise", lambda *a: None, raising=False)
    monkeypatch.setattr(memprobe, "detect_cap_mechanism", lambda: "cgroup")
    monkeypatch.setattr(memprobe, "run_measured", _fake_measured(calls, oom_caps=("2G",)))
    e2 = tmp_path / "e2"
    argv = _argv(copy, e2, "--caps", "none", "2", "--repeats", "2", "--external")
    planned = st.execute(st.build_parser().parse_args([*argv, "--dry-run"]))
    assert planned["measured"] == 0 and not calls and not e2.exists()
    n = planned["cells"]
    assert n == 2 * 2 * 2 * (len(JIGSAW_SIDE) + len(rex.experiment_plan("E2", "synth")["methods"]))
    summary = st.execute(st.build_parser().parse_args(argv))
    assert summary["measured"] == n and summary["failed"] == n // 2 and summary["cap_mechanism"] == "cgroup"
    assert [c["memory_cap"] for c in calls[: n // 4]] == [None] * (n // 4)  # the uncapped cells come first
    assert all(c["cap_mechanism"] == "cgroup" and c["cwd"] == st.REPO_ROOT for c in calls)
    for call in calls:
        cache = call["cmd"][call["cmd"].index("--cache") + 1]
        assert (call["evict"] != []) == (cache == "cold")
    measures = sorted(e2.rglob("measure.json"))
    assert len(measures) == n and str(planned["plan"][0]).startswith(str(e2))
    m = json.loads((e2 / "synth" / "direct" / "resident-cold-cap2" / "rep1" / "measure.json").read_text("utf-8"))
    assert m["schema"] == st.SCHEMA and m["system"] == m["arm"] == "direct" and m["runner"] == "run_experiment"
    assert (m["mode"], m["cache"], m["cap_gb"], m["repeat"]) == ("resident", "cold", 2.0, 1)
    assert m["completed"] is False and m["oom_killed"] is True and m["run_status"] == "running"
    assert m["protocol_hash"] == protocol.protocol_hash() and m["memprobe"]["peak_rss_bytes"] == 3 * 2**20
    ok = json.loads((e2 / "synth" / "classical_gql" / "resident-warm-capnone" / "rep2" / "measure.json")
                    .read_text("utf-8"))
    assert ok["completed"] is True and ok["runner"] == "run_external" and ok["run_status"] == "complete"
    # a finished sweep measures nothing again; an interrupted cell (no measure.json) is set aside and redone
    calls.clear()
    broken = e2 / "synth" / "filterall" / "streaming-warm-capnone" / "rep2"
    (broken / "measure.json").unlink()
    again = st.execute(st.build_parser().parse_args(argv))
    assert again["measured"] == 1 and again["skipped"] == n - 1 and len(calls) == 1
    assert (broken.parent / "rep2.attempt1" / "run" / "manifest.json").exists()
    # --retry-failed redoes only the cells that did not complete and keeps the failed attempt
    calls.clear()
    monkeypatch.setattr(memprobe, "run_measured", _fake_measured(calls))
    retried = st.execute(st.build_parser().parse_args([*argv, "--retry-failed"]))
    assert retried["measured"] == n // 2 and retried["failed"] == 0 and {c["memory_cap"] for c in calls} == {"2G"}
    redone = e2 / "synth" / "direct" / "resident-cold-cap2"
    assert json.loads((redone / "rep1" / "measure.json").read_text("utf-8"))["completed"] is True
    assert json.loads((redone / "rep1.attempt1" / "measure.json").read_text("utf-8"))["oom_killed"] is True


def test_setups_that_cannot_measure_are_refused(synth, e2store, tmp_path, monkeypatch):
    copy = e2store
    monkeypatch.setattr(memprobe, "run_measured", lambda *a, **k: pytest.fail("nothing may run"))
    if not hasattr(os, "posix_fadvise"):
        with pytest.raises(SystemExit, match="posix_fadvise"):
            st.main(_argv(copy, tmp_path / "e2", "--caps", "none"))
    monkeypatch.setattr(os, "posix_fadvise", lambda *a: None, raising=False)
    monkeypatch.setattr(memprobe, "detect_cap_mechanism", lambda: "none")
    with pytest.raises(SystemExit, match="cgroup v2"):
        st.main(_argv(copy, tmp_path / "e2", "--caps", "none", "4"))
    with pytest.raises(SystemExit, match="at least 1"):
        st.main(_argv(copy, tmp_path / "e2", "--caps", "none", "--repeats", "0"))
    with pytest.raises(SystemExit, match="sealed query sets only"):
        st.main(_argv(copy, tmp_path / "e2", "--caps", "none", "--unseal"))
    with pytest.raises(SystemExit, match="missing"):
        st.main(["--store-root", str(copy), "--dataset", "synth", "--queries", str(tmp_path), "--e2-root",
                 str(tmp_path / "e2"), "--caps", "none"])
    assert not (tmp_path / "e2").exists()


def test_sealed_sweeps_are_the_registered_ones(synth, e2store, tmp_path, monkeypatch):
    copy = e2store
    log = tmp_path / "UNSEAL_LOG.jsonl"
    monkeypatch.setattr(seal, "UNSEAL_LOG", log)
    monkeypatch.setattr(os, "posix_fadvise", lambda *a: None, raising=False)
    monkeypatch.setattr(memprobe, "detect_cap_mechanism", lambda: "cgroup")
    monkeypatch.setattr(memprobe, "run_measured", lambda *a, **k: pytest.fail("nothing may run"))
    e2 = tmp_path / "e2"
    refused = {("--limit", "2"): "--limit", ("--per-cell", "2"): "--per-cell", ("--router-tau", "0.5"): "--router-tau",
               ("--backend", "networkx"): "--backend", ("--repeats", "1"): "--repeats 1",
               ("--caps", "none", "3"): "outside the protocol"}
    for extra, match in refused.items():
        with pytest.raises(SystemExit, match=match):
            st.main(_argv(copy, e2, *extra, "--unseal", split="test"))
    with pytest.raises(SystemExit, match="need --unseal"):
        st.main(_argv(copy, e2, split="test"))
    # a registered sweep, or a registered part of one, reaches the gate conditions the children check
    for extra in ((), ("--systems", "filterall", "--caches", "warm", "--caps", "none", "8")):
        with pytest.raises(SystemExit, match="not frozen"):
            st.main(_argv(copy, e2, *extra, "--unseal", split="test"))
    assert not log.exists() and not e2.exists()


def test_a_real_cell_runs_the_runner_under_the_probe(synth, e2store, tmp_path, monkeypatch):
    copy = e2store
    real = st.child_command

    def wrapped(cell, args, run_dir):
        cmd = real(cell, args, run_dir)
        i = cmd.index("-m")
        return [*cmd[:i], "-c", WRAPPER, *cmd[i + 2:]]

    monkeypatch.setattr(st, "child_command", wrapped)
    e2 = tmp_path / "e2"
    argv = _argv(copy, e2, "--systems", "jigsaw_index", "filterall", "--caps", "none", "--caches", "warm",
                 "--repeats", "1", "--limit", "2", "--backend", "networkx")
    summary = st.execute(st.build_parser().parse_args(argv))
    assert summary["measured"] == 3 and summary["failed"] == 0, summary
    for system, mode in (("jigsaw_index", "streaming"), ("filterall", "streaming"), ("filterall", "resident")):
        cell = e2 / "synth" / system / f"{mode}-warm-capnone" / "rep1"
        m = json.loads((cell / "measure.json").read_text(encoding="utf-8"))
        assert m["completed"] is True, (cell / "stderr.txt").read_text(encoding="utf-8", errors="replace")[-2000:]
        assert m["memprobe"]["exit_code"] == 0 and m["memprobe"]["peak_rss_bytes"] > 2**20
        info = json.loads((cell / "run" / "manifest.json").read_text(encoding="utf-8"))
        assert info["status"] == "complete" and info["experiment"] == "E2"
        assert info["config"]["mode"] == mode and info["config"]["cache"] == "warm"
        records = rx.read_records(cell / "run" / "records" / "R0" / f"{system}+backend=networkx.jsonl")
        assert len(records) == 2 and all(r["mode"] == mode and r["cache"] == "warm" for r in records)
    assert st.execute(st.build_parser().parse_args(argv))["skipped"] == 3
