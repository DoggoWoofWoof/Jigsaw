"""reducer: the paper's numbers from small fake run directories, and the refusals that guard them."""

from __future__ import annotations

import csv
import inspect
import itertools
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.canonical import (
    cascade,
    certify_negatives,
    manifest,
    numbers_json,
    offline_build,
    protocol,
    queries,
    rankers,
    reducer,
    run_external,
    serving,
    stats,
    streaming,
)
from scripts.canonical import failure_localize as fl
from scripts.canonical import pattern_audit as pa
from scripts.canonical import run_experiment as rx
from scripts.canonical import store as store_mod
from scripts.canonical.launch import reduce_plan as rp

REPO_ROOT = Path(__file__).resolve().parents[2]
DS = "synth"
SEEDS = (7301, 7302, 7303)
FIRST = "jigsaw@7301"
LOW_DRAWS = {"boot": 400, "perm": 2000, "quantile": 200}
UTC1, UTC2 = "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"
QUERIES = {"p0": ("positive", "single", 8), "p1": ("positive", "single", 16), "p2": ("positive", "k_hop", 8),
           "p3": ("positive", "k_hop", 16), "p4": ("positive", "random_walk", 8),
           "p5": ("positive", "random_walk", 16), "n0": ("negative", "neg_absent_label", 8),
           "n1": ("negative", "neg_structure", 8), "n2": ("negative", "neg_swapped_label", 8),
           "n3": ("negative", "neg_structure", 16), "n4": ("negative", "neg_absent_label", 16)}
POS = [f"p{i}" for i in range(6)]
SMOKE_FAMILIES = ["single", "k_hop", "neg_absent_label", "neg_swapped_label", "neg_structure"]
_names = itertools.count()


@pytest.fixture(autouse=True)
def _registered(monkeypatch):
    """The synthetic dataset in the protocol (so its hash, which every manifest records, includes it), no
    git, GPU or library probes in the manifests, and few resampling draws."""
    monkeypatch.setattr(manifest, "git_info", lambda *a, **k: {"available": False})
    monkeypatch.setattr(manifest, "machine_info", lambda: {"cpu_model": "test-cpu"})
    monkeypatch.setattr(manifest, "library_versions", lambda *a, **k: {})
    monkeypatch.setitem(protocol.DATASETS, DS, dict(protocol.DATASETS["corafull"], partitions=6, ladder=[1, 2, 3]))
    for name, value in LOW_DRAWS.items():
        monkeypatch.setitem(reducer.DRAWS, name, value)


# ----------------------------------------------------------------------------
# Fake runs
# ----------------------------------------------------------------------------
def outcome(code: str, T: float) -> dict:
    """F<t> verified FOUND, X<t> FOUND with a failed check, N<t> NONE proof, U<t> NONE without proof,
    I<t> INCOMPLETE, T TIMEOUT, E<message> ERROR."""
    if code == "T":
        return {"status": "TIMEOUT", "elapsed": float(T)}
    head, rest = code[0], code[1:]
    if head == "E":
        return {"status": "ERROR", "error": rest, "elapsed": 0.1}
    t = float(rest)
    return {"F": {"status": "FOUND", "witness_check": "ok", "time_found": t, "time_decided": t, "elapsed": t},
            "X": {"status": "FOUND", "witness_check": "failed", "elapsed": t},
            "N": {"status": "NONE", "time_decided": t, "elapsed": t},
            "U": {"status": "NONE", "elapsed": t},
            "I": {"status": "INCOMPLETE", "elapsed": t}}[head]


def rec(uid, arm, code, *, T, regime="R0", external=False, none_certified=True, **extra):
    kind, family, size = QUERIES[uid]
    method, seed = (arm, None) if external else reducer.parse_arm(arm)[:2]
    out = {"schema": reducer.RECORD_SCHEMA, "uid": uid, "dataset": DS, "regime": regime, "arm": arm,
           "method": method, "train_seed": seed, "model_id": None, "T": float(T), "kind": kind, "family": family,
           "size": size, "certified": "planted" if kind == "positive" else "uncertified", "time_found": None,
           "time_decided": None, "elapsed": None, "witness_check": None, "error": None, "steps": [],
           "stages": {"retrieval": 0.01, "solve": 0.02}, "decided_step": None, **outcome(code, T)}
    if external:  # a cascade record never has the key (failure_localize.is_external)
        out["external"] = {"none_certified": none_certified}
    out.update(extra)
    return out


def records(arm, codes, *, T, **kw):
    return [rec(uid, arm, code, T=T, **kw) for uid, code in codes.items()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def query_file(root, regime="R0", version=1):
    path = root / "queries" / f"v{version}" / f"{regime}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"query set {regime} version {version}".encode("ascii"))
    return path


def write_run(path, experiment, arms, *, T, qfile, split="test", regime="R0", partition="primary", shard=None,
              options=None, mode=None, cache=None, selected=None, extra=None):
    """A complete run directory as run_experiment (or run_external) leaves it."""
    path = Path(path)
    chosen = max(map(len, arms.values())) if selected is None else selected
    config = {"schema": reducer.RECORD_SCHEMA, "experiment": experiment, "dataset": DS, "partition": partition,
              "regimes": [regime], "arms": list(arms), "T": float(T), "shard": list(shard) if shard else None,
              "query_sets": {regime: {"path": str(qfile), "split": split, "selected": chosen}},
              "cascade_options": {regime: dict(options or {})}}
    if mode is not None:
        config["mode"] = mode
    if cache is not None:
        config["cache"] = cache
    rdir = path / "records" / regime
    rdir.mkdir(parents=True, exist_ok=True)
    for arm, recs in arms.items():
        write_jsonl(rdir / reducer.record_file(arm, shard), recs)
    manifest.write_manifest(path, experiment=experiment, config=config, inputs={f"queries/{regime}": qfile},
                            split=split)
    manifest.finalize_manifest(path, extra=extra)
    return path


def edit_json(path, change):
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")


def set_manifest(run, **fields):
    edit_json(run / manifest.MANIFEST_NAME, lambda d: d.update(fields))


def set_selected(run, n, regime="R0"):
    edit_json(run / manifest.MANIFEST_NAME, lambda d: d["config"]["query_sets"][regime].update(selected=n))


def edit_records(run, arm, change, regime="R0", shard=None):
    """Rewrite an arm's records and re-finalize the run, so only the reducer's own checks can object."""
    path = run / "records" / regime / reducer.record_file(arm, shard)
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    write_jsonl(path, change(recs))
    manifest.finalize_manifest(run)


def set_records(run, arm, uid=None, **fields):
    def change(recs):
        for r in recs:
            if uid is None or r["uid"] == uid:
                r.update(fields)
        return recs

    edit_records(run, arm, change)


_AUTO = object()


def cert_entry(certified, by=None, seconds=None, *, witness_check=_AUTO, rejected=None, errors=None):
    out = {"certified": certified, "by": by, "seconds": dict(seconds or {}),
           "witness_check": ("ok" if certified == "found" else None) if witness_check is _AUTO else witness_check}
    if rejected is not None:
        out["rejected"] = rejected
    if errors is not None:
        out["errors"] = errors
    return out


def write_cert(path, regime, entries, qfile, *, split="test", complete=True, validate=True, protocol_hash=None):
    """A jigsaw-certification-v1 file for the query set qfile."""
    doc = {"schema": reducer.CERT_SCHEMA, "dataset": DS, "regime": regime, "split": split, "partition": "primary",
           "query_set": {"path": str(qfile), "sha256": manifest.sha256_file(qfile)["sha256"], "master": "m"},
           "protocol_hash": protocol_hash or protocol.protocol_hash(), "stages": [], "complete": complete,
           "counts": {}, "queries": {uid: {"family": QUERIES[uid][1], "size": QUERIES[uid][2], **e}
                                     for uid, e in entries.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    if validate and complete:
        certify_negatives.load_certification(path, query_set_sha256=doc["query_set"]["sha256"])
    return path


def entry(experiment, *dirs):
    return {"experiment": experiment, "dataset": DS, "dirs": [str(d) for d in dirs]}


def write_plan(tmp, plan, name):
    path = tmp / f"plan-{name}.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def run_plan(tmp, plan, flags=None, *, name="out"):
    code = reducer.execute(write_plan(tmp, plan, name), tmp / name, flags)
    return code, tmp / name, numbers_json.load(tmp / name / "numbers.json")


def refused(tmp, plan, match, flags=None):
    name = f"refused{next(_names)}"
    with pytest.raises(reducer.ReduceError, match=re.escape(match)):
        reducer.execute(write_plan(tmp, plan, name), tmp / name, flags)
    assert not (tmp / name).exists()  # nothing is written before every check passed


def val(doc, key):
    return doc["numbers"][key]["value"]


def assert_rate(number, successes, n):
    """A rate in percent with the Wilson interval of `successes` (a float for a seed-averaged rate) in n."""
    lo, hi = stats.wilson_interval(successes, n)
    assert number["value"] == pytest.approx(100.0 * successes / n)
    assert number["ci"] == pytest.approx([100.0 * lo, 100.0 * hi])
    assert number["n"] == n


def read_csv(path):
    return list(csv.reader(path.read_text(encoding="utf-8").splitlines()))


# ----------------------------------------------------------------------------
# Seed-averaged rates: Wilson of the mean over queries of the per-query seed means
# ----------------------------------------------------------------------------
def test_mean_rate_is_wilson_of_the_per_query_seed_means():
    p, lo, hi = reducer.mean_rate(np.array([[1, 0, 1, 1], [1, 0, 1, 1]], dtype=float))
    assert (p, lo, hi) == pytest.approx((0.75, *stats.wilson_interval(3, 4)))
    p, lo, hi = reducer.mean_rate(np.ones((3, 8)))
    assert p == 1.0 and hi == 1.0 and lo == pytest.approx(stats.wilson_interval(8, 8)[0]) and lo < 0.7
    p, lo, hi = reducer.mean_rate(np.array([[1, 0], [0, 0], [1, 1]], dtype=float))  # query means 2/3 and 1/3
    assert (p, lo, hi) == pytest.approx((0.5, *stats.wilson_interval(1.0, 2)))


def _cell(reps, system="jigsaw"):
    return reducer.Cell("E1", DS, "R0", "primary", system, 120.0, False, reps=reps, runs={"r"})


def _answered(uids, solved):
    return {u: {"uid": u, "T": 120.0, **(outcome("F1", 120.0) if u in solved else outcome("T", 120.0))}
            for u in uids}


def test_writer_rates_of_seeded_and_fixed_cells(tmp_path):
    nums = numbers_json.Numbers(split="test")
    nums.source("r", path=tmp_path, manifest_sha256=None, experiment="E1", dataset=DS)
    w = reducer.Writer(nums)
    uids = [f"u{i}" for i in range(8)]
    solved = reducer.solved_at(30.0)
    # every seed solves every query: the Wilson interval of 8 of 8, never the degenerate [100, 100]
    w.rate("agree", _cell({s: _answered(uids, uids) for s in SEEDS}), uids, solved)
    agree = nums.numbers["agree"]
    assert agree["value"] == 100.0 and agree["n"] == 8
    assert agree["ci"] == pytest.approx([100.0 * stats.wilson_interval(8, 8)[0], 100.0])
    assert agree["ci"][0] < 100.0
    assert "3 training seeds" in agree["note"]
    for suffix in (".s7301", ".s7302", ".s7303", ".seed_min", ".seed_max"):
        assert nums.numbers[f"agree{suffix}"]["value"] == 100.0
    # seeds disagree on two queries: 8, 6 and 7 of 8 solved; the query means sum to 7
    w.rate("mixed", _cell({7301: _answered(uids, uids), 7302: _answered(uids, uids[:6]),
                           7303: _answered(uids, uids[:7])}), uids, solved)
    assert_rate(nums.numbers["mixed"], 7.0, 8)
    assert_rate(nums.numbers["mixed.s7302"], 6, 8)
    assert nums.numbers["mixed.seed_min"]["value"] == 75.0 and nums.numbers["mixed.seed_max"]["value"] == 100.0
    # a fixed method keeps the ordinary Wilson rate and has no seed keys
    w.rate("fixed", _cell({None: _answered(uids, uids[:5])}, "filterall"), uids, solved)
    assert_rate(nums.numbers["fixed"], 5, 8)
    assert not any(k.startswith("fixed.") for k in nums.numbers)


# ----------------------------------------------------------------------------
# E1 (with PE1, PE2, E3, E14 and E18)
# ----------------------------------------------------------------------------
BASE = {"p0": "F0.5", "p1": "F3", "p2": "F40", "p3": "T", "p4": "F0.5", "p5": "F0.8",
        "n0": "N0.2", "n1": "T", "n2": "T", "n3": "T", "n4": "N50"}
SEED_CODES = {7301: BASE, 7302: {**BASE, "p1": "T", "p3": "F100"}, 7303: {**BASE, "p2": "T"}}
FILTERALL = {"p0": "F1", "p1": "F2", "p2": "F10", "p3": "F20", "p4": "T", "p5": "F1.5",
             "n0": "N0.5", "n1": "N5", "n2": "F3", "n3": "T", "n4": "T"}
INDEX = {**{u: "F1" for u in POS}, "n0": "N1", "n1": "N2", "n2": "T", "n3": "F60", "n4": "T"}
E1_CERT = {"n0": cert_entry("none", "stage1", {"stage1": 3.0}),
           "n1": cert_entry("none", "stage2", {"stage1": 5.0, "stage2": 20.0}),
           "n2": cert_entry("found", "stage2", {"stage1": 1.0, "stage2": 30.0}),
           "n3": cert_entry("undecided", None, {"stage1": 2.0, "stage2": 10.0}, errors={"stage2": "boom"}),
           "n4": cert_entry("undecided", None, {"stage1": 4.0}, rejected=[{"stage": "stage1", "error": "bad"}])}


def e1_world(tmp, *, split="test", seeds=SEEDS, fixed=None, cert=True, entries=None, cert_kw=None):
    """E1 as two runs (the three seeds of jigsaw; filterall and jigsaw_index) and a certification."""
    qfile = query_file(tmp)
    learned = write_run(tmp / "runs" / "learned", "E1",
                        {f"jigsaw@{s}": records(f"jigsaw@{s}", SEED_CODES[s], T=120) for s in seeds},
                        T=120, qfile=qfile, split=split)
    fixed = dict(fixed or {})
    T = fixed.pop("T", 120.0)
    fq = query_file(tmp, version=fixed.pop("qversion")) if "qversion" in fixed else qfile
    fixed_run = write_run(tmp / "runs" / "fixed", "E1",
                          {"filterall": records("filterall", FILTERALL, T=T),
                           "jigsaw_index": records("jigsaw_index", INDEX, T=T)},
                          T=T, qfile=fq, split=fixed.pop("split", split), **fixed)
    certs = {}
    if cert:
        kw = dict(cert_kw or {})
        cq = query_file(tmp, version=kw.pop("qversion")) if "qversion" in kw else qfile
        path = write_cert(tmp / "certs" / "R0.json", "R0", E1_CERT if entries is None else entries, cq,
                          split=kw.pop("split", split), **kw)
        certs = {DS: {"R0": str(path)}}
    return SimpleNamespace(qfile=qfile, learned=learned, fixed=fixed_run,
                           plan={"runs": [entry("E1", learned, fixed_run)], "certifications": certs})


def test_e1_numbers_from_one_reduction(tmp_path):
    w = e1_world(tmp_path)
    code, out, doc = run_plan(tmp_path, w.plan)
    assert code == 0 and doc["split"] == "test"
    nums = doc["numbers"]
    J = "E1.synth.R0.jigsaw"
    # seed-averaged: per-query means over the 3 seeds, Wilson of their sum over the 6 positives
    assert_rate(nums[f"{J}.solved.t1"], 3, 6)
    assert_rate(nums[f"{J}.solved.t5"], 11 / 3, 6)
    assert_rate(nums[f"{J}.solved.t30"], 11 / 3, 6)
    assert_rate(nums[f"{J}.solved.t120"], 14 / 3, 6)
    for seed, k in ((7301, 4), (7302, 3), (7303, 4)):
        assert_rate(nums[f"{J}.solved.t30.s{seed}"], k, 6)
    assert nums[f"{J}.solved.t30.seed_min"]["value"] == pytest.approx(50.0)
    assert nums[f"{J}.solved.t30.seed_max"]["value"] == pytest.approx(400 / 6)
    assert_rate(nums[f"{J}.single.solved.t30"], 5 / 3, 2)
    one = nums[f"{J}.single.n8.solved.t30"]
    assert one["value"] == 100.0 and one["ci"][0] < 100.0 and one["ci"][1] == pytest.approx(100.0)
    assert_rate(nums[f"{J}.correct_none.t30"], 1, 2)
    assert val(doc, f"{J}.neg_absent_label.correct_none.t30") == 100.0
    assert val(doc, f"{J}.neg_structure.correct_none.t30") == 0.0
    # fixed methods: the ordinary Wilson rate
    assert_rate(nums["E1.synth.R0.filterall.solved.t30"], 5, 6)
    assert_rate(nums["E1.synth.R0.filterall.correct_none.t30"], 2, 2)
    assert "E1.synth.R0.filterall.solved.t30.seed_min" not in nums
    # counts, summed over seeds
    assert val(doc, f"{J}.timeout_pos.t30") == 7 and val(doc, "E1.synth.R0.filterall.timeout_pos.t30") == 1
    assert val(doc, f"{J}.timeout_neg.t30") == 3
    assert val(doc, f"{J}.newly_certified") == 3 and val(doc, f"{J}.newly_found") == 0
    assert val(doc, "E1.synth.R0.jigsaw_index.newly_found") == 1
    assert val(doc, f"{J}.fp") == 0 and val(doc, f"{J}.crash") == 0
    assert val(doc, "E1.synth.R0.relabelled") == 1 and val(doc, "E1.synth.R0.undecided") == 2
    assert "E1.synth.R0.uncertified" not in nums
    # PE2: paired over queries on the seed means
    a = {"p0": 100.0, "p1": 100.0 * float(np.mean([1.0, 0.0, 1.0])), "p2": 0.0, "p3": 0.0, "p4": 100.0, "p5": 100.0}
    b = {"p0": 100.0, "p1": 100.0, "p2": 100.0, "p3": 100.0, "p4": 0.0, "p5": 100.0}
    ref = stats.paired_compare(a, b, n_boot=LOW_DRAWS["boot"], n_perm=LOW_DRAWS["perm"], seed=0)
    pe2 = "PE2.synth.R0.jigsaw_vs_filterall"
    assert nums[f"{pe2}.diff"]["value"] == pytest.approx(-200 / 9)
    assert nums[f"{pe2}.diff"]["ci"] == pytest.approx([ref.lo, ref.hi])
    assert val(doc, f"{pe2}.p") == pytest.approx(ref.p_value) and val(doc, f"{pe2}.p_holm") == val(doc, f"{pe2}.p")
    assert (val(doc, f"{pe2}.wins"), val(doc, f"{pe2}.losses"), val(doc, f"{pe2}.ties")) == (1, 3, 2)
    assert val(doc, f"{pe2}.noninferior") == "no" and val(doc, f"{pe2}.superior") == "no"
    assert val(doc, f"{pe2}.diff_hier") == pytest.approx(-200 / 9)
    assert val(doc, "PE2.synth.R0.c2.pairs") == 7 and val(doc, "PE2.synth.R0.c2.retrieval") == 7
    assert val(doc, "PE2.margin") == 1.0
    assert "PE2.synth.R0.jigsaw_index_vs_filterall.diff_hier" not in nums  # two fixed methods
    assert "PE2.synth.R0.jigsaw_vs_jigsaw_index.diff_hier" in nums
    # PE1
    assert val(doc, "PE1.holds") == "yes" and val(doc, "PE1.found") == 27 and val(doc, "PE1.witness_pass") == 100.0
    # E14 from the certification
    E14 = "E14.synth.R0"
    assert {k: val(doc, f"{E14}.{k}") for k in ("generated", "none", "found", "undecided", "rejected_witnesses",
                                                 "errors")} == {"generated": 5, "none": 2, "found": 1,
                                                                "undecided": 2, "rejected_witnesses": 1,
                                                                "errors": 1}
    assert val(doc, f"{E14}.stage1.seconds_median") == 3.0 and val(doc, f"{E14}.stage1.seconds_max") == 5.0
    assert val(doc, f"{E14}.stage2.seconds_median") == 20.0 and val(doc, f"{E14}.stage2.seconds_max") == 30.0
    assert val(doc, f"{E14}.neg_absent_label.n8.none.stage1") == 1
    assert val(doc, f"{E14}.neg_swapped_label.n8.found.stage2") == 1
    assert doc["sources"]["cert.synth.R0"]["experiment"] == "E14"
    # E18: a step-less undecided positive is a retrieval failure; seeded shares are per-query means
    assert_rate(nums["E18.E1.synth.R0.jigsaw.pos.retrieval.t30"], 7 / 3, 6)
    assert_rate(nums["E18.E1.synth.R0.filterall.pos.retrieval.t30"], 1, 6)
    # tables, figures and side files
    table = read_csv(out / "tables" / "E1.synth.R0.csv")
    assert table[0] == ["system", "solved@1s", "solved@5s", "solved@30s", "solved@120s", "correct NONE@30s", "FP",
                        "FN", "crash"]
    assert [row[0] for row in table[1:]] == ["jigsaw", "jigsaw_index", "filterall"]
    rows = list(csv.DictReader((out / "figures" / "E3.synth.R0.csv").read_text(encoding="utf-8").splitlines()))
    row = next(r for r in rows if r["series"] == "jigsaw" and float(r["x"]) == 30.0)
    lo, hi = stats.wilson_interval(11 / 3, 6)
    assert float(row["y"]) == pytest.approx(100 * 11 / 18)
    assert (float(row["lo"]), float(row["hi"])) == pytest.approx((100 * lo, 100 * hi))
    excl = json.loads((out / "exclusions.json").read_text(encoding="utf-8"))
    assert {(e["reason"], e["uid"]) for e in excl["exclusions"]} == {("relabelled", "n2"), ("undecided", "n3"),
                                                                     ("undecided", "n4")}
    assert excl["provisional"] == []
    loc = json.loads((out / "failure_localization.json").read_text(encoding="utf-8"))
    assert loc["E1"][DS]["R0"]["-"]["jigsaw@7301"]["p3"]["t30"] == "retrieval"
    assert json.loads((out / "pe1.json").read_text(encoding="utf-8"))["holds"] is True
    assert manifest.verify_manifest(out) == []
    run = json.loads((out / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert run["experiment"] == "reduce" and run["split"] == "test" and run["config"]["draws"] == LOW_DRAWS


def _duplicate_p0(w):
    edit_records(w.learned, "jigsaw@7301", lambda recs: recs + [recs[0]])
    set_selected(w.learned, 12)


def _drop_n4(w):
    for arm in ("filterall", "jigsaw_index"):
        edit_records(w.fixed, arm, lambda recs: [r for r in recs if r["uid"] != "n4"])
    set_selected(w.fixed, 10)


def _stray_output(w):
    (w.learned / "stray.txt").write_text("not a record", encoding="utf-8")


def _delete_arm_file(w):
    (w.fixed / "records" / "R0" / "jigsaw_index.jsonl").unlink()
    manifest.finalize_manifest(w.fixed)


def _unconfigured_arm_file(w):
    rdir = w.fixed / "records" / "R0"
    shutil.copy(rdir / "filterall.jsonl", rdir / "random.jsonl")
    manifest.finalize_manifest(w.fixed)


F = reducer.Flags
E1_REFUSALS = [
    # (e1_world arguments, a change to the world, the refusal, the flag that lets it through)
    pytest.param({"seeds": SEEDS[:2]}, None, "training seeds [7301, 7302], the protocol's are", F(allow_dev=True),
                 id="seed_set"),
    pytest.param({}, _duplicate_p0, "p0 recorded twice", None, id="duplicate_record"),
    pytest.param({"split": "dev"}, None, "split 'dev' is not a test split", F(allow_dev=True), id="dev_split"),
    pytest.param({"fixed": {"T": 60.0}}, None, "E1 on the test split runs at T = 120 s, not 60", None, id="T"),
    pytest.param({"fixed": {"split": "reserve_a"}}, None, "the runs come from several splits", None,
                 id="another_split"),
    pytest.param({}, lambda w: set_manifest(w.learned, status="running"), "manifest status 'running', not 'complete'",
                 None, id="incomplete_manifest"),
    pytest.param({}, _stray_output, "differs from its manifest: unrecorded output stray.txt", F(no_verify=True),
                 id="changed_outputs"),
    pytest.param({}, lambda w: set_manifest(w.learned, experiment="arms"), "ad-hoc runs need --allow-dev",
                 F(allow_dev=True), id="ad_hoc_run"),
    pytest.param({"fixed": {"partition": "kahip_fast"}}, None, "only E16 runs on a sensitivity partition", None,
                 id="partition"),
    pytest.param({}, lambda w: set_manifest(w.learned, protocol_hash="0" * 64), "run under protocol 000000000000...",
                 None, id="another_protocol"),
    pytest.param({}, lambda w: set_records(w.fixed, "filterall", train_seed=7301),
                 "a fixed method with training seeds [7301]", None, id="seeded_fixed_method"),
    pytest.param({}, lambda w: set_records(w.learned, "jigsaw@7301", train_seed=None),
                 "seeded and unseeded arms in one system", None, id="unseeded_replicate"),
    pytest.param({}, _drop_n4, "filterall lacks 1 of the 11 queries", F(allow_missing=True), id="missing_query"),
    pytest.param({"fixed": {"qversion": 2}}, None, "must use one query set", None, id="query_set"),
    pytest.param({"cert_kw": {"qversion": 2}}, None, "certifies query set", None, id="cert_query_set"),
    pytest.param({"entries": {**E1_CERT, "n2": cert_entry("found", "stage2", witness_check=None)},
                  "cert_kw": {"validate": False}}, None, "n2 is certified found without a checked witness", None,
                 id="cert_unchecked_witness"),
    pytest.param({"cert_kw": {"complete": False}}, None, "certification is partial", None, id="cert_partial"),
    pytest.param({"entries": {k: v for k, v in E1_CERT.items() if k != "n1"}}, None, "does not certify negative n1",
                 None, id="cert_missing_negative"),
    pytest.param({"entries": {**E1_CERT, "p0": cert_entry("none", "stage1")}}, None, "certifies p0, a positive", None,
                 id="cert_positive"),
    pytest.param({"cert_kw": {"protocol_hash": "0" * 64}}, None, "made under another protocol", None,
                 id="cert_protocol"),
    pytest.param({"cert": False}, None, "pass --allow-uncertified", F(allow_uncertified=True), id="uncertified"),
    pytest.param({"cert_kw": {"split": "reserve_a"}}, None, "split 'reserve_a', the runs' is 'test'", None,
                 id="cert_split"),
    pytest.param({}, lambda w: set_records(w.fixed, "filterall", "p0", family="weird"),
                 "unknown query family 'weird'", None, id="record_family"),
    pytest.param({}, _delete_arm_file, "missing ['jigsaw_index.jsonl']", None, id="missing_arm_file"),
    pytest.param({}, _unconfigured_arm_file, "files of arms the run did not configure: ['random.jsonl']", None,
                 id="unconfigured_arm_file"),
    pytest.param({}, lambda w: set_records(w.fixed, "filterall", T=60.0), "T = 60, the run's is 120", None,
                 id="record_T"),
    pytest.param({}, lambda w: set_records(w.fixed, "filterall", "p0", time_found=None),
                 "a verified FOUND without time_found", None, id="record_time_found"),
    pytest.param({}, lambda w: set_records(w.fixed, "filterall", "p0", size=16),
                 "kind, family or size differ between runs", None, id="record_size"),
]


@pytest.mark.parametrize("world_kw, change, match, passing", E1_REFUSALS)
def test_e1_refusals(tmp_path, world_kw, change, match, passing):
    w = e1_world(tmp_path, **world_kw)
    if change is not None:
        change(w)
    refused(tmp_path, w.plan, match)
    if passing is not None:
        assert run_plan(tmp_path, w.plan, passing, name="passed")[0] == 0


def test_plan_and_output_refusals(tmp_path):
    w = e1_world(tmp_path)
    refused(tmp_path, {"runs": [entry("E1", w.learned, w.learned)]}, "is listed twice")
    refused(tmp_path, {**w.plan, "bogus": 1}, "unknown keys ['bogus']")
    refused(tmp_path, {"runs": [{"experiment": "E1", "dataset": DS, "dirs": ["nothing*"]}]}, "nothing* matches nothing")
    refused(tmp_path, {"runs": [entry("E5", w.learned)]}, "unknown experiment 'E5'")
    refused(tmp_path, {}, "names no runs, certifications, measurement roots, offline stages or pattern audits")
    plan = write_plan(tmp_path, w.plan, "busy")
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(reducer.ReduceError, match="exists and is not an empty directory"):
        reducer.execute(plan, busy)
    with pytest.raises(reducer.ReduceError, match="lies inside the input directory"):
        reducer.execute(plan, w.learned / "out")
    assert not (w.learned / "out").exists()


def test_plan_globs_are_relative_to_the_plan_file(tmp_path):
    base = tmp_path / "we[ird]"  # a "[" of the plan's own directory is not a pattern
    for name in ("a", "b"):
        (base / "runs" / name).mkdir(parents=True)
    path = base / "plan.json"
    path.write_text(json.dumps({"runs": [{"experiment": "E1", "dataset": DS, "dirs": "runs/*"}]}), encoding="utf-8")
    plan = reducer.load_plan(path)
    assert plan.entries == [("E1", DS, [(base / "runs" / n).resolve() for n in ("a", "b")])]


# ----------------------------------------------------------------------------
# E8, PE1, E9 and E3x
# ----------------------------------------------------------------------------
E8_CODES = {"direct": {"p0": "F10", "p1": "T", "p2": "T", "p3": "T", "p4": "F1", "p5": "F20"},
            "filterall": {"p0": "F5", "p1": "F100", "p2": "F200", "p3": "F300", "p4": "F500", "p5": "F25"}}


def e8_world(tmp, shards=((0, 2), (1, 2)), *, unsharded=False):
    qfile = query_file(tmp)
    halves = (("p0", "p1", "p2"), ("p3", "p4", "p5"))
    dirs = [write_run(tmp / "runs" / f"shard{i}of{n}", "E8",
                      {arm: records(arm, {u: codes[u] for u in halves[i]}, T=600) for arm, codes in E8_CODES.items()},
                      T=600, qfile=qfile, shard=(i, n)) for i, n in shards]
    if unsharded:
        dirs.append(write_run(tmp / "runs" / "whole", "E8", {"direct": [], "filterall": []}, T=600, qfile=qfile))
    return {"runs": [entry("E8", *dirs)]}


def test_e8_shards_and_exact_mcnemar(tmp_path):
    (tmp_path / "out").mkdir()  # an empty output directory is taken
    code, _, doc = run_plan(tmp_path, e8_world(tmp_path))
    assert code == 0
    nums = doc["numbers"]
    for arm, t30, t600 in (("direct", 3, 3), ("filterall", 2, 6)):
        assert_rate(nums[f"E8.synth.R0.{arm}.solved.t30"], t30, 6)
        assert_rate(nums[f"E8.synth.R0.{arm}.solved.t600"], t600, 6)
    pair = "E8.synth.R0.direct_vs_filterall"
    assert (val(doc, f"{pair}.t30.b"), val(doc, f"{pair}.t30.c")) == (1, 0)
    assert val(doc, f"{pair}.t30.p") == pytest.approx(1.0) and val(doc, f"{pair}.t30.p_holm") == pytest.approx(1.0)
    assert (val(doc, f"{pair}.t600.b"), val(doc, f"{pair}.t600.c")) == (0, 3)
    assert val(doc, f"{pair}.t600.p") == pytest.approx(0.25) and val(doc, f"{pair}.t600.p_holm") == pytest.approx(0.5)
    assert val(doc, "E8.synth.R0.direct.tts_median") == 10.0 and val(doc, "E8.synth.R0.filterall.tts_median") == 150.0
    assert val(doc, "E8.synth.R0.direct.single.n8.solved.t30") == 100.0


@pytest.mark.parametrize("shards, unsharded, match", [
    (((0, 2),), False, "shards [0] of 2, missing [1]"),
    (((0, 2), (1, 2)), True, "an unsharded run beside 2 other(s)"),
    (((0, 2), (1, 3)), False, "shards of different counts [2, 3]"),
])
def test_e8_shard_refusals(tmp_path, shards, unsharded, match):
    refused(tmp_path, e8_world(tmp_path, shards, unsharded=unsharded), match)


def test_pe1_violations_exit_3(tmp_path):
    qfile = query_file(tmp_path)
    direct = {"p0": "X5", "p1": "Ewitness rejected: bad edge", "p2": "N5", "p3": "ERuntimeError: boom", "p4": "F1",
              "p5": "F1", "n0": "F3", "n1": "N1"}
    filterall = {**{u: "F1" for u in POS[:5]}, "p5": "N3", "n0": "N2", "n1": "N1"}
    run = write_run(tmp_path / "runs" / "e8", "E8", {"direct": records("direct", direct, T=600),
                                                     "filterall": records("filterall", filterall, T=600)},
                    T=600, qfile=qfile)
    cert = write_cert(tmp_path / "certs" / "R0.json", "R0", {"n0": cert_entry("none", "stage1"),
                                                             "n1": cert_entry("none", "stage1")}, qfile)
    plan = {"runs": [entry("E8", run)], "certifications": {DS: {"R0": str(cert)}}}
    code, out, doc = run_plan(tmp_path, plan)
    assert code == 3  # written, then refused
    assert val(doc, "PE1.holds") == "no"
    assert {m: val(doc, f"PE1.{m}") for m in ("fp", "fn", "contradiction", "crash")} == {
        "fp": 2, "fn": 3, "contradiction": 1, "crash": 1}
    assert val(doc, "PE1.E8.fp") == 2
    pe1 = json.loads((out / "pe1.json").read_text(encoding="utf-8"))
    assert {(v["arm"], v["uid"], v["kind"]) for v in pe1["violations"]} == {
        ("direct", "p0", "fp"), ("direct", "p1", "fp"), ("direct", "p2", "fn"), ("direct", "p3", "crash"),
        ("direct", "n0", "contradiction"), ("filterall", "p5", "fn"),
        ("filterall", "n0", "fn")}  # n0 has a verified witness: a NONE proof on it is false
    assert run_plan(tmp_path, plan, F(allow_pe1_failure=True), name="allowed")[0] == 0


def test_e9_uncertified_negatives_are_provisional_or_relabelled(tmp_path):
    qfile = query_file(tmp_path)
    arms = {FIRST: {"p0": "F1", "p1": "F2", "n0": "N1", "n1": "T"},
            f"{FIRST}+halo=0": {"p0": "F1", "p1": "T", "n0": "N1", "n1": "F5"}}
    run = write_run(tmp_path / "runs" / "e9", "E9", {a: records(a, c, T=30) for a, c in arms.items()}, T=30,
                    qfile=qfile)
    plan = {"runs": [entry("E9", run)]}
    refused(tmp_path, plan, "pass --allow-uncertified")
    code, out, doc = run_plan(tmp_path, plan, F(allow_uncertified=True))
    assert code == 0
    excl = json.loads((out / "exclusions.json").read_text(encoding="utf-8"))
    assert [(e["reason"], e["uid"]) for e in excl["exclusions"]] == [("relabelled", "n1")]
    assert "verified witness" in excl["exclusions"][0]["detail"]
    assert [e["uid"] for e in excl["provisional"]] == ["n0"]
    assert any("uncertified negatives have a verified witness" in n for n in excl["notes"])
    none = doc["numbers"]["E9.synth.R0.full.correct_none.t30"]
    assert none["n"] == 1 and "uncertified" in none["note"]
    assert "uncertified" in doc["tables"]["E9.synth.R0"]["notes"]


def test_e9_mcnemar_with_holm_across_ablations(tmp_path):
    qfile = query_file(tmp_path)
    neg = {"n0": "N1", "n1": "N1"}
    arms = {FIRST: {**{u: "F1" for u in POS}, **neg},
            f"{FIRST}+halo=0": {**{u: "F1" for u in POS[:2]}, **{u: "T" for u in POS[2:]}, "n0": "N1", "n1": "T"},
            f"{FIRST}+prune_passes=0": {**{u: "F1" for u in POS[:5]}, "p5": "T", **neg}}
    run = write_run(tmp_path / "runs" / "e9", "E9", {a: records(a, c, T=30) for a, c in arms.items()}, T=30,
                    qfile=qfile)
    cert = write_cert(tmp_path / "certs" / "R0.json", "R0", {"n0": cert_entry("none", "stage1"),
                                                             "n1": cert_entry("none", "stage1")}, qfile)
    code, _, doc = run_plan(tmp_path, {"runs": [entry("E9", run)], "certifications": {DS: {"R0": str(cert)}}})
    assert code == 0
    halo = "E9.synth.R0.halo0_vs_full.pos"
    assert (val(doc, f"{halo}.b"), val(doc, f"{halo}.c")) == (0, 4)
    assert val(doc, f"{halo}.p") == pytest.approx(0.125) and val(doc, f"{halo}.p_holm") == pytest.approx(0.25)
    assert val(doc, f"{halo}.diff") == pytest.approx(-200 / 3)
    prune = "E9.synth.R0.prune_passes0_vs_full.pos"
    assert (val(doc, f"{prune}.b"), val(doc, f"{prune}.c")) == (0, 1) and val(doc, f"{prune}.p") == pytest.approx(1.0)
    assert val(doc, "E9.synth.R0.halo0_vs_full.neg.c") == 1


def e3x_world(tmp, reruns):
    """E1 at 120 s (three seeds of jigsaw and filterall) and one E3x run per rerun arm at 600 s."""
    qfile = query_file(tmp)
    seeded = {"p0": "F1", "p1": "F2", **{u: "T" for u in POS[2:]}}
    arms = {f"jigsaw@{s}": records(f"jigsaw@{s}", seeded, T=120) for s in SEEDS}
    arms["filterall"] = records("filterall", {**{u: "F1" for u in POS[:5]}, "p5": "T"}, T=120)
    e1 = write_run(tmp / "runs" / "e1", "E1", arms, T=120, qfile=qfile)
    dirs = [write_run(tmp / "runs" / f"e3x{i}", "E3x", {arm: records(arm, codes, T=600)}, T=600, qfile=qfile)
            for i, (arm, codes) in enumerate(reruns)]
    return {"runs": [entry("E1", e1), entry("E3x", *dirs)]}


def test_e3x_extends_e1_on_its_undecided_queries(tmp_path):
    plan = e3x_world(tmp_path, [(FIRST, {"p2": "F300", "p3": "T"}), ("filterall", {"p5": "F400"})])
    code, out, doc = run_plan(tmp_path, plan)
    assert code == 0
    nums = doc["numbers"]
    J = "E3x.synth.R0.jigsaw-at-7301"
    for h in ("t5", "t30", "t120"):
        assert_rate(nums[f"{J}.solved.{h}"], 2, 6)
    assert_rate(nums[f"{J}.solved.t600"], 3, 6)
    assert (val(doc, f"{J}.rerun"), val(doc, f"{J}.not_rerun"), val(doc, f"{J}.newly_solved")) == (2, 2, 1)
    assert_rate(nums["E3x.synth.R0.filterall.solved.t600"], 6, 6)
    assert [val(doc, f"E3x.synth.R0.filterall.{k}") for k in ("rerun", "not_rerun", "newly_solved")] == [1, 0, 1]
    excl = json.loads((out / "exclusions.json").read_text(encoding="utf-8"))["exclusions"]
    assert sorted(e["uid"] for e in excl if e["reason"] == "e3x_not_rerun") == ["p4", "p5"]
    assert len(doc["tables"]["E3x.synth"]["rows"]) == 2 and "E3x.synth.R0" in doc["figures"]


def test_e3x_run_with_nothing_to_rerun_and_refusals(tmp_path):
    code, _, doc = run_plan(tmp_path / "a", e3x_world(tmp_path / "a", [(FIRST, {}), ("filterall", {"p5": "F400"})]))
    J = "E3x.synth.R0.jigsaw-at-7301"
    assert code == 0 and [val(doc, f"{J}.{k}") for k in ("rerun", "not_rerun", "newly_solved")] == [0, 4, 0]
    assert_rate(doc["numbers"][f"{J}.solved.t600"], 2, 6)
    refused(tmp_path / "b", e3x_world(tmp_path / "b", [(FIRST, {"p0": "F5"})]), "p0 was already decided in E1")
    refused(tmp_path / "c", e3x_world(tmp_path / "c", [("jigsaw_index", {"p2": "F5"})]),
            "arm jigsaw_index has no E1 records to extend")


# ----------------------------------------------------------------------------
# E7 with E7-classical and E12; E16
# ----------------------------------------------------------------------------
def e7_world(tmp):
    q = query_file(tmp, "R1")
    neg = {"n0": "N1", "n1": "N1"}
    jig = {"p0": "F1", "p1": "F2", "p2": "F3", "p3": "F4", "p4": "T", "p5": "T", **neg}
    ours = [rec(u, FIRST, c, T=120, regime="R1",
                **({"ranker_info": {"selectivity": float(u[1]), "route": "index"}} if u in POS else {}))
            for u, c in jig.items()]
    e7 = write_run(tmp / "runs" / "e7", "E7", {FIRST: ours}, T=120, qfile=q, regime="R1")
    classical = []
    for method, solved in (("classical_gql", POS[:1]), ("classical_cfl", POS[:2]), ("classical_dpiso", POS[:2])):
        codes = {**{u: "F1" if u in solved else "T" for u in POS}, **neg}
        classical.append(write_run(tmp / "runs" / method, "E7",
                                   {method: records(method, codes, T=120, regime="R1", external=True)},
                                   T=120, qfile=q, regime="R1"))
    gnnpe = {"p0": "F1", "p1": "U0.5", "p2": "U0.5", "n0": "U0.5", "p3": "T", "p4": "T", "p5": "T", "n1": "T"}
    shards = []
    for i, uids in enumerate((("p0", "p1", "p2", "n0"), ("p3", "p4", "p5", "n1"))):
        build = {"R1": {"offline_seconds": 12.5, "prepared_utc": UTC1, "reused": i > 0}}
        shards.append(write_run(tmp / "runs" / f"gnnpe{i}", "E12",
                                {"gnnpe": records("gnnpe", {u: gnnpe[u] for u in uids}, T=120, regime="R1",
                                                  external=True, none_certified=False)},
                                T=120, qfile=q, regime="R1", shard=(i, 2), extra={"prepare": build}))
    cert = write_cert(tmp / "certs" / "R1.json", "R1", {"n0": cert_entry("none", "stage1"),
                                                        "n1": cert_entry("none", "stage1")}, q)
    plan = {"runs": [entry("E7", e7), entry("E7-classical", *classical), entry("E12", *shards)],
            "certifications": {DS: {"R1": str(cert)}}}
    return SimpleNamespace(plan=plan, shards=shards)


def test_e7_classical_gnnpe_and_selectivity_bins(tmp_path):
    w = e7_world(tmp_path)
    code, _, doc = run_plan(tmp_path, w.plan)
    assert code == 0
    nums = doc["numbers"]
    assert_rate(nums["E7.synth.R1.jigsaw-at-7301.solved.t120"], 4, 6)
    best = "E7.synth.R1.jigsaw-at-7301_vs_best_classical"
    assert val(doc, f"{best}.best") == "classical_cfl"  # cfl and dpiso tie at 2; the name breaks the tie
    assert (val(doc, f"{best}.b"), val(doc, f"{best}.c")) == (2, 0)
    assert val(doc, f"{best}.p") == pytest.approx(0.5) and val(doc, f"{best}.p_holm") == pytest.approx(0.5)
    vs = "E7.synth.R1.jigsaw-at-7301_vs_gnnpe"
    assert (val(doc, f"{vs}.b"), val(doc, f"{vs}.c")) == (3, 0) and f"{vs}.p_holm" not in nums
    assert val(doc, "E7.synth.R1.gnnpe.false_none") == 2 and val(doc, "E7.synth.R1.gnnpe.none_unverified") == 3
    assert val(doc, "PE1.external.false_none") == 2 and val(doc, "PE1.holds") == "yes"
    assert [val(doc, f"E7.synth.selectivity.{q}") for q in ("q25", "q50", "q75")] == pytest.approx([1.25, 2.5, 3.75])
    assert val(doc, "E7.synth.jigsaw-at-7301.quartile.q1.solved.t120") == 100.0
    assert val(doc, "E7.synth.jigsaw-at-7301.quartile.q4.solved.t120") == 0.0
    assert val(doc, "E7.synth.jigsaw-at-7301.log10.b3.solved.t120") == pytest.approx(100 / 3)
    assert val(doc, "E17.synth.gnnpe.R1.offline_seconds") == 12.5 and val(doc, "E17.synth.gnnpe.R1.builds") == 1
    assert {"E7.synth.best_classical", "E7.synth"} <= set(doc["tables"])
    assert {"E7.synth.crossover_quartile", "E7.synth.crossover_log10"} <= set(doc["figures"])


def test_e12_gnnpe_builds(tmp_path):
    w = e7_world(tmp_path)
    path = w.shards[1] / manifest.MANIFEST_NAME
    edit_json(path, lambda d: d["extra"]["prepare"]["R1"].update(prepared_utc=UTC2, offline_seconds=20.0))
    code, _, doc = run_plan(tmp_path, w.plan, name="two_builds")
    assert code == 0
    assert val(doc, "E17.synth.gnnpe.R1.builds") == 2 and val(doc, "E17.synth.gnnpe.R1.offline_seconds") == 20.0
    edit_json(path, lambda d: d["extra"]["prepare"]["R1"].update(prepared_utc=UTC1))
    refused(tmp_path, w.plan, f"the GNN-PE build of {UTC1} took 20 s, another E12 run of synth/R1 records 12.5 s")
    edit_json(path, lambda d: d["extra"].pop("prepare"))
    refused(tmp_path, w.plan, "extra.prepare.R1 records no GNN-PE build")


def e16_world(tmp, *, kahip_qversion=1):
    qfile = query_file(tmp)
    neg = {"n0": "N1", "n1": "N1"}
    every = {**{u: "F1" for u in POS}, **neg}
    half = {**{u: "F1" for u in POS[:3]}, **{u: "T" for u in POS[3:]}, **neg}
    jig = {**{u: "F1" for u in POS[:5]}, "p5": "T", **neg}
    codes = {FIRST: jig, f"{FIRST}+halo=0": {**half, "n1": "T"}, f"{FIRST}+halo_hops=2": every,
             f"{FIRST}+seed_count=10": jig, f"{FIRST}+seed_count=40": jig,
             f"{FIRST}+policy=topk": {**{u: "F1" for u in POS[:4]}, "p4": "T", "p5": "T", **neg},
             "jigsaw_index": every, "filterall": {**{u: "F2" for u in POS}, "n0": "N2", "n1": "N2"}}
    primary = rx.experiment_plan("E16", DS)["partitions"]["primary"]
    assert set(codes) == {rx.Arm.parse(a).name for a in primary}
    runs = [write_run(tmp / "runs" / "primary", "E16", {a: records(a, c, T=30) for a, c in codes.items()}, T=30,
                      qfile=qfile),
            write_run(tmp / "runs" / "kahip_fast", "E16",
                      {FIRST: records(FIRST, every, T=30), "jigsaw_index": records("jigsaw_index", half, T=30),
                       "filterall": records("filterall", half, T=30)},
                      T=30, qfile=query_file(tmp, version=kahip_qversion), partition="kahip_fast"),
            write_run(tmp / "runs" / "metis_x0.5", "E16", {FIRST: records(FIRST, every, T=30)}, T=30, qfile=qfile,
                      partition="metis_x0.5", options={FIRST: {"budgets": [1, 2]}})]
    cert = write_cert(tmp / "certs" / "R0.json", "R0", {"n0": cert_entry("none", "stage1"),
                                                        "n1": cert_entry("none", "stage1")}, qfile)
    return {"runs": [entry("E16", *runs)], "certifications": {DS: {"R0": str(cert)}}}


def test_e16_sensitivity_and_partitions(tmp_path):
    code, _, doc = run_plan(tmp_path, e16_world(tmp_path))
    assert code == 0
    base = "E16.synth.R0"
    table = doc["tables"][f"{base}.sensitivity"]["rows"]
    assert [row[0] for row in table] == ["halo"] * 3 + ["seed_count"] * 3 + ["policy"] * 2 + ["reference"] * 2
    assert [row[2] for row in table] == ["halo0", "jigsaw", "halo2", "seeds10", "jigsaw", "seeds40", "jigsaw",
                                         "topk", "jigsaw_index", "filterall"]
    tested = ["filterall", "halo0", "halo2", "jigsaw_index", "seeds10", "seeds40", "topk"]
    holm = stats.holm({a: val(doc, f"{base}.primary.{a}_vs_jigsaw.p") for a in tested})
    assert {a: val(doc, f"{base}.primary.{a}_vs_jigsaw.p_holm") for a in tested} == pytest.approx(holm)
    assert val(doc, f"{base}.primary.halo0.solved.t30") == 50.0
    assert val(doc, f"{base}.primary.halo0_vs_jigsaw.diff") == pytest.approx(-100 / 3)
    parts = doc["tables"][f"{base}.partitions"]["rows"]
    assert [row[0] for row in parts] == ["primary", "kahip_fast", "metis_x0.5"]
    assert val(doc, f"{base}.kahip_fast.jigsaw_vs_filterall.diff") == pytest.approx(50.0)
    assert val(doc, f"{base}.metis_x0p5.jigsaw.solved.t30") == 100.0
    assert val(doc, f"{base}.metis_x0p5.budgets") == "1,2"


def test_e16_partitions_share_one_query_set(tmp_path):
    refused(tmp_path, e16_world(tmp_path, kahip_qversion=2), "must use one query set")


# ----------------------------------------------------------------------------
# Measured trees: E2 (E19, PE3, PE4) and E10
# ----------------------------------------------------------------------------
def write_cell(root, exp, system, mode, cache, cap, repeat, *, qfile, codes=None, peak=1000, io=None,
               runner="run_experiment", experiment_field=True, run_arm=None, overrides=None, measure=None,
               entries=None):
    """streaming.py's rep<k> directory: measure.json and, for a completed cell, the runner's run/."""
    cap_name = "none" if cap is None else f"{cap:g}"
    rep = root / DS / system.replace("/", "_") / f"{mode}-{cache}-cap{cap_name}" / f"rep{repeat}"
    rep.mkdir(parents=True)
    done = codes is not None
    doc = {"schema": reducer.MEASURE_SCHEMA, "dataset": DS, "system": system, "mode": mode, "cache": cache,
           "repeat": repeat, "protocol_hash": protocol.protocol_hash(), "cap_gb": cap, "runner": runner,
           "completed": done, "oom_killed": not done,
           "memprobe": {"exit_code": 0 if done else -9, "timed_out": False, "peak_rss_bytes": peak,
                        "oom_killed": not done}}
    if experiment_field:
        doc["experiment"] = exp
    doc.update(measure or {})
    (rep / "measure.json").write_text(json.dumps(doc), encoding="utf-8")
    if done:
        arm = run_arm or system
        recs = []
        for i, (uid, code) in enumerate(codes.items()):
            fields = {"mode": mode, "cache": cache, **(overrides or {}).get(uid, {})}
            if io is not None:
                fields["io"] = {"read_bytes": io[i], "children_block_input_bytes": 0}
            if entries is not None:
                fields["steps"] = [{"step": 0, "budget": 1, "final": True, "closed": True, "t_start": 0.0, "t_end": 0.01,
                                    "status": "NONE", "candidate": {"adjacency_entries_read": entries[i]}}]
            recs.append(rec(uid, arm, code, T=30, external=runner == "run_external", **fields))
        write_run(rep / "run", exp, {arm: recs}, T=30, qfile=qfile, mode=mode, cache=cache)
    return rep


FAST = {"p0": "F1", "p1": "F2", "p2": "F4"}
SLOW = {"p0": "F2", "p1": "F4", "p2": "F8"}


def e2_tree(root, qfile):
    cell = {"qfile": qfile}
    write_cell(root, "E2", FIRST, "streaming", "cold", None, 1, codes=FAST, peak=1000, io=[100, 400, 900],
               entries=[10, 40, 90], **cell)
    write_cell(root, "E2", FIRST, "streaming", "cold", None, 2, codes=FAST, peak=1100, io=[110, 390, 950], **cell)
    write_cell(root, "E2", "filterall", "streaming", "cold", None, 1, codes=SLOW, peak=3000, io=[1000, 2000, 4000],
               entries=[100, 200, 400], **cell)
    write_cell(root, "E2", "filterall", "streaming", "cold", None, 2, codes=SLOW, peak=3100, io=[1100, 2100, 3900],
               **cell)
    write_cell(root, "E2", FIRST, "streaming", "warm", None, 1, codes=FAST, peak=900, io=[0, 0, 0], **cell)
    write_cell(root, "E2", "filterall", "streaming", "warm", None, 1, codes=SLOW, peak=2900, io=[0, 0, 0], **cell)
    write_cell(root, "E2", FIRST, "streaming", "cold", 2.0, 1, peak=2_100_000_000, **cell)  # killed by the cap
    write_cell(root, "E2", "filterall", "resident", "cold", None, 1, codes=SLOW, peak=5000, experiment_field=False,
               **cell)
    write_cell(root, "E2", "jigsaw_index", "streaming", "cold", None, 1, codes=FAST, peak=1200, **cell)  # no io
    (root / DS / FIRST / "streaming-cold-capnone" / "rep1.attempt1").mkdir()  # a superseded attempt
    return root


def e10_tree(root, qfile):
    write_cell(root, "E10", FIRST, "streaming", "cold", None, 1, qfile=qfile,
               codes={"p0": "F8", "p1": "F16", "p2": "T"}, overrides={"p2": {"elapsed": 32.0}}, peak=1000)
    write_cell(root, "E10", FIRST, "streaming", "cold", 32.0, 1, qfile=qfile, peak=40e9)
    write_cell(root, "E10", "classical_gql", "resident", "cold", None, 1, qfile=qfile,
               codes={u: "F1" for u in POS[:3]}, peak=5000, runner="run_external")
    return root


def test_measured_e2_and_e10(tmp_path):
    qfile = query_file(tmp_path)
    plan = {"e2_root": str(e2_tree(tmp_path / "E2", qfile)), "e10_root": str(e10_tree(tmp_path / "E10", qfile))}
    code, out, doc = run_plan(tmp_path, plan)
    assert code == 0 and doc["split"] == "test"
    nums = doc["numbers"]
    # PE3: bytes read + 1 and latency, paired over queries, on repeat 1; the other repeats as a range
    pe3 = "PE3.synth.cold.jigsaw-at-7301_vs_filterall"
    ref = stats.paired_log_ratio({"p0": 101.0, "p1": 401.0, "p2": 901.0}, {"p0": 1001.0, "p1": 2001.0, "p2": 4001.0},
                                 n_boot=LOW_DRAWS["boot"], n_perm=LOW_DRAWS["perm"], seed=0)
    assert val(doc, f"{pe3}.bytes.gm_ratio") == pytest.approx(ref["gm_ratio"])
    assert nums[f"{pe3}.bytes.gm_ratio"]["ci"] == pytest.approx([ref["lo"], ref["hi"]])
    assert val(doc, f"{pe3}.bytes.repeats") == 2
    assert val(doc, f"{pe3}.latency.gm_ratio") == pytest.approx(0.5)
    # the logical cost, adjacency entries + 1, is a second metric beside the physical bytes
    ref_entries = stats.paired_log_ratio({"p0": 11.0, "p1": 41.0, "p2": 91.0}, {"p0": 101.0, "p1": 201.0, "p2": 401.0},
                                         n_boot=LOW_DRAWS["boot"], n_perm=LOW_DRAWS["perm"], seed=0)
    assert val(doc, f"{pe3}.entries.gm_ratio") == pytest.approx(ref_entries["gm_ratio"])
    assert val(doc, f"{pe3}.entries.repeats") == 2  # repeat 2 has no candidate steps: both arms read 0, a ratio of 1
    assert val(doc, f"{pe3}.entries.gm_ratio_max") == pytest.approx(1.0)
    assert val(doc, "PE3.synth.warm.jigsaw-at-7301_vs_filterall.bytes.gm_ratio") == pytest.approx(1.0)
    assert "PE3.synth.cold.jigsaw_index_vs_filterall.bytes.gm_ratio" not in nums  # no io counters
    assert "PE3.synth.cold.jigsaw_index_vs_filterall.latency.gm_ratio" in nums
    excl = json.loads((out / "exclusions.json").read_text(encoding="utf-8"))["exclusions"]
    assert any(e["reason"] == "pe3_no_io_counters" for e in excl)
    # PE4: peaks over repeats, success under each cap, Jigsaw streaming over the resident systems
    pe4 = "PE4.synth.jigsaw-at-7301.streaming.cold"
    assert [val(doc, f"{pe4}.capnone.peak_rss.{s}") for s in ("median", "min", "max")] == [1050.0, 1000.0, 1100.0]
    assert val(doc, f"{pe4}.capnone.success") == 2 and val(doc, f"{pe4}.cap2.success") == 0
    ratio = "PE4.synth.cold.jigsaw-at-7301_vs_filterall"
    assert val(doc, f"{ratio}.conservative_ratio") == pytest.approx(0.22)
    assert val(doc, f"{ratio}.median_ratio") == pytest.approx(0.21)
    assert doc["tables"]["PE4.synth.caps"]["columns"] == ["system", "mode", "cache", "capnone", "cap2", "cap4",
                                                          "cap8"]
    assert val(doc, "E19.synth.jigsaw-at-7301.streaming_cold.elapsed.p50") == pytest.approx(2.0)
    assert any(k.startswith("E18.E2.synth.R0.jigsaw-at-7301.streaming_cold.") for k in nums)
    # E10: the same reduction under "E10."
    assert val(doc, "E10.PE4.synth.jigsaw-at-7301.streaming.cold.cap32.success") == 0
    assert val(doc, "E10.PE4.synth.cold.jigsaw-at-7301_vs_classical_gql.conservative_ratio") == pytest.approx(0.2)
    assert val(doc, "E10.E19.synth.jigsaw-at-7301.streaming_cold.elapsed.p50") == pytest.approx(16.0)
    assert any(k.startswith("E18.E10.") for k in nums)
    assert doc["tables"]["E10.PE4.synth.caps"]["columns"] == ["system", "mode", "cache", "capnone", "cap32"]
    assert val(doc, "PE1.holds") == "yes"


def test_measured_refusals(tmp_path):
    qfile = query_file(tmp_path)

    def tree(name, *, valid=True, **kw):
        root = tmp_path / name
        root.mkdir()
        if valid or kw:
            write_cell(root, "E2", FIRST, "streaming", "cold", None, 1, qfile=qfile, codes=FAST, **kw)
        return root

    refused(tmp_path, {"e2_root": str(tree("empty", valid=False))}, "holds no measured cell")
    refused(tmp_path, {"e2_root": str(tree("repeat", measure={"repeat": 2}))}, "repeat 2, its place says 1")
    refused(tmp_path, {"e2_root": str(tree("arm", run_arm="filterall"))},
            "ran 'filterall', the cell's system is 'jigsaw@7301'")
    refused(tmp_path, {"e10_root": str(tree("e10"))}, "experiment 'E2', its place says 'E10'")
    root = tree("rep2")
    (root / DS / FIRST / "streaming-cold-capnone" / "rep2").mkdir()
    refused(tmp_path, {"e2_root": str(root)}, "no measure.json")
    root = tree("hot")
    (root / DS / FIRST / "streaming-hot-capnone").mkdir()
    refused(tmp_path, {"e2_root": str(root)}, "not a <mode>-<cache>-cap<N|none> cell")
    root = tree("repx")
    (root / DS / FIRST / "streaming-cold-capnone" / "repX").mkdir()
    refused(tmp_path, {"e2_root": str(root)}, "not a rep<k> directory")
    root = tree("dataset")
    (root / "nosuch").mkdir()
    refused(tmp_path, {"e2_root": str(root)}, "'nosuch' is not a dataset")
    root = tree("both")
    refused(tmp_path, {"e2_root": str(root), "e10_root": str(root)}, "e2_root and e10_root name one directory")


# ----------------------------------------------------------------------------
# E17: offline cost from the launch layout
# ----------------------------------------------------------------------------
STORE_FILES = {"meta.json": 5, "prepare_report.json": 7, "graph/indptr.npy": 1000, "graph/attrs.npy": 2000,
               "sigma/R0/codes.npy": 300, "part/primary/node_block.npy": 40, "part/metis_x0.5/node_block.npy": 41,
               "part/primary/rankers/R0/postings_ptr.npy": 999, "part/primary/embeddings/m/R0/block_emb.npy": 999}
GPU = "NVIDIA A100-SXM4-80GB, 535.54.03, 81920 MiB"


def offline_layout(tmp, load=None):
    """The directories reduce_plan finds: the store's prepare run, the offline builds, the encoders' training
    runs, the router fit, an E12 run and a certification. load: per encoder, each epoch's (cpu_seconds,
    host_busy_cpus), the machine load train_encoder records."""
    runs_root, store_root, queries_root = tmp / "RUNS", tmp / "STORE", tmp / "QUERIES"
    prepare = store_root / DS / "_prepare_run"
    manifest.write_manifest(prepare, experiment="prepare_data", config={"dataset": DS})
    (prepare / "stages").mkdir()
    for name, wall, peak in (("load", 10.0, 100), ("partition", 30.5, 400), ("sigma", 2.0, None)):
        (prepare / "stages" / f"{name}.json").write_text(
            json.dumps({"stage": name, "wall_seconds": wall, "peak_rss_bytes": peak}), encoding="utf-8")
    manifest.finalize_manifest(prepare, extra={"wall_seconds": 0.4,  # a resumed call: its stages were skipped
                                               "store_files": {k: {"bytes": v} for k, v in STORE_FILES.items()}})
    label_index = runs_root / "offline" / DS / "label_index"
    manifest.write_manifest(label_index, experiment="offline_build", config={"stage": "label_index", "dataset": DS})
    (label_index / "artifacts.json").write_text("{}", encoding="utf-8")
    manifest.finalize_manifest(label_index, extra={"stage": "label_index", "artifacts": 1, "artifact_bytes": 1234,
                                                   "peak_memory": {"peak_wset_bytes": 777, "maxrss_bytes": 5},
                                                   "wall_seconds": 3.0})
    logs = (([100.0, 110.5], [9.5]), ([50.0, 30.0], [20.0]))  # 220 s and 100 s of kept epochs and validations
    for j, (model_id, (epochs, validations)) in enumerate(zip(rp.main_model_ids(DS), logs)):
        run = runs_root / "train" / model_id
        manifest.write_manifest(run, experiment="train_encoder", config={"model_id": model_id})
        lines = [{"epoch": i + 1, "seconds": s} for i, s in enumerate(epochs)]
        for line, (cpu, busy) in zip(lines, load[j] if load else ()):
            line.update(cpu_seconds=cpu, host_busy_cpus=busy)
        write_jsonl(run / "train_log.jsonl", lines)
        write_jsonl(run / "val_metrics.jsonl", [{"epoch": len(epochs), "seconds": s} for s in validations])
        manifest.finalize_manifest(run, extra={"wall_seconds": 12.0})  # the last segment of a resumed run
        edit_json(run / manifest.MANIFEST_NAME, lambda d: d["machine"].update(gpus=[GPU]))
    router = runs_root / "router_fit" / DS
    manifest.write_manifest(router, experiment="router_fit", config={"dataset": DS})
    (router / "router_fit.json").write_bytes(b"x" * 10)
    (router / "router_fit.queries.jsonl").write_bytes(b"y" * 32)
    manifest.finalize_manifest(router, extra={"wall_seconds": 3.5})
    q1 = query_file(tmp, "R1")
    write_run(runs_root / "runs" / "E12" / DS / "run", "E12",
              {"gnnpe": records("gnnpe", {"p0": "F1", "p1": "U0.5"}, T=120, regime="R1", external=True,
                                none_certified=False)},
              T=120, qfile=q1, regime="R1",
              extra={"prepare": {"R1": {"offline_seconds": 12.5, "prepared_utc": UTC1, "reused": False}}})
    write_cert(queries_root / DS / "test" / "certification" / "R0.json", "R0", {"n0": cert_entry("none", "stage1")},
               query_file(tmp, "R0"))
    return SimpleNamespace(runs=runs_root, store=store_root, queries=queries_root, prepare=prepare,
                           label_index=label_index)


def test_e17_offline_cost_from_the_launch_layout(tmp_path):
    lay = offline_layout(tmp_path)
    plan, _ = rp.build_plan(runs_root=lay.runs, store_root=lay.store, queries_root=lay.queries, datasets=[DS])
    offline = plan["offline"][DS]
    assert set(offline) == {"prepare", "label_index", "training", "router_fit"} and len(offline["training"]) == 2
    assert [e["experiment"] for e in plan["runs"]] == ["E12"]
    code, _, doc = run_plan(tmp_path, plan)
    assert code == 0
    nums = doc["numbers"]

    def e17(name):
        return val(doc, f"E17.synth.{name}")

    assert [e17(f"prepare.{k}") for k in ("wall_seconds", "bytes", "peak_bytes", "runs", "machine")] == [
        42.5, 3393, 400, 1, "test-cpu"]
    assert "E17.synth.prepare.wall_seconds_median" not in nums
    assert [e17(f"prepare.{s}.wall_seconds") for s in ("load", "partition", "sigma")] == [10.0, 30.5, 2.0]
    assert e17("prepare.load.peak_bytes") == 100 and e17("prepare.partition.peak_bytes") == 400
    assert "E17.synth.prepare.sigma.peak_bytes" not in nums
    disk = ("graph", "sigma", "meta", "partition.primary", "partition.metis_x0p5", "total")
    assert {k: e17(f"disk.{k}") for k in disk} == {"graph": 3000, "sigma": 300, "meta": 12, "partition.primary": 40,
                                                   "partition.metis_x0p5": 41, "total": 3393}
    assert e17("training.wall_seconds") == 320.0 and e17("training.wall_seconds_median") == 160.0
    assert e17("training.gpu_hours") == pytest.approx(320 / 3600)
    assert e17("training.gpu_hours_per_encoder") == pytest.approx(160 / 3600)
    assert e17("training.gpu") == "NVIDIA A100-SXM4-80GB" and e17("training.runs") == 2
    assert "E17.synth.training.bytes" not in nums and "E17.synth.training.peak_bytes" not in nums
    assert not [k for k in nums if "quiet" in k]  # logs without the machine's load
    assert [e17(f"label_index.{k}") for k in ("wall_seconds", "bytes", "peak_bytes")] == [3.0, 1234, 777]
    assert e17("router_fit.bytes") == 42 and e17("router_fit.wall_seconds") == 3.5
    assert e17("gnnpe.R1.offline_seconds") == 12.5 and e17("gnnpe.R1.builds") == 1
    table = doc["tables"]["E17"]
    assert table["columns"] == ["dataset", "stage", "wall seconds", "bytes", "peak bytes", "runs"]
    assert [row[1] for row in table["rows"]] == ["label_index", "prepare", "router_fit", "training", "gnnpe R1"]
    assert doc["sources"]["E17.synth.prepare.0"]["experiment"] == "E17"


def test_e17_quiet_training_hours(tmp_path, monkeypatch):
    """Quiet epochs (other work below one busy CPU) time the encoders as if nothing ran beside them."""
    # encoder 1: 100 s quiet (0.5 other CPUs), 110.5 s beside 5 other CPUs; encoder 2: both epochs quiet
    load = (((400.0, 4.5), (442.0, 9.0)), ((200.0, 4.2), (120.0, 4.0)))
    lay = offline_layout(tmp_path, load)
    plan, _ = rp.build_plan(runs_root=lay.runs, store_root=lay.store, queries_root=lay.queries, datasets=[DS])
    monkeypatch.setattr(offline_build, "QUIET_MIN_EPOCHS", 1)
    code, _, doc = run_plan(tmp_path, plan, name="quiet1")
    assert code == 0
    assert val(doc, "E17.synth.training.quiet_epoch_share") == 0.75
    assert doc["numbers"]["E17.synth.training.quiet_epoch_share"]["n"] == 4
    # encoder 1: 100 s x 2 epochs + 9.5 s; encoder 2: median(50, 30) x 2 + 20 s
    assert val(doc, "E17.synth.training.gpu_hours_quiet") == pytest.approx((209.5 + 100.0) / 3600)
    assert val(doc, "E17.synth.training.gpu_hours_per_encoder_quiet") == pytest.approx(154.75 / 3600)
    assert val(doc, "E17.synth.training.gpu_hours") == pytest.approx(320 / 3600)  # the plain sum is unchanged
    monkeypatch.setattr(offline_build, "QUIET_MIN_EPOCHS", 2)  # encoder 1 has one quiet epoch: no quiet total
    code, _, doc = run_plan(tmp_path, plan, name="quiet2")
    assert code == 0 and val(doc, "E17.synth.training.quiet_epoch_share") == 0.75
    assert not [k for k in doc["numbers"] if "gpu_hours" in k and "quiet" in k]


def test_e17_refusals(tmp_path):
    lay = offline_layout(tmp_path)
    for stage in ("disk", "gnnpe"):
        refused(tmp_path, {"offline": {DS: {stage: [str(lay.prepare)]}}},
                f"'{stage}' names E17 keys, not an offline stage")
    refused(tmp_path, {"offline": {DS: {"prepare": [str(lay.prepare), str(lay.label_index)]}}},
            "2 runs; a dataset's store is prepared by one run")
    bare = tmp_path / "bare"
    manifest.write_manifest(bare, experiment="prepare_data", config={})
    manifest.finalize_manifest(bare)
    refused(tmp_path, {"offline": {DS: {"prepare": [str(bare)]}}}, "offline stage prepare cannot be costed (ValueError")
    running = tmp_path / "running"
    manifest.write_manifest(running, experiment="offline_build", config={})
    refused(tmp_path, {"offline": {DS: {"label_index": [str(running)]}}}, "manifest status 'running', not 'complete'")
    refused(tmp_path, {"offline": {DS: {"prepare": [str(lay.prepare)]}}},
            "no run, certification or measured cell names a split")


# ----------------------------------------------------------------------------
# E20: the pattern-overlap audit
# ----------------------------------------------------------------------------
def encoder_ids(seeds=SEEDS, regime="R0"):
    return [reducer.main_model_id(DS, regime, s) for s in seeds]


def write_audit(path, qfile, seen, *, regime="R0", split="test", seeds=SEEDS, uids=None, complete=True,
                protocol_hash=None, smoke=False):
    """A pattern_audit.py output directory of the main encoders of seeds; seen {uid: {seed: (epoch, step, draw)}}."""
    encoders = encoder_ids(seeds, regime)
    uids = list(QUERIES) if uids is None else list(uids)
    kind = {u: QUERIES[u][0] for u in uids}
    by_id = {int(reducer._MODEL_SEED.search(m).group(1)): m for m in encoders}
    seen_doc = {u: {by_id.get(s, f"other-seed{s}-main"): {"first": list(first), "hits": 1}
                    for s, first in per.items()} for u, per in seen.items()}
    entry = {"query_set": {"path": str(qfile), "sha256": manifest.sha256_file(qfile)["sha256"]},
             "queries": len(uids), "positives": sum(k == "positive" for k in kind.values()),
             "negatives": sum(k == "negative" for k in kind.values()),
             "seen_positives": sum(kind.get(u) == "positive" for u in seen),
             "seen_negatives": sum(kind.get(u) == "negative" for u in seen),
             "uids": uids, "encoders": encoders, "seen": seen_doc}
    doc = {"schema": reducer.AUDIT_SCHEMA, "dataset": DS, "split": split, "master": "m", "query_protocol_hash": None,
           "protocol_hash": protocol_hash or protocol.protocol_hash(), "complete": complete,
           "regimes": {regime: entry},
           "encoders": {m: {"regime": protocol.TRAINED_REGIMES[regime], "serves": [regime], "seed": s, "smoke": smoke}
                        for s, m in by_id.items()}}
    path.mkdir(parents=True, exist_ok=True)
    (path / reducer.AUDIT_FILE).write_text(json.dumps(doc), encoding="utf-8")
    manifest.write_manifest(path, experiment="pattern_audit", config={}, inputs={f"queries/{regime}": qfile},
                            split=split)
    manifest.finalize_manifest(path)
    return path


SEEN = {"p0": {7301: (1, 0, 0)}, "p2": {7302: (2, 1, 3), 7303: (1, 1, 1)}, "n2": {7301: (3, 0, 2)}}


def test_e20_seen_and_novel_rates(tmp_path):
    w = e1_world(tmp_path)
    audit = write_audit(tmp_path / "audit", w.qfile, SEEN)
    code, out, doc = run_plan(tmp_path, {**w.plan, "pattern_audit": {DS: str(audit)}})
    assert code == 0
    nums = doc["numbers"]
    A = "E20.synth.R0"
    assert {k: val(doc, f"{A}.{k}") for k in ("queries", "positives", "negatives", "seen_positives",
                                               "seen_negatives", "encoders")} == {
        "queries": 11, "positives": 6, "negatives": 5, "seen_positives": 2, "seen_negatives": 1, "encoders": 3}
    assert_rate(nums[f"{A}.seen_share"], 2, 6)
    assert [val(doc, f"{A}.seen.s{s}") for s in SEEDS] == [2, 1, 1]
    E = "E20.E1.synth.R0"
    assert val(doc, f"{E}.positives.seen") == 2 and val(doc, f"{E}.positives.novel") == 4
    # seed means at 30 s: p0 1 and p2 0 are seen; p1 2/3, p3 0, p4 1 and p5 1 are novel (E1's 11/3 of 6 split)
    assert_rate(nums[f"{E}.jigsaw.seen.solved.t30"], 1, 2)
    assert_rate(nums[f"{E}.jigsaw.novel.solved.t30"], 8 / 3, 4)
    assert_rate(nums[f"{E}.jigsaw.novel.solved.t30.s7302"], 2, 4)
    assert_rate(nums[f"{E}.filterall.seen.solved.t30"], 2, 2)
    assert_rate(nums[f"{E}.filterall.novel.solved.t30"], 3, 4)
    assert "E20.synth.0" in nums[f"{E}.filterall.novel.solved.t30"]["source"]
    vs = f"{E}.jigsaw_vs_filterall.novel"
    assert val(doc, f"{vs}.diff") == pytest.approx((100 * 2 / 3 - 100 - 100 + 100 + 0) / 4)
    assert (val(doc, f"{vs}.wins"), val(doc, f"{vs}.losses"), val(doc, f"{vs}.ties")) == (1, 2, 1)
    assert f"{vs}.p_holm" not in nums and f"{E}.jigsaw_vs_jigsaw_index.novel.diff_hier" in nums
    table = read_csv(out / "tables" / f"{E}.csv")
    assert table[0] == ["system", "solved@30s, all positives", "seen", "novel"]
    assert [row[0] for row in table[1:]] == ["jigsaw", "jigsaw_index", "filterall"]
    assert read_csv(out / "tables" / "E20.audit.csv")[1][:5] == ["synth", "R0", "6", "2",
                                                                 table_cell(nums[f"{A}.seen_share"])]
    assert doc["sources"]["E20.synth.0"]["experiment"] == "E20"
    run = json.loads((out / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert {"pattern_audit/E20.synth.0", f"pattern_audit/E20.synth.0/{reducer.AUDIT_FILE}"} <= set(run["inputs"])


def test_an_audit_from_before_the_freeze_leaves_the_numbers_unfrozen(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "FROZEN_PROTOCOL_HASH", protocol.protocol_hash())
    w = e1_world(tmp_path)  # every manifest records protocol_frozen true
    audit = write_audit(tmp_path / "audit", w.qfile, SEEN)
    monkeypatch.setattr(protocol, "FROZEN_PROTOCOL_HASH", None)
    early = write_audit(tmp_path / "audit-early", w.qfile, SEEN)
    monkeypatch.setattr(protocol, "FROZEN_PROTOCOL_HASH", protocol.protocol_hash())
    for name, entry, frozen in (("frozen", audit, True), ("early", early, False)):
        code, _, doc = run_plan(tmp_path, {**w.plan, "pattern_audit": {DS: str(entry)}}, name=name)
        assert code == 0 and doc["frozen"] is frozen


def table_cell(number):
    return reducer._render(number, True)


def _second_audit(tmp, w):
    return {DS: [str(write_audit(tmp / "audit", w.qfile, SEEN)), str(write_audit(tmp / "audit2", w.qfile, {}))]}


E20_REFUSALS = [
    # (e1_world arguments, write_audit arguments or a plan entry maker, the refusal)
    pytest.param({}, {"qversion": 2}, "audits query set", id="query_set"),
    pytest.param({}, {"seeds": SEEDS[:2]}, "a test split needs the main encoders of every training seed",
                 id="test_encoders"),
    pytest.param({}, {"smoke": True}, "not smoke runs", id="smoke"),
    pytest.param({"split": "dev"}, {"split": "dev", "seeds": SEEDS[:2], "seen": {"p0": {7301: (1, 0, 0)}}},
                 "jigsaw@7303 served R0 with synth-R0-graphsage-seed7303-main, which", id="unaudited_encoder"),
    pytest.param({}, {"seen": {"n0": {7301: (1, 0, 0)}}}, "negative n0 is certified none, but training query "
                 "(epoch, step, draw) (1, 0, 0) of synth-R0-graphsage-seed7301-main", id="seen_none_negative"),
    pytest.param({}, {"uids": [u for u in QUERIES if u != "p5"]}, "1 queries of the E1 and E7 runs are not in the "
                 "audited set, e.g. p5", id="unaudited_query"),
    pytest.param({}, {"seen": {"p0": {7399: (1, 0, 0)}}}, "seen query p0 is not in the set or names another encoder",
                 id="seen_by_another_encoder"),
    pytest.param({}, {"complete": False}, "not a complete jigsaw-pattern-audit-v1 audit", id="incomplete"),
    pytest.param({}, {"protocol_hash": "0" * 64}, "under protocol 000000000000..., listed as synth", id="protocol"),
    pytest.param({}, {"split": "reserve_a"}, "split 'reserve_a', the runs' is 'test'", id="split"),
    pytest.param({}, _second_audit, "synth/R0 is also audited by", id="audited_twice"),
    pytest.param({}, lambda tmp, w: {"cora_x": str(write_audit(tmp / "audit", w.qfile, {}))}, "unknown dataset",
                 id="dataset"),
]


@pytest.mark.parametrize("world_kw, audit, match", E20_REFUSALS)
def test_e20_refusals(tmp_path, world_kw, audit, match):
    w = e1_world(tmp_path, **world_kw)
    if callable(audit):
        entry = audit(tmp_path, w)
    else:
        kw = dict(audit)
        q = query_file(tmp_path, version=kw.pop("qversion")) if "qversion" in kw else w.qfile
        entry = {DS: str(write_audit(tmp_path / "audit", q, kw.pop("seen", SEEN), **kw))}
    flags = F(allow_dev=True) if world_kw.get("split") == "dev" else None
    refused(tmp_path, {**w.plan, "pattern_audit": entry}, match, flags)


def test_e20_audit_manifest_and_uncertified_negatives(tmp_path):
    w = e1_world(tmp_path, cert=False)
    audit = write_audit(tmp_path / "audit", w.qfile, {"n0": {7301: (1, 0, 0)}})
    plan = {**w.plan, "pattern_audit": {DS: str(audit)}}
    code, out, doc = run_plan(tmp_path, plan, F(allow_uncertified=True))
    assert code == 0 and val(doc, "E20.synth.R0.seen_negatives") == 1
    notes = json.loads((out / "exclusions.json").read_text(encoding="utf-8"))["notes"]
    assert any("negative n0 (negative_uncertified) is isomorphic to a training query" in n for n in notes)
    set_manifest(audit, experiment="E1")
    refused(tmp_path, plan, "manifest experiment 'E1', not 'pattern_audit'", F(allow_uncertified=True))
    set_manifest(audit, experiment="pattern_audit")
    (audit / "stray.txt").write_text("x", encoding="utf-8")
    refused(tmp_path, plan, "differs from its manifest: unrecorded output stray.txt", F(allow_uncertified=True))


# ----------------------------------------------------------------------------
# Key grammar, arms and the names repeated from other modules
# ----------------------------------------------------------------------------
def test_key_grammar():
    assert reducer.escape("jigsaw") == "jigsaw"
    assert reducer.escape("jigsaw@7301+backend=gql") == "jigsaw-at-7301-plus-backend-eq-gql"
    assert reducer.escape("R2-128") == "R2-dash-128"
    assert reducer.escape("é") == "-xe9-"
    names = {rx.Arm.parse(a).name for exp in ("E1", "E3x", "E7", "E8", "E9", "E16")
             for a in rx.experiment_plan(exp, DS)["arms"]}
    names |= {*run_external.METHODS, "mean_rrf+model=synth-R0-graphsage-seed*-main", "a b/c:d,e.f", "é∑"}
    for name in sorted(names):
        segment = reducer.escape(name)
        assert numbers_json.KEY_RE.match(segment) and reducer.unescape(segment) == name
        assert ("-" in segment) != bool(re.fullmatch(r"[A-Za-z0-9_]+", name))  # never equal to a plain name
    with pytest.raises(reducer.ReduceError):
        reducer.escape("")
    for bad in ("a-b", "a-bogus-b", "a-at-b-", "a-at-b@"):
        with pytest.raises(ValueError):
            reducer.unescape(bad)
    assert (reducer.seg_horizon(30), reducer.seg_horizon(0.5), reducer.seg_horizon(120.0)) == ("t30", "t0p5", "t120")
    assert (reducer.seg_cap(None), reducer.seg_cap(2), reducer.seg_cap(0.5)) == ("capnone", "cap2", "cap0p5")
    assert (reducer.seg_size(8), reducer.seg_seed(7301)) == ("n8", "s7301")
    assert [reducer.seg_partition(p) for p in ("primary", "kahip_fast", "metis_x0.5", "metis_x2")] == [
        "primary", "kahip_fast", "metis_x0p5", "metis_x2"]
    for check, bad in ((reducer.seg_partition, "nowhere"), (reducer.seg_regime, "R9"), (reducer.seg_family, "weird"),
                       (reducer.seg_system, "jigsaw_vs_filterall")):
        with pytest.raises(reducer.ReduceError):
            check(bad)
    assert reducer.measured_key("E10", "PE4", "x") == "E10.PE4.x" and reducer.measured_key("E2", "PE4", "x") == "PE4.x"


def test_arms_and_systems():
    for text in ("jigsaw@7301+halo=0+component_cap=50", "jigsaw@7301+seed_count=40+halo_hops=2",
                 "filterall+backend=glasgow", "jigsaw_index"):
        assert reducer.arm_name(*reducer.parse_arm(text)) == rx.Arm.parse(text).name
    for bad in ("jigsaw@x", "filterall+halo", "filterall+halo=0+halo=1", "filterall+=1"):
        with pytest.raises(reducer.ReduceError):
            reducer.parse_arm(bad)
    assert reducer.strip_backend("filterall+backend=networkx", "networkx") == "filterall"
    assert reducer.strip_backend("filterall+backend=gql", "networkx") == "filterall+backend=gql"
    assert reducer.strip_backend("filterall+backend=gql", None) == "filterall+backend=gql"
    assert reducer.e1_system("jigsaw@7302+halo=0") == "jigsaw+halo=0"
    assert reducer.e1_system("mean_rrf+model=synth-R0-graphsage-seed7302-main") == \
        "mean_rrf+model=synth-R0-graphsage-seed*-main"
    assert reducer.replicate_of({"train_seed": 7302, "model_id": "x-seed7303-main"}) == 7302
    assert reducer.replicate_of({"train_seed": None, "model_id": "x-seed7303-main"}) == 7303
    assert reducer.replicate_of({}) is None
    assert reducer.system_order(["filterall", "classical_gql", "direct", "jigsaw_index", "jigsaw"]) == [
        "jigsaw", "jigsaw_index", "filterall", "direct", "classical_gql"]
    e9 = [rx.Arm.parse(a).name for a in rx.experiment_plan("E9", DS)["arms"]]
    assert [reducer.e9_alias(a)[0] for a in e9] == ["full", "halo0", "prune_passes0", "components_none",
                                                    "component_cap50", "label_filter0"]
    e16 = [rx.Arm.parse(a).name for a in rx.experiment_plan("E16", DS)["partitions"]["primary"]]
    assert [reducer.e16_alias(a)[0] for a in e16] == ["jigsaw", "halo0", "halo2", "seeds10", "seeds40", "topk",
                                                      "jigsaw_index", "filterall"]
    for alias, bad in ((reducer.e9_alias, f"{FIRST}+halo_hops=2"), (reducer.e16_alias, f"{FIRST}+halo=1"),
                       (reducer.e16_alias, "jigsaw"), (reducer.e16_alias, "jigsaw_index@7301")):
        with pytest.raises(reducer.ReduceError):
            alias(bad)


def test_names_repeated_from_modules_the_reducer_does_not_import():
    assert reducer.RECORD_SCHEMA == rx.SCHEMA == run_external.SCHEMA
    assert reducer.CERT_SCHEMA == certify_negatives.SCHEMA
    assert reducer.MEASURE_SCHEMA == streaming.SCHEMA and reducer.MEASURED == streaming.EXPERIMENTS
    assert reducer.LEARNED_METHODS == rx.LEARNED_METHODS and reducer.FIXED_METHODS == rx.FIXED_METHODS
    assert reducer.ABLATIONS == rx.ABLATIONS == tuple(reducer.E9_ALIASES)
    assert reducer.CLASSICAL == tuple(run_external.CLASSICAL)
    assert set(reducer.EXTERNAL_METHODS) == set(run_external.METHODS) == set(fl.EXTERNAL_METHODS)
    assert reducer.sensitivity_partitions() == rx.sensitivity_partitions()
    assert set(reducer.GROUP) == {experiment for experiment, _ in rp.RUN_ENTRIES}
    for exp in ("E1", "E3x", "E7", "E8", "E9", "E16", "E2"):
        assert reducer.planned_T(exp) == float(rx.experiment_plan(exp, DS)["T"])
    assert reducer.planned_T("E10") == float(rx.experiment_plan("E10", "papers100m")["T"])
    assert reducer.planned_T("E7-classical") == float(run_external.experiment_plan("E7", DS)["T"])
    assert reducer.planned_T("E12") == float(run_external.experiment_plan("E12", DS)["T"])
    assert reducer.planned_caps("E2") == streaming.plan_caps(streaming.experiment_plan("E2", DS))
    assert reducer.planned_caps("E10") == streaming.plan_caps(streaming.experiment_plan("E10", "papers100m"))
    for shard in (None, (1, 4)):
        for arm in ("jigsaw@7301+halo=0", "filterall", "mean_rrf+model=a/b"):
            assert reducer.record_file(rx.Arm.parse(arm).name, shard) == rx.record_name(rx.Arm.parse(arm), shard)
        assert reducer.record_file("classical_gql", shard) == run_external.record_name("classical_gql", shard)
    assert f"{reducer.PROTOCOL_DRAWS['perm'] // 1000}k" in protocol.STATISTICS["paired_test"]
    assert f"{reducer.PROTOCOL_DRAWS['boot'] // 1000}k" in protocol.STATISTICS["paired_interval"]
    assert reducer.PROTOCOL_DRAWS["quantile"] == inspect.signature(stats.quantile_with_ci).parameters["n_boot"].default
    assert reducer.SEED == protocol.STATISTICS["bootstrap_seed"]
    assert (reducer.AUDIT_SCHEMA, reducer.AUDIT_EXPERIMENT, reducer.AUDIT_FILE) == (pa.SCHEMA, pa.EXPERIMENT, pa.OUTPUT)
    for ds in protocol.DATASETS:
        for regime in protocol.regimes_for(ds):
            for seed in protocol.SEEDS["training"]:
                assert reducer.main_model_id(ds, regime, int(seed)) == serving.default_model_id(ds, regime, int(seed))


def test_main_resamples_with_the_protocol_draws_and_exits_2_on_a_refusal(monkeypatch, capsys):
    seen = {}

    def fake(plan, out, flags=None, argv=None):
        seen.update(draws=dict(reducer.DRAWS), flags=flags)
        return 0

    monkeypatch.setattr(reducer, "execute", fake)
    assert reducer.main(["--plan", "p.json", "--out", "o", "--allow-dev", "--no-verify"]) == 0
    assert seen == {"draws": reducer.PROTOCOL_DRAWS, "flags": F(allow_dev=True, no_verify=True)}
    assert json.loads(capsys.readouterr().out) == {"exit": 0, "out": "o"}

    def refuse(*args, **kwargs):
        raise reducer.ReduceError("nope")

    monkeypatch.setattr(reducer, "execute", refuse)
    assert reducer.main(["--plan", "p.json", "--out", "o"]) == 2
    assert "reducer refused: nope" in capsys.readouterr().err


def test_the_reducer_imports_neither_torch_nor_seal():
    banned = ["torch", "scripts.canonical.seal", "scripts.canonical.run_experiment",
              "scripts.canonical.certify_negatives", "scripts.canonical.streaming", "scripts.canonical.train_encoder",
              "scripts.canonical.run_external"]
    code = f"import sys; import scripts.canonical.reducer; print([m for m in {banned!r} if m in sys.modules])"
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
                          check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


# ----------------------------------------------------------------------------
# A real dev run of the runner
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dev_store(clustered_store, tmp_path_factory):
    root, _ = clustered_store
    copy = tmp_path_factory.mktemp("reducer_dev")
    shutil.copytree(root / DS, copy / DS)
    gstore = store_mod.GraphStore(copy, DS, mmap=True)
    rankers.build_artifacts(gstore, ["R0"])
    cascade.build_label_index(gstore, "R0")
    queries.SplitBuilder(gstore, DS, "dev", "20260607", sizes=[8], max_per_cell=1).build(
        ["R0"], copy / "queries" / "dev", master_tag="20260607")
    return copy


def test_a_dev_run_of_the_runner_reduces(dev_store, tmp_path, capsys):
    run = tmp_path / "run"
    assert rx.main(["--arms", "jigsaw_index", "filterall", "jigsaw_index+component_cap=1", "--store-root",
                    str(dev_store), "--dataset", DS, "--queries", str(dev_store / "queries" / "dev"), "--regimes",
                    "R0", "--T", "2.0", "--budgets", "1", "2", "--backend", "networkx", "--families",
                    *SMOKE_FAMILIES, "--out", str(run)]) == 0
    capsys.readouterr()
    plan = {"runs": [entry("E1", run)]}
    refused(tmp_path, plan, "ad-hoc runs need --allow-dev")
    code, _, doc = run_plan(tmp_path, plan, F(allow_dev=True, allow_uncertified=True))  # generated negatives
    assert code == 0 and doc["split"] == "dev"
    assert val(doc, "PE1.holds") == "yes"
    for system in ("filterall", "jigsaw_index", "jigsaw_index-plus-component_cap-eq-1"):  # the common backend dropped
        assert f"E1.synth.R0.{system}.solved.t1" in doc["numbers"]
