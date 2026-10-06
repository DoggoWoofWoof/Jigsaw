"""launch/reduce_plan: the reducer's plan from the canonical directory layout."""

from __future__ import annotations

import json

import pytest

from scripts.canonical import manifest, reducer
from scripts.canonical.launch import reduce_plan as rp


def _run(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / manifest.MANIFEST_NAME).write_text("{}", encoding="utf-8")
    return path


@pytest.fixture
def layout(tmp_path):
    runs_root, store_root, queries_root = tmp_path / "RUNS", tmp_path / "STORE", tmp_path / "Q"
    r = runs_root / "runs"
    for i in range(2):
        _run(r / "E1" / "arxiv" / f"shard{i}")
    for sub in ("E3x/arxiv", "E7/arxiv/jigsaw", "E7/arxiv/classical_gql", "E7/arxiv/classical_cfl", "E9/arxiv",
                "E16/arxiv/primary", "E16/arxiv/kahip_fast", "E12/arxiv/run", "E2/cell/rep1/run"):
        _run(r / sub)
    (r / "E12" / "arxiv" / "prepare").mkdir(parents=True)  # --prepare-only writes nothing there; not a run entry
    cert = queries_root / "arxiv" / "test" / "certification"
    cert.mkdir(parents=True)
    for regime in ("R0", "R1"):
        (cert / f"{regime}.json").write_text("{}", encoding="utf-8")
    _run(store_root / "arxiv" / "_prepare_run")
    _run(runs_root / "offline" / "arxiv" / "label_index")
    _run(runs_root / "offline" / "arxiv" / "rankers-primary")
    _run(runs_root / "offline" / "arxiv" / "rankers-kahip_fast")  # a sensitivity partition: not the system's cost
    ids = rp.main_model_ids("arxiv")
    for i in ids[:2]:
        _run(runs_root / "offline" / "arxiv" / f"embed-{i}-primary")
    for i in ids:
        _run(runs_root / "train" / i)
    _run(runs_root / "train" / "arxiv-R0-gat-seed7301-arch")  # an ablation row: not the system's cost
    _run(runs_root / "router_fit" / "arxiv")
    _run(runs_root / "audit" / "pattern" / "arxiv-test")
    return runs_root, store_root, queries_root


def test_the_plan_names_the_layout(layout, tmp_path):
    runs_root, store_root, queries_root = layout
    plan, absent = rp.build_plan(runs_root=runs_root, store_root=store_root, queries_root=queries_root,
                                 datasets=["arxiv"])
    r = (runs_root / "runs").resolve()
    got = {e["experiment"]: e["dirs"] for e in plan["runs"]}
    assert got == {
        "E1": [str(r / "E1/arxiv/shard0"), str(r / "E1/arxiv/shard1")],
        "E3x": [str(r / "E3x/arxiv")],
        "E7": [str(r / "E7/arxiv/jigsaw")],
        "E7-classical": [str(r / "E7/arxiv/classical_cfl"), str(r / "E7/arxiv/classical_gql")],
        "E9": [str(r / "E9/arxiv")],
        "E16": [str(r / "E16/arxiv/kahip_fast"), str(r / "E16/arxiv/primary")],
        "E12": [str(r / "E12/arxiv/run")],
    }
    assert all(e["dataset"] == "arxiv" for e in plan["runs"])
    assert set(plan["certifications"]["arxiv"]) == {"R0", "R1"}
    offline = plan["offline"]["arxiv"]
    ids = rp.main_model_ids("arxiv")
    assert offline["training"] == [str((runs_root / "train" / i).resolve()) for i in ids]
    assert len(offline["embeddings"]) == 2 and offline["rankers"] == [
        str((runs_root / "offline/arxiv/rankers-primary").resolve())]
    assert offline["prepare"] == [str((store_root / "arxiv/_prepare_run").resolve())]
    assert plan["pattern_audit"] == {"arxiv": [str((runs_root / "audit/pattern/arxiv-test").resolve())]}
    assert plan["e2_root"] == str(r / "E2") and "e10_root" not in plan
    assert "arxiv E8 (" in " ".join(absent) and any(a.startswith("E10 root") for a in absent)
    assert sum(a.startswith("arxiv certification") for a in absent) == 3  # R2-512, R2-128, R2-32
    assert sum(a.startswith("arxiv offline embeddings") for a in absent) == len(ids) - 2
    # the reducer takes the plan as written
    out = tmp_path / "plan.json"
    assert rp.main(["--runs-root", str(runs_root), "--store-root", str(store_root), "--queries-root",
                    str(queries_root), "--datasets", "arxiv", "--out", str(out)]) == 0
    loaded = reducer.load_plan(out)
    assert json.loads(out.read_text(encoding="utf-8")) == plan
    assert len(loaded.entries) == len(plan["runs"]) and loaded.measured["E2"] == r / "E2"
    assert loaded.audits == {"arxiv": [(runs_root / "audit/pattern/arxiv-test").resolve()]}


def test_refusals(layout, tmp_path):
    runs_root, store_root, queries_root = layout
    (runs_root / "runs" / "E8" / "arxiv").mkdir(parents=True)  # started nothing
    with pytest.raises(SystemExit, match="no manifest.json"):
        rp.build_plan(runs_root=runs_root, store_root=store_root, queries_root=queries_root, datasets=["arxiv"])
    with pytest.raises(SystemExit, match="unknown dataset"):
        rp.build_plan(runs_root=runs_root, store_root=store_root, queries_root=queries_root, datasets=["cora"])
    with pytest.raises(SystemExit, match="none of the layout"):
        rp.build_plan(runs_root=tmp_path / "empty", store_root=store_root, queries_root=queries_root,
                      datasets=["mag"])
