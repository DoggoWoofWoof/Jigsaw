"""E15: enumeration on the label-filtered graph and the recall of every ladder step."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts.canonical import backends, cascade, manifest, protocol, queries, rankers, seal, verify
from scripts.canonical import enumeration as en
from scripts.canonical import store as store_mod

BUDGETS = [1, 2, 4]


@pytest.fixture(scope="module")
def estore(clustered_store, tmp_path_factory):
    root, _ = clustered_store
    copy = tmp_path_factory.mktemp("enumeration")
    shutil.copytree(root / "synth", copy / "synth")
    gstore = store_mod.GraphStore(copy, "synth", mmap=True)
    rankers.build_artifacts(gstore, ["R0"])
    cascade.build_label_index(gstore, "R0")
    for split, master, regimes in (("dev", "20260607", ["R0"]), ("test", "sealed-for-tests", ["R0", "R1"])):
        queries.SplitBuilder(gstore, "synth", split, master, sizes=[6, 8], max_per_cell=2).build(
            regimes, copy / "queries" / split, master_tag=master)
    qs = queries.QuerySet.load(copy / "queries" / "dev" / "R0.npz")
    return gstore, copy, qs


def _cascade(gstore, method="jigsaw_index", **kw):
    bank = rankers.RankerBank(gstore, "R0") if method != "filterall" else None
    kw.setdefault("budgets", BUDGETS)
    return cascade.Cascade(gstore, "R0", method, backends.NetworkXBackend(), bank=bank, **kw)


def _all_embeddings(gstore, q):
    """Every embedding by VF2 on the label-filtered graph, independently of the module."""
    import networkx as nx
    from networkx.algorithms import isomorphism

    codes = gstore.sigma("R0").query_codes(q.payload, gstore)
    cand = gstore.candidate(range(gstore.num_blocks), codes, "R0")
    gt, gq = nx.Graph(), nx.Graph()
    for g, h in ((cand, gt), (q.local_graph(codes), gq)):
        for i, lab in enumerate(np.asarray(g.labels).tolist()):
            h.add_node(i, label=lab)
        h.add_edges_from(map(tuple, g.edge_list().tolist()))
    matcher = isomorphism.GraphMatcher(gt, gq, node_match=lambda a, b: a["label"] == b["label"])
    out = set()
    for mapping in matcher.subgraph_monomorphisms_iter():
        phi = np.full(len(mapping), -1, dtype=np.int64)
        for t, j in mapping.items():
            phi[j] = int(cand.global_ids[t])
        out.add(tuple(phi.tolist()))
    return out


def test_recall_is_the_share_of_embeddings_inside_each_candidate(estore):
    gstore, _, qs = estore
    c = _cascade(gstore)
    codes_of = gstore.sigma("R0").query_codes
    checked = 0
    for q in [q for q in qs.queries if q.kind == "positive"][:6]:
        rec = en.enumerate_query(c, q, limit=100_000, time_limit=60.0, solve_steps=True)
        truth = _all_embeddings(gstore, q)
        assert rec["status"] == "complete" and rec["sound"] and rec["planted_found"]
        assert rec["global"]["embeddings"] == len(truth) and len(truth) >= 1
        codes = codes_of(q.payload, gstore)
        ranking = c.bank.rank("index", c.query_inputs(q, codes))[0]
        selector = rankers.make_selector("hybrid", ranking, gstore.part)
        for step in rec["steps"]:
            if step["final"]:
                assert step["recall"] == 1.0 and step["located"]["kept"] == len(truth)
                assert "solve" not in step
                continue
            cand = gstore.candidate(selector.prefix(step["budget"]), codes, "R0")
            inside = {phi for phi in truth if np.isin(np.asarray(phi), cand.global_ids).all()}
            assert step["located"]["kept"] == len(inside) and step["located"]["unsound"] == 0
            assert step["located"]["outside"] == len(truth) - len(inside)
            assert step["recall"] == pytest.approx(len(inside) / len(truth))
            assert step["solve"]["status"] == "complete" and step["solve"]["consistent"] is True
            assert step["solve"]["embeddings"] == len(inside)
            assert step["planted"] in (("kept",) if np.isin(q.planted, cand.global_ids).all() else (None,))
            checked += 1
    assert checked


def test_partial_enumeration_keeps_the_first_embeddings(estore):
    gstore, _, qs = estore
    c = _cascade(gstore)
    q = next((q for q in qs.queries if q.kind == "positive" and len(_all_embeddings(gstore, q)) >= 2), None)
    if q is None:
        pytest.skip("every positive has a single embedding")
    rec = en.enumerate_query(c, q, limit=1, time_limit=60.0, solve_steps=True)
    assert rec["status"] == "partial" and rec["global"]["embeddings"] == 1 and rec["sound"]
    for step in rec["steps"]:
        assert sum(step["located"].values()) == 1
        if not step["final"]:
            assert step["solve"]["status"] == "skipped"
    exact = en.enumerate_query(c, q, limit=len(_all_embeddings(gstore, q)), time_limit=60.0)
    assert exact["status"] == "complete"  # asking for one more than the limit proves the list complete


class WrongEmbeddings:
    name = "fake_wrong"

    def solve(self, query, target, *, time_limit, max_solutions=1):
        if target.n < query.n:
            return backends.SolveOutcome(backends.NONE, [], 0.0, self.name, {})
        return backends.SolveOutcome(backends.FOUND, [np.arange(query.n)[::-1].copy()], 0.0, self.name, {})


def test_rejected_embeddings_and_lost_embeddings_are_unsound(estore, monkeypatch):
    gstore, _, qs = estore
    positives = [q for q in qs.queries if q.kind == "positive"]
    c = cascade.Cascade(gstore, "R0", "filterall", WrongEmbeddings())
    recs = [en.enumerate_query(c, q, limit=10, time_limit=10.0) for q in positives[:4]]
    bad = [r for r in recs if r["status"] == "error"]
    assert bad and all(r["error"].startswith("witness rejected") and r["sound"] is False for r in bad)
    # a pruning step that loses an embedding of its candidate is flagged
    real = en.prune.locate
    monkeypatch.setattr(en.prune, "locate", lambda result, cand, nodes: {**real(result, cand, nodes),
                                                                         "component": "split"})
    rec = en.enumerate_query(_cascade(gstore), positives[0], limit=50, time_limit=30.0)
    assert rec["sound"] is False and rec["steps"][-1]["located"]["unsound"] == rec["global"]["embeddings"]


def _argv(copy: Path, out: Path, *extra: str, split: str = "dev") -> list[str]:
    return ["--store-root", str(copy), "--dataset", "synth", "--queries", str(copy / "queries" / split),
            "--out", str(out), *extra]


def test_runner_writes_resumable_records(estore, tmp_path):
    gstore, copy, qs = estore
    out = tmp_path / "e15"
    argv = _argv(copy, out, "--arms", "jigsaw_index", "filterall", "jigsaw_index+halo_hops=2", "--backend",
                 "networkx", "--budgets", "1", "2", "--limit", "500", "--T", "30", "--positives", "4",
                 "--solve-steps")
    assert en.main(argv) == 0
    info = json.loads((out / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert info["status"] == "complete" and info["experiment"] == "E15" and info["split"] == "dev"
    assert info["config"]["limit"] == 500 and info["config"]["solve_steps"] is True
    assert info["config"]["cascade_options"]["R0"]["jigsaw_index+backend=networkx+halo_hops=2"]["halo"] == 2
    uids = None
    for arm in ("jigsaw_index+backend=networkx", "filterall+backend=networkx",
                "jigsaw_index+backend=networkx+halo_hops=2"):
        recs = en.rx.read_records(out / "records" / "R0" / f"{arm}.jsonl")
        assert recs and all(r["schema"] == en.SCHEMA and r["arm"] == arm and r["kind"] == "positive" for r in recs)
        assert all(r["sound"] and r["status"] == "complete" for r in recs), [r.get("error") for r in recs]
        uids = uids or {r["uid"] for r in recs}
        assert {r["uid"] for r in recs} == uids and len(uids) == 4
        for r in recs:
            recalls = [s["recall"] for s in r["steps"]]
            assert recalls[-1] == 1.0 and recalls == sorted(recalls)  # nested prefixes never lose an embedding
    summary = en.execute(en.build_parser().parse_args(argv))
    assert summary["attempt"] == 2 and all(v["run"] == 0 for v in summary["regimes"]["R0"]["arms"].values())
    with pytest.raises(SystemExit, match="other settings"):
        en.main(_argv(copy, out, "--arms", "jigsaw_index", "filterall", "jigsaw_index+halo_hops=2", "--backend",
                      "networkx", "--budgets", "1", "2", "--limit", "7", "--T", "30", "--positives", "4",
                      "--solve-steps"))
    with pytest.raises(SystemExit, match="label-filtered"):
        en.main(_argv(copy, tmp_path / "x", "--arms", "jigsaw_index+label_filter=0", "--budgets", "1"))
    assert not (tmp_path / "x").exists()


def test_sealed_sets_need_the_protocol_settings_and_the_gate(estore, tmp_path, monkeypatch):
    _, copy, _ = estore
    log = tmp_path / "UNSEAL_LOG.jsonl"
    monkeypatch.setattr(seal, "UNSEAL_LOG", log)
    out = tmp_path / "e15"
    with pytest.raises(seal.SealError, match="--unseal"):
        en.main(_argv(copy, out, "--arms", "jigsaw_index", split="test"))
    with pytest.raises(seal.SealError, match="not frozen"):
        en.main(_argv(copy, out, "--arms", "jigsaw_index", "--unseal", split="test"))
    refused = {("--budgets", "1"): "--budgets", ("--limit", "5"): "--limit 5", ("--T", "9"): "--T 9",
               ("--positives", "3"): "--positives 3", ("--max-queries", "2"): "--max-queries",
               ("--backend", "networkx"): "--backend", ("--arms", "random"): "arms outside E15",
               ("--partition", "kahip_fast"): "--partition kahip_fast", ("--regimes", "R1"): "regimes outside"}
    for extra, match in refused.items():
        arms = () if "--arms" in extra else ("--arms", "jigsaw_index")
        with pytest.raises(SystemExit, match=match):
            en.main(_argv(copy, out, *arms, *extra, "--unseal", split="test"))
    assert not log.exists() and not out.exists()


def test_plan_arms_follow_the_protocol():
    first = int(protocol.SEEDS["training"][0])
    assert en.plan_arms() == [f"jigsaw@{first}", "jigsaw_index"]
    assert all(en.rx.Arm.parse(a) for a in en.plan_arms())
    assert verify.check_embedding is not None
