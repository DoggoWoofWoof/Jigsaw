from __future__ import annotations

import csv
import json

import pytest

from scripts.canonical import numbers_json as nj


def _numbers(split="dev"):
    n = nj.Numbers(split=split, frozen=False, protocol_hash="h" * 64)
    n.source("E1.arxiv", path="runs/E1/arxiv", manifest_sha256="0" * 64, experiment="E1", dataset="arxiv")
    return n


def test_units_and_intervals_are_checked():
    n = _numbers()
    n.number("E1.arxiv.R0.jigsaw.solved_30", 95.4, "pct", ci=(93.1, 96.8), n=1800, source=["E1.arxiv"])
    n.number("PE2.arxiv.diff", -0.3, "pp", ci=(-0.9, 0.2), source=["E1.arxiv"])
    n.number("E4.arxiv.cost_mean", -1.25, "number")
    n.number("E1.arxiv.queries", 1800, "count")
    n.number("E2.mag.bytes", 123456789, "bytes")
    n.number("PE3.mag.ratio", 5.4, "ratio", ci=(4.9, 6.0))
    n.number("PE2.arxiv.p", 3e-6, "pvalue")
    n.number("E1.arxiv.note", "router", "text")
    key = n.rate("E1.arxiv.R0.filterall.solved_30", 1595, 1800, source=["E1.arxiv"])
    assert n.numbers[key]["n"] == 1800 and n.numbers[key]["ci"][0] < n.numbers[key]["value"] < n.numbers[key]["ci"][1]
    bad = [("a.b", 101.0, "pct", {}), ("a.c", -0.1, "frac", {}), ("a.d", 1.5, "count", {}), ("a.e", -1, "count", {}),
           ("a.f", 0.0, "ratio", {}), ("a.g", float("nan"), "number", {}), ("a.h", 2.0, "pvalue", {}),
           ("a.i", 1.0, "furlongs", {}), ("a.j", 3, "text", {}), ("a.k", 50.0, "pct", {"ci": (60.0, 40.0)}),
           ("a.l", 50.0, "pct", {"source": ["nope"]}), ("a.m", True, "count", {}), ("a.n", 5.0, "pp", {"n": -1}),
           ("bad key", 1.0, "number", {}), ("a..b", 1.0, "number", {})]
    for key, value, unit, kw in bad:
        with pytest.raises(nj.NumbersError):
            n.number(key, value, unit, **kw)
    with pytest.raises(nj.NumbersError):
        n.number("E1.arxiv.queries", 1, "count")  # twice


def test_tables_figures_write_and_load(tmp_path):
    n = _numbers()
    n.number("E1.arxiv.R0.jigsaw.solved_30", 95.4, "pct", ci=(93.1, 96.8), n=1800)
    n.number("E1.arxiv.R0.filterall.solved_30", 96.0, "pct")
    n.table("T.main", ["Dataset", "Jigsaw", "FilterAll"],
            [["Arxiv", {"key": "E1.arxiv.R0.jigsaw.solved_30", "ci": True}, {"key": "E1.arxiv.R0.filterall.solved_30"}]])
    with pytest.raises(nj.NumbersError):
        n.table("T.bad", ["a"], [[{"key": "missing"}]])
    with pytest.raises(nj.NumbersError):
        n.table("T.noci", ["a"], [[{"key": "E1.arxiv.R0.filterall.solved_30", "ci": True}]])
    with pytest.raises(nj.NumbersError):
        n.table("T.width", ["a", "b"], [["x"]])
    n.figure("F.solved", "curve", [{"series": "jigsaw", "x": 0.05, "y": 10.0, "lo": 8.0, "hi": 12.0},
                                   {"series": "jigsaw", "x": 30.0, "y": 95.4}], x="t (s)", y="solved (%)", log_x=True)
    n.figure("F.bars", "bars", [{"group": "R0", "series": "jigsaw", "y": 95.4}], x="regime", y="solved (%)")
    with pytest.raises(nj.NumbersError):
        n.figure("F.bad", "curve", [{"series": "a", "y": 1.0}], x="x", y="y")
    path = n.write(tmp_path)
    data = nj.load(path)
    assert data["schema"] == nj.SCHEMA and data["tables"]["T.main"]["rows"][0][1] == {
        "key": "E1.arxiv.R0.jigsaw.solved_30", "ci": True}
    assert data["figures"]["F.solved"] == {"type": "curve", "csv": "figures/F.solved.csv", "x": "t (s)",
                                           "y": "solved (%)", "log_x": True}
    with open(tmp_path / "figures" / "F.solved.csv", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["series", "x", "y", "lo", "hi"] and rows[2] == ["jigsaw", "30.0", "95.4", "", ""]
    (tmp_path / "figures" / "F.bars.csv").unlink()
    with pytest.raises(nj.NumbersError):
        nj.load(path)
    broken = json.loads(path.read_text())
    broken["numbers"]["E1.arxiv.R0.jigsaw.solved_30"]["value"] = 140.0
    path.write_text(json.dumps(broken))
    with pytest.raises(nj.NumbersError):
        nj.load(path)


def test_merge_refuses_conflicts():
    a, b = _numbers(), _numbers()
    a.number("x.one", 1.0, "number")
    b.number("x.two", 2.0, "number")
    a.merge(b)
    assert set(a.numbers) == {"x.one", "x.two"}
    c = _numbers()
    c.number("x.one", 1.0, "number")
    with pytest.raises(nj.NumbersError):
        a.merge(c)
    with pytest.raises(nj.NumbersError):
        a.merge(_numbers(split="test"))
