"""build_paper.py (numbers.json -> LaTeX) and validate_paper_numbers.py (no hand-typed result).

Everything is written into pytest tmp dirs. The pdflatex tests skip when TeX is not
installed. protocol.FROZEN_PROTOCOL_HASH is None until the freeze, so every test
stands FROZEN_HASH in for it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.canonical import build_paper as bp
from scripts.canonical import numbers_json as nj
from scripts.canonical import protocol
from scripts.canonical import validate_paper_numbers as vpn

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = REPO_ROOT / "scripts" / "canonical" / "paper_allowlist.json"
FROZEN_HASH = "f" * 64
KEY = "E1.arxiv.R0.jigsaw.solved.t30"
FA = "E1.arxiv.R0.filterall.solved.t30"
TIME = "E1.arxiv.R0.jigsaw.solve_time.median"
DIFF = "PE2.arxiv.R0.jigsaw_vs_filterall.diff"
GB = "E2.mag.R2-512.graph_gb"
EXPECTED_VALUES = {
    KEY: r"95.4\%",
    FA: r"96.0\%",
    "E1.arxiv.R0.jigsaw.hit_rate": "0.954",
    "E1.arxiv.queries": "1{,}800",
    TIME: r"2.68\,s",
    "E1.arxiv.R0.jigsaw.retrieval_ms": r"12.3\,ms",
    "E1.arxiv.R0.jigsaw.peak_rss": r"2.43\,GiB",
    DIFF: r"\ensuremath{-}0.3\,pp",
    "PE2.arxiv.R0.zero_diff": r"0.0\,pp",
    "PE2.arxiv.R0.cost.mean": r"\ensuremath{-}1{,}234.57",
    GB: r"1.30\,GB",
    "E2.mag.overlap_ratio": r"5.4\ensuremath{\times}",
    "PE2.arxiv.R0.p": r"\ensuremath{p = 0.003}",
    "PE2.arxiv.R1.p": r"\ensuremath{p < 10^{-4}}",
    "E1.arxiv.router": r"mean\_rrf \& co",
}
EXPECTED_INTERVALS = {KEY: "[93.1, 96.8]", FA: "[95.1, 96.9]", DIFF: r"[\ensuremath{-}0.9, 0.2]",
                      "E2.mag.overlap_ratio": "[4.9, 6.0]"}
PREAMBLE = [r"\documentclass{article}", r"\usepackage{booktabs}", r"\usepackage{graphicx}", r"\usepackage{amsmath}",
            r"\input{generated/numbers}", r"\begin{document}", ""]
END = ["", r"\end{document}"]
FIRST_BODY_LINE = len(PREAMBLE) + 1


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numbers(split: str = "test", frozen: bool = True, protocol_hash: str = FROZEN_HASH) -> nj.Numbers:
    return nj.Numbers(split=split, frozen=frozen, protocol_hash=protocol_hash)


def _file_a() -> nj.Numbers:
    n = _numbers()
    n.source("E1.arxiv", path="runs/E1/arxiv", manifest_sha256="0" * 64, experiment="E1", dataset="arxiv")
    n.number(KEY, 95.4, "pct", ci=(93.1, 96.8), n=1800, source=["E1.arxiv"])
    n.number(FA, 96.0, "pct", ci=(95.05, 96.85), n=1800, source=["E1.arxiv"])
    n.number("E1.arxiv.R0.jigsaw.hit_rate", 0.9535, "frac")
    n.number("E1.arxiv.queries", 1800, "count")
    n.number(TIME, 2.675, "seconds")
    n.number("E1.arxiv.R0.jigsaw.retrieval_ms", 12.25, "ms")
    n.number("E1.arxiv.R0.jigsaw.peak_rss", 2610000000, "bytes")
    n.table("T.main", ["Dataset", "Jigsaw", "FilterAll", "Median time"],
            [["Arxiv", {"key": KEY, "ci": True}, {"key": FA, "ci": True}, {"key": TIME}],
             ["R0 (all_blocks) & more", "--", "--", "--"]], notes="solve rate within 30 s")
    n.figure("F.solved", "curve",
             [{"series": "jigsaw", "x": 0.05, "y": 10.0, "lo": 8.0, "hi": 12.0},
              {"series": "jigsaw", "x": 1.0, "y": 60.0, "lo": 57.0, "hi": 63.0},
              {"series": "jigsaw", "x": 30.0, "y": 95.4, "lo": 93.1, "hi": 96.8},
              {"series": "filterall", "x": 0.05, "y": 2.0},
              {"series": "filterall", "x": 1.0, "y": 40.0},
              {"series": "filterall", "x": 30.0, "y": 96.0},
              {"series": "zeta_custom", "x": 1.0, "y": 20.0},
              {"series": "zeta_custom", "x": 30.0, "y": 50.0}],
             x="time budget (s)", y="solved (%)", log_x=True)
    return n


def _file_b() -> nj.Numbers:
    n = _numbers()
    n.source("PE2.arxiv", path="runs/PE2/arxiv", manifest_sha256="1" * 64, experiment="PE2", dataset="arxiv")
    n.number(DIFF, -0.3, "pp", ci=(-0.9, 0.2), source=["PE2.arxiv"])
    n.number("PE2.arxiv.R0.zero_diff", -0.04, "pp")
    n.number("PE2.arxiv.R0.cost.mean", -1234.5678, "number")
    n.number(GB, 1.3, "gb")
    n.number("E2.mag.overlap_ratio", 5.4, "ratio", ci=(4.9, 6.0))
    n.number("PE2.arxiv.R0.p", 0.003, "pvalue")
    n.number("PE2.arxiv.R1.p", 3e-6, "pvalue")
    n.number("E1.arxiv.router", "mean_rrf & co", "text")
    n.figure("F.bars", "bars",
             [{"group": "R0", "series": "jigsaw", "y": 95.4, "lo": 93.1, "hi": 96.8},
              {"group": "R0", "series": "filterall", "y": 96.0, "lo": 95.0, "hi": 96.9},
              {"group": "R1", "series": "jigsaw", "y": 90.0},
              {"group": "R1", "series": "filterall", "y": 91.0}], x="regime", y="solved (%)")
    return n


@pytest.fixture(autouse=True)
def _frozen_hash(monkeypatch):
    monkeypatch.setattr(protocol, "FROZEN_PROTOCOL_HASH", FROZEN_HASH)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Two frozen test numbers.json files built into ROOT/paper/generated."""
    root = tmp_path_factory.mktemp("paperbuild")
    a = _file_a().write(root / "in" / "a")
    b = _file_b().write(root / "in" / "b")
    out = root / "paper" / "generated"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(protocol, "FROZEN_PROTOCOL_HASH", FROZEN_HASH)
        result = bp.build([a, b], out_dir=out)
        removed = bp.write_build(result, out)
    return SimpleNamespace(root=root, a=a, b=b, out=out, paper=root / "paper", result=result, removed=removed)


def _copy(built, name: str) -> Path:
    """A copy of the generated dir at the same depth, so its relative input paths stay valid."""
    target = built.paper / name
    shutil.copytree(built.out, target)
    return target


def _snapshot(folder: Path) -> dict[str, tuple[int, bytes]]:
    return {p.relative_to(folder).as_posix(): (p.stat().st_mtime_ns, p.read_bytes())
            for p in sorted(folder.rglob("*")) if p.is_file()}


def _paper(folder: Path, name: str, body: list[str], preamble: list[str] = PREAMBLE) -> Path:
    path = folder / f"{name}.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(preamble + body + END) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------- formatting
@pytest.mark.parametrize("value, unit, digits, expected", [
    (95.4, "pct", None, r"95.4\%"),
    (0.05, "pct", 1, r"0.1\%"),
    (100, "pct", None, r"100.0\%"),
    (95.4321, "pct", 2, r"95.43\%"),
    (-0.3, "pp", None, r"\ensuremath{-}0.3\,pp"),
    (0.3, "pp", None, r"0.3\,pp"),
    (-0.05, "pp", None, r"\ensuremath{-}0.1\,pp"),
    (-0.04, "pp", None, r"0.0\,pp"),
    (-100.0, "pp", 0, r"\ensuremath{-}100\,pp"),
    (0.954, "frac", None, "0.954"),
    (0.9535, "frac", None, "0.954"),
    (-1234.5678, "number", None, r"\ensuremath{-}1{,}234.57"),
    (2.675, "number", 2, "2.68"),
    (-0.0, "number", None, "0.00"),
    (-0.004, "number", None, "0.00"),
    (7, "number", 0, "7"),
    (1800, "count", None, "1{,}800"),
    (1800.0, "count", None, "1{,}800"),
    (1234567, "count", 3, "1{,}234{,}567"),
    (999, "count", None, "999"),
    (0, "count", None, "0"),
    (2.675, "seconds", None, r"2.68\,s"),
    (2.74, "seconds", None, r"2.74\,s"),
    (12.25, "ms", None, r"12.3\,ms"),
    (512, "bytes", None, r"512\,B"),
    (1023, "bytes", None, r"1{,}023\,B"),
    (1024, "bytes", None, r"1.00\,KiB"),
    (1048570, "bytes", None, r"1{,}023.99\,KiB"),
    (1048575, "bytes", None, r"1.00\,MiB"),
    (2610000000, "bytes", None, r"2.43\,GiB"),
    (5 * 1024 ** 4, "bytes", None, r"5.00\,TiB"),
    (3 * 1024 ** 5, "bytes", None, r"3{,}072.00\,TiB"),
    (1.3, "gb", None, r"1.30\,GB"),
    (5.4, "ratio", None, r"5.4\ensuremath{\times}"),
    (921000, "ratio", 0, r"921{,}000\ensuremath{\times}"),
    (0.003, "pvalue", None, r"\ensuremath{p = 0.003}"),
    (0.0449, "pvalue", None, r"\ensuremath{p = 0.045}"),
    (0.04999, "pvalue", None, r"\ensuremath{p = 0.05}"),
    (0.5, "pvalue", None, r"\ensuremath{p = 0.5}"),
    (1.0, "pvalue", None, r"\ensuremath{p = 1}"),
    (1e-4, "pvalue", None, r"\ensuremath{p = 0.0001}"),
    (9.99e-5, "pvalue", None, r"\ensuremath{p < 10^{-4}}"),
    (0.0, "pvalue", None, r"\ensuremath{p < 10^{-4}}"),
    ("a_b & 50% #1 {x} ~^\\", "text", None,
     r"a\_b \& 50\% \#1 \{x\} \textasciitilde{}\textasciicircum{}\textbackslash{}"),
])
def test_format_value_every_unit(value, unit, digits, expected):
    assert bp.format_value(value, unit, digits, key="E1.test") == expected


def test_rounding_is_decimal_half_up_not_binary_float():
    assert round(2.675, 2) == 2.67 and round(95.05, 1) == 95.0  # what float rounding would print
    assert bp.format_value(2.675, "seconds") == r"2.68\,s"
    assert bp.format_value(95.05, "pct") == r"95.1\%"
    assert bp.format_value(0.05, "pct", 1) == r"0.1\%"
    assert bp.format_value(0.125, "frac", 2) == "0.13"


def test_pvalue_floor_is_a_module_constant(monkeypatch):
    monkeypatch.setattr(bp, "PVALUE_FLOOR_EXPONENT", -3)
    assert bp.format_value(0.0005, "pvalue") == r"\ensuremath{p < 10^{-3}}"
    assert bp.format_value(0.001, "pvalue") == r"\ensuremath{p = 0.001}"


def test_intervals_use_the_value_rule_without_unit():
    assert bp.format_interval(93.1, 96.8, "pct") == "[93.1, 96.8]"
    assert bp.format_interval(-0.9, 0.2, "pp") == r"[\ensuremath{-}0.9, 0.2]"
    assert bp.format_interval(4.9, 6.0, "ratio") == "[4.9, 6.0]"
    assert bp.format_interval(1500, 2100, "count") == "[1{,}500, 2{,}100]"
    # bytes keep the scale of the value, so both bounds are in GiB
    assert bp.format_interval(2.0e9, 3.0e9, "bytes", center=2.61e9) == "[1.86, 2.79]"
    assert bp.format_count(1800) == "1{,}800"


@pytest.mark.parametrize("value, unit, digits, message", [
    (float("nan"), "pct", None, "not finite"),
    (float("inf"), "number", None, "not finite"),
    (float("-inf"), "pp", None, "not finite"),
    (1.5, "count", None, "must be an integer"),
    (1.0, "furlongs", None, "unknown unit"),
    (1.0, "pct", 7, "digits"),
    (1.0, "pct", -1, "digits"),
    (True, "count", None, "not a number"),
    ("95.4", "pct", None, "not a number"),
    (95.4, "text", None, "must be a string"),
])
def test_format_errors_name_the_key(value, unit, digits, message):
    with pytest.raises(bp.BuildError, match=r"E9\.bad\.key: .*" + message):
        bp.format_value(value, unit, digits, key="E9.bad.key")


def test_series_styles_are_fixed_and_stable():
    slots = bp.assign_series_slots(["zeta", "jigsaw", "Mean-RRF", "alpha", "FilterAll"])
    fixed = len(bp.FIXED_SERIES)
    assert slots == {"FilterAll": 1, "Mean-RRF": bp.FIXED_SERIES.index("mean_rrf"), "alpha": fixed, "jigsaw": 0,
                     "zeta": fixed + 1}
    assert bp.assign_series_slots(["jigsaw", "direct"]) == {"direct": 2, "jigsaw": 0}
    assert bp.series_style(0).colour == bp.PALETTE[0] == "#0072B2"
    assert len({(s.colour, s.linestyle, s.marker) for s in map(bp.series_style, range(36))}) == 36
    assert len({(s.colour, s.hatch) for s in map(bp.series_style, range(36))}) == 36


# ---------------------------------------------------------------------- inputs and gate
def test_key_grammar(tmp_path, built):
    numbers_tex = (built.out / "numbers.tex").read_text(encoding="utf-8")
    assert r"\csname jnum@E2.mag.R2-512.graph_gb\endcsname{1.30\,GB}" in numbers_tex
    assert all(nj.KEY_RE.match(key) for key in built.result.record["keys"])
    base = _numbers()
    base.number("E1.ok", 1.0, "number")
    good = base.to_dict()
    for field, ident in (("numbers", "bad key"), ("numbers", "a..b"), ("numbers", ".lead"), ("numbers", "E1.x$"),
                         ("tables", "T bad")):
        data = json.loads(json.dumps(good))
        if field == "numbers":
            data["numbers"][ident] = {"value": 1.0, "unit": "number", "ci": None, "n": None, "source": []}
        else:
            data["tables"][ident] = {"columns": ["a"], "rows": [["x"]]}
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(bp.BuildError, match="bad"):
            bp.load_input(path)
    data = json.loads(json.dumps(good))
    data["numbers"]["E1.ok"]["valeu"] = 2.0
    path = tmp_path / "typo.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(bp.BuildError, match="unknown fields"):
        bp.load_input(path)


def test_merge_refuses_conflicts(tmp_path):
    def write(name, fill, **kw):
        n = _numbers(**kw)
        fill(n)
        return n.write(tmp_path / name)

    def value(v):
        return lambda n: n.number("E1.x.value", v, "number")

    def table(key):
        return lambda n: (n.number(key, 1.0, "number"), n.table("T.shared", ["k"], [[{"key": key}]]))

    def figure(key):
        return lambda n: (n.number(key, 1.0, "number"),
                          n.figure("F.shared", "curve", [{"series": "a", "x": 1.0, "y": 1.0}], x="x", y="y"))

    def source(path):
        return lambda n: n.source("E1.run", path=path, manifest_sha256=None, experiment="E1", dataset="arxiv")

    out = tmp_path / "out"
    a = write("a", value(1.0))
    assert bp.build([a, write("same", value(1.0))], out_dir=out).record["keys"] == ["E1.x.value"]
    with pytest.raises(bp.BuildError, match="E1.x.value has different entries"):
        bp.build([a, write("other", value(2.0))], out_dir=out)
    with pytest.raises(bp.BuildError, match="table T.shared is defined in both"):
        bp.build([write("t1", table("E1.t.one")), write("t2", table("E1.t.two"))], out_dir=out)
    with pytest.raises(bp.BuildError, match="figure F.shared is defined in both"):
        bp.build([write("f1", figure("E1.f.one")), write("f2", figure("E1.f.two"))], out_dir=out)
    with pytest.raises(bp.BuildError, match="source E1.run differs"):
        bp.build([write("s1", source("runs/one")), write("s2", source("runs/two"))], out_dir=out)
    with pytest.raises(bp.BuildError, match="given twice"):
        bp.build([a, a], out_dir=out)
    dev_a = write("h1", value(1.0), split="dev", frozen=False, protocol_hash="1" * 64)
    dev_b = write("h2", lambda n: n.number("E1.y", 1.0, "number"), split="dev", frozen=False, protocol_hash="2" * 64)
    with pytest.raises(bp.BuildError, match="different protocol hashes"):
        bp.build([dev_a, dev_b], out_dir=out, allow_dev=True)
    assert not out.exists()


@pytest.mark.parametrize("split, frozen, protocol_hash, reason", [
    ("dev", False, FROZEN_HASH, "split is 'dev'"),
    ("validation", True, FROZEN_HASH, "split is 'validation'"),
    ("test", False, FROZEN_HASH, "frozen is false"),
    ("test", True, "0" * 64, "is not protocol.FROZEN_PROTOCOL_HASH"),
])
def test_dev_numbers_are_refused(tmp_path, capsys, split, frozen, protocol_hash, reason):
    n = _numbers(split, frozen, protocol_hash)
    n.number("E1.x.value", 1.0, "number")
    path = n.write(tmp_path / "in")
    out = tmp_path / "out"
    with pytest.raises(bp.RefusedError, match=re.escape(reason)):
        bp.build([path], out_dir=out)
    assert bp.main(["--numbers", str(path), "--out", str(out)]) == bp.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "refused" in err and "--allow-dev" in err and reason in err
    assert not out.exists()


def test_allow_dev_marks_every_output(tmp_path, capsys):
    frozen = _numbers()
    frozen.number(FA, 96.0, "pct")
    dev = _numbers("dev", False)
    dev.number(KEY, 95.4, "pct", ci=(93.1, 96.8), n=1800)
    dev.table("T.dev", ["Method", "Solved"], [["Jigsaw", {"key": KEY, "ci": True}]])
    paths = [frozen.write(tmp_path / "frozen"), dev.write(tmp_path / "dev")]
    out = tmp_path / "paper" / "generated"
    with pytest.raises(bp.RefusedError):  # one development input is enough to refuse
        bp.build(paths, out_dir=out)
    assert bp.main(["--numbers", *map(str, paths), "--out", str(out), "--allow-dev"]) == bp.EXIT_OK
    assert "DEV build" in capsys.readouterr().err
    text = (out / "numbers.tex").read_text(encoding="utf-8")
    assert r"\protected\def\jigsawdevbuild{" + bp.DEV_MARKER + "}" in text
    assert r"\PackageWarningNoLine{jigsaw}" in text and r"\AddToHook{shipout/foreground}" in text
    assert bp.DEV_BANNER in text and "% mode: DEV" in text
    assert "% DEV BUILD" in (out / "tables" / "T.dev.tex").read_text(encoding="utf-8")
    record = json.loads((out / "build.json").read_text(encoding="utf-8"))
    assert record["mode"] == "dev"
    assert record["dev_reasons"] == ["../../dev/numbers.json: frozen is false",
                                     "../../dev/numbers.json: split is 'dev', not 'test'"]
    assert bp.main(["--numbers", *map(str, paths), "--out", str(out), "--draft", "--check"]) == bp.EXIT_OK
    spec = {"type": "curve", "csv": "figures/F.x.csv", "x": "x", "y": "y"}
    rows = [{"series": "jigsaw", "x": 1.0, "y": 2.0, "lo": None, "hi": None},
            {"series": "jigsaw", "x": 2.0, "y": 3.0, "lo": None, "hi": None}]
    slots = bp.assign_series_slots(["jigsaw"])
    stamped = bp.render_figure("F.x", spec, rows, slots, dev=True)
    plain = bp.render_figure("F.x", spec, rows, slots, dev=False)
    assert b"DejaVuSans-Bold" in stamped and b"DejaVuSans-Bold" not in plain  # the bold DEV NUMBERS stamp


# ---------------------------------------------------------------------- outputs
def test_numbers_tex(built):
    text = (built.out / "numbers.tex").read_text(encoding="utf-8")
    for key, value in EXPECTED_VALUES.items():
        assert rf"\expandafter\def\csname jnum@{key}\endcsname{{{value}}}" + "\n" in text
    for key, interval in EXPECTED_INTERVALS.items():
        assert rf"\expandafter\def\csname jci@{key}\endcsname{{{interval}}}" + "\n" in text
    assert text.count(r"\csname jci@") == len(EXPECTED_INTERVALS)
    assert rf"\expandafter\def\csname jn@{KEY}\endcsname{{1{{,}}800}}" in text and text.count(r"\csname jn@") == 2
    assert f"% input: ../../in/a/numbers.json sha256 {_sha(built.a)}\n" in text
    assert f"% input: ../../in/b/numbers.json sha256 {_sha(built.b)}\n" in text
    assert f"% merged_sha256: {built.result.record['merged_sha256']}\n" in text
    assert re.search(r"^% merged_sha256: [0-9a-f]{64}$", text, re.M)
    assert "% keys: 15\n" in text and "% mode: frozen" in text
    assert r"\protected\def\jigsawdevbuild{}" + "\n" in text and "DEV" not in text
    assert text.endswith("\\endinput\n") and built.removed == []


def test_build_json(built):
    record = json.loads((built.out / "build.json").read_text(encoding="utf-8"))
    assert record["schema"] == bp.BUILD_SCHEMA and record["mode"] == "frozen" and record["dev_reasons"] == []
    assert record["protocol_hash"] == record["frozen_protocol_hash"] == FROZEN_HASH
    assert record["keys"] == sorted(EXPECTED_VALUES) and record["tables"] == ["T.main"]
    assert record["figures"] == ["F.bars", "F.solved"] and record["removed_stale"] == []
    inputs = {(item["kind"], item["path"]): item["sha256"] for item in record["inputs"]}
    assert inputs == {("numbers", "../../in/a/numbers.json"): _sha(built.a),
                      ("numbers", "../../in/b/numbers.json"): _sha(built.b),
                      ("figure_csv", "../../in/a/figures/F.solved.csv"): _sha(built.a.parent / "figures/F.solved.csv"),
                      ("figure_csv", "../../in/b/figures/F.bars.csv"): _sha(built.b.parent / "figures/F.bars.csv")}
    assert set(record["generated"]) == {"numbers.tex", "tables/T.main.tex", "figures/F.bars.pdf",
                                        "figures/F.solved.pdf"}
    assert all(_sha(built.out / rel) == sha for rel, sha in record["generated"].items())
    assert record["renderer"]["matplotlib"]
    assert not [p for p in built.out.rglob("*") if p.name.endswith(".tmp")]  # atomic writes leave nothing behind


def test_table_rendering(built):
    table = json.loads(built.a.read_text(encoding="utf-8"))["tables"]["T.main"]
    sha = hashlib.sha256(protocol.canonical_json(table).encode("ascii")).hexdigest()
    expected = "\n".join([
        "% Generated by scripts/canonical/build_paper.py from numbers.json table T.main; do not edit.",
        f"% table T.main content sha256 {sha}",
        "% notes: solve rate within 30 s",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Dataset & Jigsaw & FilterAll & Median time \\",
        r"\midrule",
        rf"Arxiv & \jnum{{{KEY}}} \jci{{{KEY}}} & \jnum{{{FA}}} \jci{{{FA}}} & \jnum{{{TIME}}} \\",
        r"R0 (all\_blocks) \& more & -- & -- & -- \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]) + "\n"
    assert (built.out / "tables" / "T.main.tex").read_text(encoding="utf-8") == expected


def test_figures_are_byte_deterministic_within_a_process(built):
    again = bp.build([built.a, built.b], out_dir=built.out)
    assert sorted(again.files) == sorted(built.result.files)
    for rel, data in again.files.items():
        assert (built.out / rel).read_bytes() == data, rel
    assert (built.out / "build.json").read_bytes() == bp._json_bytes(dict(again.record, removed_stale=[]))
    for name in ("F.solved.pdf", "F.bars.pdf"):
        data = (built.out / "figures" / name).read_bytes()
        assert data.startswith(b"%PDF-") and b"CreationDate" not in data and b"ModDate" not in data
        assert b"/Type3" not in data and b"/CIDFontType2" in data  # pdf.fonttype 42: embedded TrueType
        assert re.search(rb"/MediaBox \[ 0 0 239\.76 158\.4 \]", data)  # 3.33 x 2.2 in


def test_figures_are_byte_deterministic_across_processes(built):
    out = built.paper / "generated_cli"  # same depth as generated, so even build.json must match
    code = ("import sys; from scripts.canonical import build_paper, protocol; "
            f"protocol.FROZEN_PROTOCOL_HASH = {FROZEN_HASH!r}; "
            f"sys.exit(build_paper.main(['--numbers', {str(built.a)!r}, {str(built.b)!r}, '--out', {str(out)!r}]))")
    env = dict(os.environ, PYTHONHASHSEED="12345")
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert _snapshot_bytes(out) == _snapshot_bytes(built.out)


def _snapshot_bytes(folder: Path) -> dict[str, bytes]:
    return {rel: data for rel, (_, data) in _snapshot(folder).items()}


def test_wide_figures_and_figure_errors():
    rows = [{"series": f"s{i}", "x": x, "y": i + x, "lo": None, "hi": None} for i in range(7) for x in (1.0, 2.0)]
    wide = bp.render_figure("F.wide", {"type": "curve", "x": "x", "y": "y"}, rows,
                            bp.assign_series_slots(r["series"] for r in rows), dev=False)
    assert re.search(rb"/MediaBox \[ 0 0 504 187\.2 \]", wide)  # 7 x 2.6 in above 6 series
    slots = bp.assign_series_slots(["jigsaw"])
    bars = {"type": "bars", "x": "x", "y": "y"}
    with pytest.raises(bp.BuildError, match="outside its interval"):
        bp.render_figure("F.bad", bars, [{"group": "R0", "series": "jigsaw", "y": 5.0, "lo": 6.0, "hi": 7.0}],
                         slots, dev=False)
    with pytest.raises(bp.BuildError, match="two bars"):
        bp.render_figure("F.dup", bars, [{"group": "R0", "series": "jigsaw", "y": 5.0, "lo": None, "hi": None}] * 2,
                         slots, dev=False)
    curve = {"type": "curve"}
    for raw, message in ((b"series,x,y,lo,hi\njigsaw,0.0,1.0,,\n", "x > 0"),
                         (b"series,y,x,lo,hi\njigsaw,1.0,1.0,,\n", "header"),
                         (b"series,x,y,lo,hi\njigsaw,1.0,abc,,\n", "not a number"),
                         (b"series,x,y,lo,hi\njigsaw,1.0,nan,,\n", "not finite"),
                         (b"series,x,y,lo,hi\n", "no rows")):
        with pytest.raises(bp.BuildError, match=message):
            bp.parse_figure_csv("F.csv", dict(curve, log_x=message == "x > 0"), raw)


# ---------------------------------------------------------------------- --check and stale files
def test_check_passes_and_writes_nothing(built, capsys):
    before = _snapshot(built.out)
    assert bp.main(["--numbers", str(built.a), str(built.b), "--out", str(built.out), "--check"]) == bp.EXIT_OK
    assert "up to date" in capsys.readouterr().out
    assert _snapshot(built.out) == before
    nowhere = built.paper / "nowhere"
    assert bp.main(["--numbers", str(built.a), str(built.b), "--out", str(nowhere), "--check"]) == bp.EXIT_FAILED
    assert "missing: numbers.tex" in capsys.readouterr().out and not nowhere.exists()


def test_check_lists_stale_missing_and_extra_files(built, capsys):
    out = _copy(built, "generated_edited")
    with (out / "tables" / "T.main.tex").open("a", encoding="utf-8") as handle:
        handle.write("% edited by hand\n")
    (out / "figures" / "F.bars.pdf").unlink()
    (out / "figures" / "F.old.pdf").write_bytes(b"%PDF-old")
    before = _snapshot(out)
    assert bp.main(["--numbers", str(built.a), str(built.b), "--out", str(out), "--check"]) == bp.EXIT_FAILED
    lines = capsys.readouterr().out.splitlines()
    assert sorted(line.strip() for line in lines[1:]) == ["extra: figures/F.old.pdf", "missing: figures/F.bars.pdf",
                                                          "stale: tables/T.main.tex"]
    assert _snapshot(out) == before


def test_fresh_build_removes_stale_files(built, capsys):
    out = _copy(built, "generated_stale")
    (out / "tables" / "T.old.tex").write_text("% an old table\n", encoding="utf-8")
    assert bp.main(["--numbers", str(built.a), str(built.b), "--out", str(out)]) == bp.EXIT_OK
    assert "removed stale tables/T.old.tex" in capsys.readouterr().out
    assert not (out / "tables" / "T.old.tex").exists()
    assert json.loads((out / "build.json").read_text(encoding="utf-8"))["removed_stale"] == ["tables/T.old.tex"]
    assert bp.main(["--numbers", str(built.a), str(built.b), "--out", str(out), "--check"]) == bp.EXIT_OK


def test_cli_refuses_dev_numbers_in_a_fresh_process(tmp_path):
    n = _numbers("dev", False)
    n.number("E1.x.value", 1.0, "number")
    path = n.write(tmp_path / "in")
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, "-m", "scripts.canonical.build_paper", "--numbers", str(path), "--out",
                           str(out)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == bp.EXIT_REFUSED and "--allow-dev" in proc.stderr and not out.exists()


# ---------------------------------------------------------------------- validator
def test_validator_passes_a_generated_paper(built):
    body = [
        r"\section{Results}",
        rf"Jigsaw solves \jnum{{{KEY}}} \jci{{{KEY}}} of \jn{{{KEY}}} planted queries (Algorithm 1, VLDB 2027);",
        rf"FilterAll differs by \jnum{{{DIFF}}} (Table~\ref{{tab:main}}, \cite{{smith2020,ref2019}}). % 42 in a comment",
        r"\begin{table}[t]\centering\input{generated/tables/T.main.tex}\label{tab:main}\end{table}",
        r"\begin{figure}[t]\centering\includegraphics[width=0.95\columnwidth]{generated/figures/F.solved.pdf}",
        r"\end{figure}",
    ]
    report = vpn.validate(_paper(built.paper, "clean", body), built.out, ALLOWLIST)
    assert report.ok, report.summary()
    assert [(o.numeral, bool(o.allowed_by)) for o in report.occurrences] == [("1", True), ("2027", True)]
    assert {KEY, DIFF, FA, TIME} == {use.key for use in report.key_uses}  # FA and TIME through the table
    assert "E1.arxiv.queries" in report.unused_keys and KEY not in report.unused_keys
    assert any(name.endswith("tables/T.main.tex") for name in report.files)
    assert report.counts()["keys"] == {"errors": 0, "info": 1} and "result: PASS" in report.summary()


def test_validator_reports_undefined_keys_with_their_line(built):
    body = [rf"Known \jnum{{{KEY}}}.",
            r"Unknown \jnum{E9.no-such_key.x} here.",
            rf"No interval \jci{{E1.arxiv.queries}}, no n \jn{{{DIFF}}}.",
            r"Malformed \jnum{bad key} and \jnum missing."]
    tex = _paper(built.paper, "undefined", body)
    report = vpn.validate(tex, built.out, ALLOWLIST)
    first = FIRST_BODY_LINE
    assert sorted((f.line, f.message) for f in report.errors("keys")) == [
        (first + 1, r"undefined key E9.no-such_key.x (\jnum)"),
        (first + 2, r"\jci{E1.arxiv.queries}: the key has no interval"),
        (first + 2, rf"\jn{{{DIFF}}}: the key has no n"),
        (first + 3, r"\jnum needs a braced key"),
        (first + 3, r"malformed key 'bad key' in \jnum"),
    ]
    finding = next(f for f in report.errors("keys") if "E9" in f.message)
    assert finding.file.endswith("undefined.tex") and "E9.no-such_key.x" in finding.context
    assert f"undefined.tex:{first + 1}" in finding.format()
    assert vpn.main(["--tex", str(tex), "--generated", str(built.out), "--allowlist", str(ALLOWLIST)]) == 1


def test_validator_flags_hand_typed_numerals(built):
    tex = _paper(built.paper, "handtyped", [r"Jigsaw solves 95.4\% of", rf"1{{,}}800 queries in \jnum{{{TIME}}}."])
    report = vpn.validate(tex, built.out, ALLOWLIST)
    found = [(o.line, o.numeral) for o in report.unmatched()]
    assert found == [(FIRST_BODY_LINE, "95.4"), (FIRST_BODY_LINE + 1, "1{,}800")]
    assert report.unmatched()[0].context.startswith(r"Jigsaw solves 95.4\% of 1{,}800 queries")
    assert report.counts()["numerals"] == {"errors": 2, "info": 0} and not report.ok
    assert "hand-typed numeral '95.4'" in report.summary() and "result: FAIL" in report.summary()


def test_validator_accepts_reviewed_allowlist_entries(built):
    tex = _paper(built.paper, "allowed", [r"Jigsaw solves 95.4\% of 1{,}800 queries; see Algorithm 1, the 95\% CI",
                                          r"and Smith et al. 2020."])
    first = vpn.validate(tex, built.out, ALLOWLIST)
    assert sorted(o.numeral for o in first.unmatched()) == ["1{,}800", "95.4"]
    count = next(o for o in first.unmatched() if o.numeral == "1{,}800")
    allow = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
    allow["entries"] += [{"context": count.context, "numeral": "1{,}800", "reason": "test: reviewed context"},
                         {"pattern": r"solves 95\.4\\%", "reason": "test: reviewed pattern"}]
    report = vpn.validate(tex, built.out, allow)
    assert report.ok, report.summary()
    reasons = {o.numeral: o.allowed_by for o in report.occurrences}
    assert reasons["1{,}800"] == "test: reviewed context" and reasons["95.4"] == "test: reviewed pattern"
    assert reasons["2020"].startswith("citation year") and reasons["1"].startswith("numbered reference")


def test_validator_ignores_numerals_in_masked_text(built):
    preamble = PREAMBLE[:-2] + [r"\setcounter{secnumdepth}{3}", r"\newcommand{\budget}{16}"] + PREAMBLE[-2:]
    body = [
        r"Cited \cite{smith2020}, \citep[p.~12]{ref2019}, \citet{a1b2}; \ref{sec:3}, \eqref{eq:4}, \autoref{fig:5}.",
        r"\cref{tab:6} \label{sec:7} \url{https://example.org/v8} \href{https://example.org/9}{a link} % 10 here",
        r"\vspace{-2pt}\hspace*{3mm}\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabular}{p{2cm}r@{\hspace{5pt}}l}\toprule \multicolumn{2}{c}{Head} & x \\[2pt] \cmidrule(lr){1-2}",
        r"a & b & c \\ \bottomrule\end{tabular}",
        r"\includegraphics[width=0.5\linewidth]{generated/figures/F.bars.pdf} \verb|x=11|",
        r"\begin{minipage}[t]{0.48\textwidth}text\end{minipage} \\*[3pt] \bibliography{refs2026}",
    ]
    report = vpn.validate(_paper(built.paper, "masked", body, preamble), built.out, [])
    assert report.occurrences == [], [(o.numeral, o.context) for o in report.occurrences]
    assert report.ok, report.summary()


def test_validator_follows_input_and_include(built):
    sections = built.paper / "sections"
    sections.mkdir(exist_ok=True)
    (sections / "intro.tex").write_text(rf"Intro uses \jnum{{{DIFF}}}." + "\n" + r"\input{sections/deeper}" + "\n",
                                        encoding="utf-8")
    (sections / "deeper.tex").write_text("A hand-typed 42 here.\n", encoding="utf-8")
    (sections / "appendix.tex").write_text("\n" + r"Undefined \jnum{E9.appendix.key} and 7 more." + "\n",
                                           encoding="utf-8")
    tex = _paper(built.paper, "withinputs", [r"\input{sections/intro}", r"\include{sections/appendix}"])
    report = vpn.validate(tex, built.out, ALLOWLIST)
    assert sorted((Path(o.file).name, o.line, o.numeral) for o in report.unmatched()) == [
        ("appendix.tex", 2, "7"), ("deeper.tex", 1, "42")]
    assert [(Path(f.file).name, f.line) for f in report.errors("keys")] == [("appendix.tex", 2)]
    assert DIFF in {use.key for use in report.key_uses}
    assert {Path(name).name for name in report.files} == {"withinputs.tex", "intro.tex", "deeper.tex", "appendix.tex"}


def test_validator_checks_generated_references(built):
    (built.paper / "logo.png").write_bytes(b"\x89PNG\r\n")
    shutil.copy(built.out / "figures" / "F.bars.pdf", built.paper / "stray.pdf")
    body = [r"\input{generated/tables/T.main.tex}",
            r"\input{generated/tables/T.gone.tex}",
            r"\includegraphics{generated/figures/F.solved}",
            r"\includegraphics{generated/figures/F.gone.pdf}",
            r"\includegraphics{logo.png}",
            r"\includegraphics{missing.png}",
            r"\input{sections_none/missing}",
            r"\includegraphics{stray.pdf}"]
    report = vpn.validate(_paper(built.paper, "references", body), built.out, ALLOWLIST)
    line = FIRST_BODY_LINE
    errors = sorted((f.line, f.message) for f in report.errors("build"))
    assert [n for n, _ in errors] == [line + 1, line + 2, line + 3, line + 5, line + 6, line + 7]
    assert "generated/tables/T.gone.tex is missing from the build" in errors[0][1]
    assert "write the extension" in errors[1][1]
    assert "generated/figures/F.gone.pdf is missing from the build" in errors[2][1]
    assert errors[3][1] == "graphic missing.png not found"
    assert errors[4][1] == r"\input{sections_none/missing}: file not found"
    assert "generated by build_paper outside" in errors[5][1]
    infos = [(f.line, f.message) for f in report.infos("build")]
    assert infos == [(line + 4, "graphic logo.png is not generated from numbers.json")]


def test_validator_detects_a_stale_build(tmp_path):
    def numbers(value):
        n = _numbers()
        n.number(KEY, value, "pct", ci=(90.0, 97.0), n=100)
        n.table("T.small", ["Method", "Solved"], [["Jigsaw", {"key": KEY}]])
        return n.write(tmp_path / "in")

    path = numbers(95.4)
    paper = tmp_path / "paper"
    out = paper / "generated"
    assert bp.main(["--numbers", str(path), "--out", str(out)]) == bp.EXIT_OK
    tex = _paper(paper, "main", [rf"Solved \jnum{{{KEY}}}.", r"\input{generated/tables/T.small.tex}"])
    assert vpn.validate(tex, out, ALLOWLIST).ok
    numbers(95.5)  # the reducer ran again; the paper was not rebuilt
    messages = [f.message for f in vpn.validate(tex, out, ALLOWLIST).errors("build")]
    assert messages == ["stale: ../../in/numbers.json changed since the build; rebuild the paper"]
    with (out / "tables" / "T.small.tex").open("a", encoding="utf-8") as handle:
        handle.write("% edited by hand\n")
    (out / "tables" / "T.old.tex").write_text("% left over\n", encoding="utf-8")
    old = _paper(paper, "old", [r"\input{generated/tables/T.old.tex}"])
    report = vpn.validate(old, out, ALLOWLIST)
    messages = [f.message for f in report.errors("build")]
    assert "generated file tables/T.small.tex differs from build.json (edited by hand, or a partial build)" in messages
    assert "tables/T.old.tex was not produced by the current build" in messages
    assert [f.message for f in report.infos("build")] == ["not produced by the current build"]
    empty = vpn.validate(tex, tmp_path / "empty", ALLOWLIST)
    assert empty.errors("build") and empty.errors("keys") and not empty.ok


def test_validator_rejects_dev_builds_unless_allowed(tmp_path):
    n = _numbers("dev", False)
    n.number(KEY, 95.4, "pct")
    path = n.write(tmp_path / "in")
    out = tmp_path / "paper" / "generated"
    assert bp.main(["--numbers", str(path), "--out", str(out), "--allow-dev"]) == bp.EXIT_OK
    tex = _paper(tmp_path / "paper", "main", [rf"Solved \jnum{{{KEY}}}."])
    assert [f.check for f in vpn.validate(tex, out, ALLOWLIST).errors()] == ["dev"]
    assert vpn.validate(tex, out, ALLOWLIST, allow_dev=True).ok
    args = ["--tex", str(tex), "--generated", str(out), "--allowlist", str(ALLOWLIST)]
    assert vpn.main(args) == 1 and vpn.main(args + ["--allow-dev"]) == 0


def test_validator_requires_the_paper_to_input_numbers_tex(built):
    preamble = [line for line in PREAMBLE if "numbers" not in line]
    report = vpn.validate(_paper(built.paper, "noinput", [rf"Solved \jnum{{{KEY}}}."], preamble), built.out,
                          ALLOWLIST)
    assert [f.message for f in report.errors()] == [
        rf"the paper uses \jnum/\jci/\jn but never inputs {report.generated}/numbers.tex"]


def test_draft_allowlist_entries_never_count(built, tmp_path, capsys):
    tex = _paper(built.paper, "draft", [r"Jigsaw solves 95.4\% of 1{,}800 queries."])
    draft = tmp_path / "draft.json"
    args = ["--tex", str(tex), "--generated", str(built.out), "--allowlist", str(ALLOWLIST)]
    assert vpn.main(args + ["--write-allowlist-draft", str(draft)]) == 1
    assert "wrote 2 UNREVIEWED entries" in capsys.readouterr().out
    data = json.loads(draft.read_text(encoding="utf-8"))
    assert data["schema"] == vpn.ALLOWLIST_SCHEMA
    assert sorted((e["numeral"], e["reason"]) for e in data["entries"]) == [("1{,}800", "UNREVIEWED"),
                                                                           ("95.4", "UNREVIEWED")]
    report = vpn.validate(tex, built.out, data)
    assert len(report.unmatched()) == 2 and report.unreviewed_entries == 2
    data["entries"][0]["reason"] = "test: reviewed"
    assert len(vpn.validate(tex, built.out, data).unmatched()) == 1
    own = tmp_path / "allow.json"
    shutil.copy(ALLOWLIST, own)
    before = own.read_bytes()
    own_args = ["--tex", str(tex), "--generated", str(built.out), "--allowlist", str(own)]
    assert vpn.main(own_args + ["--write-allowlist-draft", str(own)]) == 1
    assert "must not overwrite" in capsys.readouterr().err and own.read_bytes() == before


@pytest.mark.parametrize("entry, message", [
    ({"pattern": r"[0-9]+", "reason": "x"}, "bare numeral"),
    ({"pattern": r"\d+(?:[.,]\d+)*", "reason": "x"}, "bare numeral"),
    ({"pattern": r"(?:Table )?[0-9]+", "reason": "x"}, "bare numeral"),
    ({"pattern": r"Table ([0-9]+", "reason": "x"}, "bad pattern"),
    ({"pattern": r"Table [0-9]+", "reason": ""}, "non-empty reason"),
    ({"pattern": r"Table [0-9]+"}, "non-empty reason"),
    ({"context": "Table 2 shows", "numeral": "3", "reason": "x"}, "inside its context"),
    ({"context": "Table 2 shows", "numeral": "two", "reason": "x"}, "inside its context"),
    ({"context": "Table 2", "numeral": "2", "reason": "x", "extra": 1}, "expected"),
])
def test_allowlist_rules(entry, message):
    with pytest.raises(vpn.AllowlistError, match=message):
        vpn.load_allowlist([entry])


def test_repo_allowlist_is_reviewed(tmp_path):
    entries = vpn.load_allowlist(ALLOWLIST)
    assert len(entries) >= 10 and all(entry.reviewed and entry.pattern for entry in entries)
    with pytest.raises(vpn.AllowlistError, match="schema"):
        vpn.load_allowlist({"schema": "other", "entries": []})
    with pytest.raises(vpn.AllowlistError, match="unknown fields"):
        vpn.load_allowlist({"entries": [], "extra": 1})
    with pytest.raises(vpn.AllowlistError, match="cannot read"):
        vpn.load_allowlist(tmp_path / "missing.json")


def test_contexts_survive_unrelated_edits(built):
    v1 = _paper(built.paper, "ctx1", [r"Jigsaw solves 95.4\% of all queries."])
    v2 = _paper(built.paper, "ctx2", ["An unrelated new paragraph.", "", rf"Another one with \jnum{{{KEY}}}.", "",
                                      "Jigsaw   solves", r"95.4\%   of all queries.  % a new comment"])
    first = vpn.validate(v1, built.out, ALLOWLIST).unmatched()
    second = vpn.validate(v2, built.out, ALLOWLIST).unmatched()
    assert [o.context for o in first] == [o.context for o in second] == [r"Jigsaw solves 95.4\% of all queries."]
    assert first[0].line != second[0].line
    assert vpn.validate(v2, built.out, [{"context": first[0].context, "numeral": "95.4", "reason": "test"}]).ok


# ---------------------------------------------------------------------- pdflatex
def _pdflatex_command() -> list[str] | None:
    exe = shutil.which("pdflatex")
    if exe is None:
        return None
    try:
        version = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60,
                                 stdin=subprocess.DEVNULL).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    # MiKTeX would otherwise try to download missing packages.
    return [exe, *(["-disable-installer"] if "MiKTeX" in version else []), "-interaction=nonstopmode",
            "-halt-on-error"]


@pytest.fixture(scope="module")
def pdflatex():
    command = _pdflatex_command()
    if command is None:
        pytest.skip("pdflatex is not installed")
    return command


def _compile(command: list[str], folder: Path, name: str, body: list[str]) -> tuple[int, str]:
    _paper(folder, name, body)
    proc = subprocess.run(command + [f"{name}.tex"], cwd=folder, capture_output=True, encoding="utf-8",
                          errors="replace", timeout=240, stdin=subprocess.DEVNULL)
    log = folder / f"{name}.log"
    # TeX wraps log lines at 79 characters; joining them finds keys that were split.
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else proc.stdout
    return proc.returncode, "".join(text.splitlines())


def test_pdflatex_compiles_every_macro(built, pdflatex):
    body = [r"\jigsawdevbuild"]
    for key in sorted(EXPECTED_VALUES):
        body.append(rf"Value \jnum{{{key}}}.")
        if key != "E1.arxiv.router":
            body.append(rf"Math $x = \jnum{{{key}}}$.")
    body += [rf"Interval \jci{{{key}}} and $\jci{{{key}}}$." for key in sorted(EXPECTED_INTERVALS)]
    body += [rf"Count \jn{{{KEY}}}, $\jn{{{FA}}}$.",
             rf"\section{{Solved \jnum{{{KEY}}}}}",
             rf"Footnote\footnote{{Difference \jnum{{{DIFF}}}.}}",
             r"\begin{table}[h]\centering\input{generated/tables/T.main.tex}\end{table}",
             r"\begin{figure}[h]\centering\includegraphics{generated/figures/F.solved.pdf}",
             r"\includegraphics{generated/figures/F.bars.pdf}\end{figure}"]
    code, log = _compile(pdflatex, built.paper, "compile_ok", body)
    assert code == 0, log[-2000:]
    assert (built.paper / "compile_ok.pdf").stat().st_size > 1000
    assert "Undefined control sequence" not in log and "jigsaw Error" not in log
    assert vpn.validate(built.paper / "compile_ok.tex", built.out, ALLOWLIST).ok


@pytest.mark.parametrize("macro, key", [("jnum", "E9.no-such_key.x"), ("jci", "E1.arxiv.queries")])
def test_pdflatex_stops_on_an_undefined_key(built, pdflatex, macro, key):
    code, log = _compile(pdflatex, built.paper, f"compile_bad_{macro}", [rf"Value \{macro}{{{key}}}."])
    assert code != 0
    assert "Package jigsaw Error" in log and rf"\{macro}{{{key}}}: no such entry in numbers.tex" in log


def test_pdflatex_dev_numbers_compile_with_a_warning(tmp_path, pdflatex):
    n = _numbers("dev", False)
    n.number(KEY, 95.4, "pct")
    path = n.write(tmp_path / "in")
    paper = tmp_path / "paper"
    assert bp.main(["--numbers", str(path), "--out", str(paper / "generated"), "--allow-dev"]) == bp.EXIT_OK
    code, log = _compile(pdflatex, paper, "dev", [rf"\jigsawdevbuild{{}} Solved \jnum{{{KEY}}}."])
    assert code == 0, log[-2000:]
    assert "Package jigsaw Warning: numbers.tex holds development numbers" in log
