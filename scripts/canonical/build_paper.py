r"""Paper numbers, tables and figures generated from numbers.json.

    python -m scripts.canonical.build_paper --numbers FILE [FILE ...] --out DIR [--check] [--allow-dev]

Every result the paper prints is \jnum{KEY} (a value), \jci{KEY} (its interval),
\jn{KEY} (its n), a generated table or a generated figure. This module turns the
numbers.json files written by the reducers (numbers_json.py) into LaTeX:

  DIR/numbers.tex       a header with the sha256 of every input and of the merged
                        content, one \def per key through \csname jnum@KEY\endcsname
                        (jci@KEY for the interval, jn@KEY for n), then the macros
                        \jnum, \jci, \jn and \jigsawdevbuild. An unknown key, or \jci
                        or \jn of a key without an interval or n, stops LaTeX with a
                        \PackageError that names the key.
  DIR/tables/<ID>.tex   booktabs tabulars; a number cell is \jnum{KEY}, followed by
                        \jci{KEY} for {"ci": true}; l for label and r for number columns.
  DIR/figures/<ID>.pdf  matplotlib figures, 3.33 in wide (7 in above 6 series or
                        groups), 8 pt, Type 42 fonts, one fixed colour-blind-safe style
                        per series name.
  DIR/build.json        mode, inputs and figure CSVs with sha256, keys, tables, figures,
                        generated files with sha256 and the stale files removed.

Formatting. format_value() is the only rule; values, table cells and intervals all
go through it. The value is Decimal(repr(value)) rounded ROUND_HALF_UP to the
entry's "digits" or the unit default, never binary float rounding. A value that
rounds to zero prints without a sign, negatives take the math minus \ensuremath{-}
(never a hyphen), and integer parts take {,} thousands separators, which typeset
as a comma in text and in math.
  pct     95.4\%                     1 decimal
  pp      0.3\,pp                    1 (signed percentage points)
  frac    0.954                      3
  number  1.25                       2 (any signed real)
  count   1{,}800                    integers only; digits do not apply
  seconds 2.74\,s   ms 12.3\,ms      2 and 1
  bytes   2.43\,GiB                  2; the largest of B, KiB, MiB, GiB, TiB whose
                                     value is >= 1 after rounding; B prints whole
  gb      1.30\,GB                   2 (decimal gigabytes)
  ratio   5.4\ensuremath{\times}     1
  pvalue  \ensuremath{p = 0.003}     at most `digits` (2) significant digits with
                                     trailing zeros dropped; below
                                     10^PVALUE_FLOOR_EXPONENT: \ensuremath{p < 10^{-4}}
  text    LaTeX-escaped
An interval is "[lo, hi]": each bound by the value's rule without the unit (bytes
keep the value's scale). NaN or infinity is an error naming the key.

Rules:
- The paper is built only from frozen test numbers. An input whose split is not
  "test", whose frozen flag is false, or which claims frozen under a protocol_hash
  other than protocol.FROZEN_PROTOCOL_HASH is refused (exit 2, nothing written)
  unless --allow-dev (alias --draft). Such a draft is marked dev everywhere:
  \jigsawdevbuild prints [DEV NUMBERS], every page carries a DEV NUMBERS banner,
  LaTeX warns on every run, the table headers and build.json say dev, and every
  figure is stamped.
- Inputs merge: a key may repeat only with an identical entry, a table or figure id
  appears once, all inputs share one protocol_hash, a run id has one description.
- Everything is rendered in memory before anything is written; writes are atomic
  (temp file, then replace) and build.json is written last. Files in DIR/tables
  and DIR/figures that the build did not produce are removed and listed in
  build.json.
- --check renders in memory and compares every file with DIR byte for byte; it
  writes nothing and exits 1 listing stale, missing and extra files.
- Outputs carry no timestamps and no absolute paths (inputs are named relative to
  DIR); figure bytes depend only on the data and the matplotlib version, which
  build.json records.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, localcontext
from importlib import metadata
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Sequence

from . import numbers_json as nj
from . import protocol

BUILD_SCHEMA = "jigsaw-paper-build-v1"
DEFAULT_DIGITS = {"pct": 1, "pp": 1, "frac": 3, "number": 2, "count": 0, "seconds": 2, "ms": 1, "bytes": 2,
                  "gb": 2, "ratio": 1, "pvalue": 2}
UNIT_SUFFIX = {"pct": r"\%", "pp": r"\,pp", "seconds": r"\,s", "ms": r"\,ms", "gb": r"\,GB",
               "ratio": r"\ensuremath{\times}"}
BINARY_UNITS = ("B", "KiB", "MiB", "GiB", "TiB")
PVALUE_FLOOR_EXPONENT = -4
MATH_MINUS = r"\ensuremath{-}"
THOUSANDS_SEPARATOR = "{,}"
DEV_MARKER = r"\textbf{[DEV NUMBERS]}"
DEV_BANNER = "DEV NUMBERS: NOT FOR SUBMISSION"
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2

# Okabe-Ito colours (colour-blind safe); yellow last because it is faint on white.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#000000", "#999999", "#F0E442")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
LINESTYLES = ("-", "--", "-.", ":")
HATCHES = ("", "////", "....", "xxxx")
# The protocol's methods and serving modes keep one style in every figure; other
# series names take the following slots in sorted order.
FIXED_SERIES = ("jigsaw", "filterall", "direct", "jigsaw_learned", "jigsaw_index", "jigsaw_fused", "mean_rrf",
                "mean_feature", "topo_feature", "random", "classical_gql", "classical_cfl", "classical_dpiso",
                "gnnpe", "resident", "streaming")
COLUMN_WIDTH_IN, FULL_WIDTH_IN = 3.33, 7.0
COLUMN_HEIGHT_IN, FULL_HEIGHT_IN = 2.2, 2.6
MAX_COLUMN_SERIES = 6
PDF_METADATA = {"Creator": "scripts/canonical/build_paper.py", "Producer": None, "CreationDate": None,
                "ModDate": None}
FIGURE_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.compression": 6,
    "svg.hashsalt": "jigsaw-build-paper",
    "figure.dpi": 100,
    "savefig.dpi": 300,
}

_TOP_FIELDS = {"schema", "protocol_hash", "frozen", "split", "generated_utc", "sources", "numbers", "tables",
               "figures"}
_SOURCE_FIELDS = {"path", "manifest_sha256", "experiment", "dataset"}
_NUMBER_FIELDS = {"value", "unit", "digits", "ci", "n", "source", "note"}
_TABLE_FIELDS = {"columns", "rows", "notes"}
_FIGURE_FIELDS = {"type", "csv", "x", "y", "log_x", "notes"}
_LATEX_SPECIALS = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
                   "%": r"\%", "_": r"\_", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                   "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}", "\n": " ", "\r": " ",
                   "\t": " "}


class BuildError(ValueError):
    """An input that cannot become LaTeX; nothing is written."""


class RefusedError(BuildError):
    """Development numbers without --allow-dev; nothing is written."""


# ---------------------------------------------------------------------- formatting
def latex_escape(text: str) -> str:
    """Text for LaTeX text mode: specials escaped, line breaks as spaces."""
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in text)


def _decimal(value: Any, key: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BuildError(f"{key}: {value!r} is not a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise BuildError(f"{key}: {value!r} is not finite")
    return Decimal(repr(value))


def _round(d: Decimal, digits: int) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = max(28, d.adjusted() + digits + 3)
        q = d.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return q.copy_abs() if q.is_zero() else q


def _fixed(q: Decimal) -> str:
    """A rounded Decimal as fixed point with {,} separators and the math minus."""
    text = format(q, "f")
    sign = ""
    if text.startswith("-"):
        sign, text = MATH_MINUS, text[1:]
    whole, dot, frac = text.partition(".")
    groups = []
    while len(whole) > 3:
        groups.insert(0, whole[-3:])
        whole = whole[:-3]
    groups.insert(0, whole)
    return sign + THOUSANDS_SEPARATOR.join(groups) + dot + frac


def _check_digits(digits: Any, key: str) -> None:
    if isinstance(digits, bool) or not isinstance(digits, int) or not 0 <= digits <= 6:
        raise BuildError(f"{key}: digits must be an integer in 0..6, got {digits!r}")


def _scaled_bytes(d: Decimal, index: int) -> Decimal:
    # Division by 2**(10 * index) terminates, so this precision keeps it exact.
    with localcontext() as ctx:
        ctx.prec = len(d.as_tuple().digits) + 12 * index + 12
        return d / (Decimal(1024) ** index)


def byte_scale(value: Any, digits: int | None = None, *, key: str = "value") -> int:
    """Index into BINARY_UNITS of the largest unit in which the rounded value is >= 1."""
    d = _decimal(value, key)
    digits = DEFAULT_DIGITS["bytes"] if digits is None else digits
    index = 0
    while index < len(BINARY_UNITS) - 1 and d >= Decimal(1024) ** (index + 1):
        index += 1
    while index < len(BINARY_UNITS) - 1 and _round(_scaled_bytes(d, index), digits if index else 0) >= 1024:
        index += 1
    return index


def _pvalue_text(d: Decimal, digits: int) -> tuple[str, bool]:
    """(text, below_floor): at most `digits` significant digits, trailing zeros dropped."""
    if d < Decimal(1).scaleb(PVALUE_FLOOR_EXPONENT):
        return f"10^{{{PVALUE_FLOOR_EXPONENT}}}", True
    with localcontext() as ctx:
        ctx.prec = 50
        q = d.quantize(Decimal(1).scaleb(d.adjusted() - max(1, digits) + 1), rounding=ROUND_HALF_UP)
    text = format(q, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text, False


def format_value(value: Any, unit: str, digits: int | None = None, *, key: str = "value", suffix: bool = True,
                 scale: int | None = None) -> str:
    """The LaTeX text of one value: the single formatting rule for numbers, table cells and intervals.

    suffix=False drops the unit (interval bounds); scale fixes the binary unit of a bytes value.
    """
    if unit not in nj.UNITS:
        raise BuildError(f"{key}: unknown unit {unit!r}")
    if unit == "text":
        if not isinstance(value, str):
            raise BuildError(f"{key}: a text value must be a string, got {value!r}")
        return latex_escape(value)
    d = _decimal(value, key)
    if digits is None:
        digits = DEFAULT_DIGITS[unit]
    _check_digits(digits, key)
    if unit == "count":
        if d != d.to_integral_value():
            raise BuildError(f"{key}: a count must be an integer, got {value!r}")
        return _fixed(_round(d, 0))
    if unit == "pvalue":
        text, below = _pvalue_text(d, digits)
        if not suffix:
            return rf"\ensuremath{{<{text}}}" if below else text
        return rf"\ensuremath{{p {'<' if below else '='} {text}}}"
    if unit == "bytes":
        index = byte_scale(value, digits, key=key) if scale is None else scale
        body = _fixed(_round(_scaled_bytes(d, index), digits if index else 0))
        return body + (rf"\,{BINARY_UNITS[index]}" if suffix else "")
    return _fixed(_round(d, digits)) + (UNIT_SUFFIX.get(unit, "") if suffix else "")


def format_interval(lo: Any, hi: Any, unit: str, digits: int | None = None, *, key: str = "value",
                    center: Any = None) -> str:
    """"[lo, hi]" with the value's rule and no unit; bytes use the scale of `center` (the value)."""
    scale = None
    if unit == "bytes":
        scale = byte_scale(hi if center is None else center, digits, key=key)
    bounds = [format_value(v, unit, digits, key=key, suffix=False, scale=scale) for v in (lo, hi)]
    return f"[{bounds[0]}, {bounds[1]}]"


def format_count(n: Any, *, key: str = "value") -> str:
    return format_value(n, "count", key=key)


# ---------------------------------------------------------------------- series styles
@dataclass(frozen=True)
class SeriesStyle:
    colour: str
    marker: str
    linestyle: str
    hatch: str


def series_key(name: str) -> str:
    """Names that differ only in case or punctuation are one series (Mean-RRF == mean_rrf)."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def assign_series_slots(names: Iterable[str]) -> dict[str, int]:
    """Style slot per name: FIXED_SERIES first, then the other names in sorted order."""
    fixed = {series_key(name): slot for slot, name in enumerate(FIXED_SERIES)}
    names = sorted(set(names))
    others = sorted({series_key(name) for name in names} - set(fixed))
    slot_of = dict(fixed)
    slot_of.update({key: len(FIXED_SERIES) + i for i, key in enumerate(others)})
    return {name: slot_of[series_key(name)] for name in names}


def series_style(slot: int) -> SeriesStyle:
    cycle = slot // len(PALETTE)
    return SeriesStyle(colour=PALETTE[slot % len(PALETTE)],
                       marker=MARKERS[(slot + cycle // len(LINESTYLES)) % len(MARKERS)],
                       linestyle=LINESTYLES[cycle % len(LINESTYLES)],
                       hatch=HATCHES[cycle % len(HATCHES)])


# ---------------------------------------------------------------------- inputs
@dataclass
class Input:
    """One validated numbers.json with its figure rows."""

    path: Path
    sha256: str
    data: dict[str, Any]
    rows: dict[str, list[dict[str, Any]]]
    csvs: dict[str, tuple[Path, str]]  # figure id -> (csv path, sha256)


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def _check_key(value: Any, what: str) -> None:
    try:
        nj.check_key(value, what)
    except nj.NumbersError as exc:
        raise BuildError(str(exc)) from None


def _checked(key: str, value: Any, unit: str) -> Any:
    try:
        return nj.check_value(key, value, unit)
    except nj.NumbersError as exc:
        raise BuildError(str(exc)) from None


def _check_entry(key: str, entry: Any, sources: Mapping[str, Any]) -> None:
    _check_key(key, "key")
    _need(isinstance(entry, dict), f"{key}: an entry must be an object")
    extra = set(entry) - _NUMBER_FIELDS
    _need(not extra, f"{key}: unknown fields {sorted(extra)}")
    _need("value" in entry and "unit" in entry, f"{key}: value and unit are required")
    unit = entry["unit"]
    _checked(key, entry["value"], unit)
    if entry.get("digits") is not None:
        _check_digits(entry["digits"], key)
    ci = entry.get("ci")
    if ci is not None:
        _need(isinstance(ci, list) and len(ci) == 2, f"{key}: ci must be [lo, hi] or null")
        lo, hi = (_checked(key, v, unit) for v in ci)
        _need(unit == "text" or lo <= hi + 1e-9, f"{key}: interval [{lo}, {hi}] is reversed")
    n = entry.get("n")
    _need(n is None or (type(n) is int and n >= 0), f"{key}: n must be a non-negative integer or null")
    source = entry.get("source", [])
    _need(isinstance(source, list) and all(isinstance(s, str) for s in source), f"{key}: source must be a list")
    unknown = [s for s in source if s not in sources]
    _need(not unknown, f"{key}: unknown sources {unknown}")
    _need(entry.get("note") is None or isinstance(entry["note"], str), f"{key}: note must be a string")


def _check_table(table_id: str, table: Any, numbers: Mapping[str, Any]) -> None:
    _check_key(table_id, "table id")
    _need(isinstance(table, dict) and set(table) <= _TABLE_FIELDS and {"columns", "rows"} <= set(table),
          f"table {table_id}: expected fields columns, rows and optional notes")
    columns = table["columns"]
    _need(isinstance(columns, list) and bool(columns) and all(isinstance(c, str) for c in columns),
          f"table {table_id}: columns must be a non-empty list of strings")
    _need(isinstance(table["rows"], list), f"table {table_id}: rows must be a list")
    for number, row in enumerate(table["rows"], start=1):
        _need(isinstance(row, list) and len(row) == len(columns),
              f"table {table_id}: row {number} does not have {len(columns)} cells")
        for cell in row:
            if isinstance(cell, str):
                continue
            _need(isinstance(cell, dict) and "key" in cell and set(cell) <= {"key", "ci"},
                  f'table {table_id}: a cell must be a string or {{"key": KEY}}, got {cell!r}')
            _need(cell["key"] in numbers, f"table {table_id}: unknown key {cell['key']!r}")
            _need(cell.get("ci") in (None, True, False), f"table {table_id}: ci must be true or false")
            _need(not cell.get("ci") or numbers[cell["key"]].get("ci") is not None,
                  f"table {table_id}: {cell['key']} has no interval")
    _need(table.get("notes") is None or isinstance(table["notes"], str), f"table {table_id}: notes must be text")


def _check_figure_spec(fig_id: str, fig: Any) -> None:
    _check_key(fig_id, "figure id")
    _need(isinstance(fig, dict) and set(fig) <= _FIGURE_FIELDS and {"type", "csv", "x", "y"} <= set(fig),
          f"figure {fig_id}: expected fields type, csv, x, y and optional log_x, notes")
    _need(fig["type"] in nj.FIGURE_COLUMNS, f"figure {fig_id}: unknown type {fig['type']!r}")
    _need(isinstance(fig["csv"], str) and bool(fig["csv"]) and not Path(fig["csv"]).is_absolute(),
          f"figure {fig_id}: csv must be a path relative to numbers.json")
    _need(isinstance(fig["x"], str) and isinstance(fig["y"], str), f"figure {fig_id}: x and y must be labels")
    _need(isinstance(fig.get("log_x", False), bool), f"figure {fig_id}: log_x must be true or false")
    _need(fig.get("notes") is None or isinstance(fig["notes"], str), f"figure {fig_id}: notes must be text")


def check_document(data: Any) -> dict[str, Any]:
    """The numbers.json checks beyond numbers_json.load(); returns the document with defaults filled."""
    _need(isinstance(data, dict), "not a JSON object")
    extra = set(data) - _TOP_FIELDS
    _need(not extra, f"unknown fields {sorted(extra)}")
    _need(data.get("schema") == nj.SCHEMA, f"schema {data.get('schema')!r}, expected {nj.SCHEMA}")
    _need(isinstance(data.get("protocol_hash"), str) and bool(data["protocol_hash"]),
          "protocol_hash must be a non-empty string")
    _need(isinstance(data.get("frozen"), bool), "frozen must be true or false")
    _need(isinstance(data.get("split"), str) and bool(data["split"]), "split must be a non-empty string")
    for name in ("sources", "numbers", "tables", "figures"):
        data[name] = data.get(name) or {}
        _need(isinstance(data[name], dict), f"{name} must be an object")
    for run_id, src in data["sources"].items():
        _check_key(run_id, "run id")
        _need(isinstance(src, dict) and set(src) <= _SOURCE_FIELDS and {"path", "experiment", "dataset"} <= set(src),
              f"source {run_id}: expected fields {sorted(_SOURCE_FIELDS)}")
        _need(src.get("manifest_sha256") is None or isinstance(src["manifest_sha256"], str),
              f"source {run_id}: manifest_sha256 must be a string or null")
    for key, entry in data["numbers"].items():
        _check_entry(key, entry, data["sources"])
    for table_id, table in data["tables"].items():
        _check_table(table_id, table, data["numbers"])
    for fig_id, fig in data["figures"].items():
        _check_figure_spec(fig_id, fig)
    return data


def parse_figure_csv(fig_id: str, spec: Mapping[str, Any], raw: bytes) -> list[dict[str, Any]]:
    """Rows of a figure CSV (header series,x,y,lo,hi or group,series,y,lo,hi; lo and hi may be empty)."""
    columns = nj.FIGURE_COLUMNS[spec["type"]]
    try:
        records = list(csv.reader(io.StringIO(raw.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise BuildError(f"figure {fig_id}: unreadable csv ({exc})") from None
    _need(bool(records) and tuple(records[0]) == columns,
          f"figure {fig_id}: csv header must be {','.join(columns)}")
    rows = []
    for line, record in enumerate(records[1:], start=2):
        if not record:
            continue
        _need(len(record) == len(columns), f"figure {fig_id}: csv line {line} has {len(record)} fields")
        row: dict[str, Any] = {}
        for column, cell in zip(columns, record):
            if column in ("series", "group"):
                row[column] = cell
            elif cell == "" and column in ("lo", "hi"):
                row[column] = None
            else:
                try:
                    value = float(cell)
                except ValueError:
                    raise BuildError(f"figure {fig_id}: csv line {line}: {column} {cell!r} is not a number") from None
                _need(math.isfinite(value), f"figure {fig_id}: csv line {line}: {column} is not finite")
                row[column] = value
        rows.append(row)
    _need(bool(rows), f"figure {fig_id} has no rows")
    if spec.get("log_x"):
        _need(all(row["x"] > 0 for row in rows), f"figure {fig_id}: a log x-axis needs x > 0")
    return rows


def load_input(path: str | os.PathLike) -> Input:
    """Read, validate and hash one numbers.json and its figure CSVs."""
    path = Path(path)
    try:
        raw = path.read_bytes()
        data = nj.load(path)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise BuildError(f"{path}: {exc}") from None
    try:
        check_document(data)
        rows: dict[str, list[dict[str, Any]]] = {}
        csvs: dict[str, tuple[Path, str]] = {}
        for fig_id, spec in sorted(data["figures"].items()):
            csv_path = path.parent / spec["csv"]
            try:
                csv_raw = csv_path.read_bytes()
            except OSError as exc:
                raise BuildError(f"figure {fig_id}: {exc}") from None
            rows[fig_id] = parse_figure_csv(fig_id, spec, csv_raw)
            csvs[fig_id] = (csv_path, hashlib.sha256(csv_raw).hexdigest())
    except BuildError as exc:
        raise BuildError(f"{path}: {exc}") from None
    return Input(path=path, sha256=hashlib.sha256(raw).hexdigest(), data=data, rows=rows, csvs=csvs)


def dev_reasons(inputs: Sequence[Input]) -> list[tuple[Input, str]]:
    """Why each input is not camera-ready: split, frozen flag, or a frozen claim off the frozen hash."""
    frozen_hash = protocol.FROZEN_PROTOCOL_HASH
    reasons = []
    for item in inputs:
        data = item.data
        if data["split"] != "test":
            reasons.append((item, f"split is {data['split']!r}, not 'test'"))
        if not data["frozen"]:
            reasons.append((item, "frozen is false"))
        elif data["protocol_hash"] != frozen_hash:
            reasons.append((item, f"claims frozen, but protocol_hash {data['protocol_hash'][:12]} is not "
                                  f"protocol.FROZEN_PROTOCOL_HASH ({(frozen_hash or 'unset')[:12]})"))
    return reasons


@dataclass
class Merged:
    protocol_hash: str
    sources: dict[str, dict[str, Any]]
    numbers: dict[str, dict[str, Any]]
    tables: dict[str, dict[str, Any]]
    figures: dict[str, dict[str, Any]]
    rows: dict[str, list[dict[str, Any]]]

    def content(self) -> dict[str, Any]:
        return {"protocol_hash": self.protocol_hash, "sources": self.sources, "numbers": self.numbers,
                "tables": self.tables, "figures": self.figures, "figure_rows": self.rows}


def _normal_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    out = {"value": entry["value"], "unit": entry["unit"], "ci": entry.get("ci"), "n": entry.get("n"),
           "source": sorted(entry.get("source", []))}
    if entry.get("digits") is not None:
        out["digits"] = entry["digits"]
    if entry.get("note"):
        out["note"] = entry["note"]
    return out


def merge_inputs(inputs: Sequence[Input]) -> Merged:
    """One namespace from many numbers.json files; any conflict is an error."""
    _need(bool(inputs), "no numbers.json given")
    hashes = sorted({item.data["protocol_hash"] for item in inputs})
    _need(len(hashes) == 1, f"inputs have different protocol hashes: {', '.join(h[:12] for h in hashes)}")
    merged = Merged(hashes[0], {}, {}, {}, {}, {})
    owner: dict[tuple[str, str], Path] = {}
    for item in inputs:
        for run_id, src in item.data["sources"].items():
            if run_id in merged.sources and merged.sources[run_id] != src:
                raise BuildError(f"source {run_id} differs between {owner['source', run_id]} and {item.path}")
            merged.sources.setdefault(run_id, src)
            owner.setdefault(("source", run_id), item.path)
        for key, entry in item.data["numbers"].items():
            entry = _normal_entry(entry)
            if key in merged.numbers and protocol.canonical_json(merged.numbers[key]) != protocol.canonical_json(entry):
                raise BuildError(f"key {key} has different entries in {owner['key', key]} and {item.path}")
            merged.numbers.setdefault(key, entry)
            owner.setdefault(("key", key), item.path)
        for kind, target, source in (("table", merged.tables, item.data["tables"]),
                                     ("figure", merged.figures, item.data["figures"])):
            for ident, spec in source.items():
                if ident in target:
                    raise BuildError(f"{kind} {ident} is defined in both {owner[kind, ident]} and {item.path}")
                target[ident] = spec
                owner[kind, ident] = item.path
        merged.rows.update(item.rows)
    return merged


# ---------------------------------------------------------------------- rendering
def _comment(text: str) -> str:
    return " ".join(text.split())


def _rel(path: Path, base: Path) -> str:
    """Path relative to DIR with forward slashes (absolute only across Windows drives)."""
    try:
        return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()
    except ValueError:
        return path.resolve().as_posix()


_MACROS = (
    r"% \jnum{KEY}: the value; \jci{KEY}: its interval [lo, hi] without the unit; \jn{KEY}: its n.",
    r"% An undefined key, or \jci/\jn of a key without an interval/n, stops LaTeX with an error naming it.",
    r"\protected\def\jnum#1{\jigsawnumlookup{jnum}{#1}}",
    r"\protected\def\jci#1{\jigsawnumlookup{jci}{#1}}",
    r"\protected\def\jn#1{\jigsawnumlookup{jn}{#1}}",
    r"\def\jigsawnumlookup#1#2{%",
    r"  \ifcsname #1@\detokenize{#2}\endcsname",
    r"    \expandafter\jigsawnumfirst",
    r"  \else",
    r"    \expandafter\jigsawnumsecond",
    r"  \fi",
    r"  {\csname #1@\detokenize{#2}\endcsname}{\jigsawnummissing{#1}{#2}}}",
    r"\long\def\jigsawnumfirst#1#2{#1}",
    r"\long\def\jigsawnumsecond#1#2{#2}",
    r"\def\jigsawnummissing#1#2{%",
    r"  \PackageError{jigsaw}{\expandafter\string\csname #1\endcsname{\detokenize{#2}}: no such entry in"
    r" numbers.tex}%",
    r"    {The key is undefined, or it has no interval (\string\jci) or no n (\string\jn).",
    r"     Fix the key or rebuild numbers.tex with scripts/canonical/build_paper.py.}}",
)
_DEV_LINES = (
    r"\PackageWarningNoLine{jigsaw}{numbers.tex holds development numbers (build_paper --allow-dev);"
    r" not for submission}",
    r"\ifdefined\AddToHook",
    r"  \AddToHook{shipout/foreground}{\put(0,0){\makebox[\paperwidth]{\raisebox{-2\baselineskip}[0pt][0pt]{%",
    r"    \normalfont\bfseries\small " + DEV_BANNER + "}}}}",
    r"\fi",
)


def render_numbers_tex(merged: Merged, inputs: Sequence[Mapping[str, Any]], merged_sha: str, dev: bool) -> str:
    lines = [
        "% numbers.tex: generated by scripts/canonical/build_paper.py from numbers.json; do not edit, rebuild.",
        "% mode: " + ("DEV, development numbers, not for submission" if dev else "frozen (test split, frozen protocol)"),
        f"% protocol_hash: {merged.protocol_hash}",
        f"% merged_sha256: {merged_sha}",
    ]
    lines += [f"% input: {item['path']} sha256 {item['sha256']}" for item in inputs if item["kind"] == "numbers"]
    lines.append(f"% keys: {len(merged.numbers)}")
    for key in sorted(merged.numbers):
        entry = merged.numbers[key]
        unit, digits = entry["unit"], entry.get("digits")
        lines.append(rf"\expandafter\def\csname jnum@{key}\endcsname"
                     f"{{{format_value(entry['value'], unit, digits, key=key)}}}")
        if entry["ci"] is not None:
            interval = format_interval(entry["ci"][0], entry["ci"][1], unit, digits, key=key, center=entry["value"])
            lines.append(rf"\expandafter\def\csname jci@{key}\endcsname{{{interval}}}")
        if entry["n"] is not None:
            lines.append(rf"\expandafter\def\csname jn@{key}\endcsname{{{format_count(entry['n'], key=key)}}}")
    lines += _MACROS
    if dev:
        lines.append(rf"\protected\def\jigsawdevbuild{{{DEV_MARKER}}}")
        lines += _DEV_LINES
    else:
        lines.append(r"\protected\def\jigsawdevbuild{}")
    lines.append(r"\endinput")
    return "\n".join(lines) + "\n"


def render_table(table_id: str, table: Mapping[str, Any], dev: bool) -> str:
    columns, rows = table["columns"], table["rows"]
    numeric = [any(isinstance(row[i], dict) for row in rows) for i in range(len(columns))]
    sha = hashlib.sha256(protocol.canonical_json(table).encode("ascii")).hexdigest()
    lines = [f"% Generated by scripts/canonical/build_paper.py from numbers.json table {table_id}; do not edit.",
             f"% table {table_id} content sha256 {sha}"]
    if dev:
        lines.append("% DEV BUILD: development numbers, not for submission.")
    if table.get("notes"):
        lines.append(f"% notes: {_comment(table['notes'])}")
    lines += [r"\begin{tabular}{" + "".join("r" if flag else "l" for flag in numeric) + "}", r"\toprule",
              " & ".join(latex_escape(column) for column in columns) + r" \\", r"\midrule"]
    for row in rows:
        cells = []
        for cell in row:
            if isinstance(cell, str):
                cells.append(latex_escape(cell))
            else:
                text = rf"\jnum{{{cell['key']}}}"
                if cell.get("ci"):
                    text += rf" \jci{{{cell['key']}}}"
                cells.append(text)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _mpl_text(text: str) -> str:
    return text.replace("$", r"\$")  # a lone $ would start matplotlib mathtext


def render_figure(fig_id: str, spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                  slots: Mapping[str, int], dev: bool) -> bytes:
    """One figure as PDF bytes; identical inputs and matplotlib version give identical bytes."""
    import matplotlib.style
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    series = sorted({row["series"] for row in rows}, key=lambda name: (slots[name], name))
    styles = {name: series_style(slots[name]) for name in series}
    groups = list(dict.fromkeys(row["group"] for row in rows)) if spec["type"] == "bars" else []
    seen: dict[tuple[str, ...], str] = {}
    for name in series:
        style = styles[name]
        look = (style.colour, style.hatch) if spec["type"] == "bars" else (style.colour, style.linestyle,
                                                                              style.marker)
        _need(look not in seen, f"figure {fig_id}: series {seen.get(look)!r} and {name!r} would share a style")
        seen[look] = name
    wide = len(series) > MAX_COLUMN_SERIES or len(groups) > MAX_COLUMN_SERIES
    size = (FULL_WIDTH_IN, FULL_HEIGHT_IN) if wide else (COLUMN_WIDTH_IN, COLUMN_HEIGHT_IN)
    with matplotlib.style.context("default"), rc_context(FIGURE_RC):
        fig = Figure(figsize=size, layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        if spec["type"] == "curve":
            _draw_curves(ax, rows, series, styles)
            if spec.get("log_x"):
                ax.set_xscale("log")
        else:
            _draw_bars(fig_id, ax, rows, series, groups, styles, wide)
        ax.set_xlabel(_mpl_text(spec["x"]))
        ax.set_ylabel(_mpl_text(spec["y"]))
        ax.legend(frameon=False, loc="best", ncols=min(len(series), 3 if wide else 2))
        if dev:
            fig.text(0.995, 0.005, "DEV NUMBERS", ha="right", va="bottom", fontsize=6, fontweight="bold",
                     color=PALETTE[1])
        buffer = io.BytesIO()
        fig.savefig(buffer, format="pdf", metadata=PDF_METADATA)
    return buffer.getvalue()


def _draw_curves(ax: Any, rows: Sequence[Mapping[str, Any]], series: Sequence[str],
                 styles: Mapping[str, SeriesStyle]) -> None:
    for name in series:
        points = sorted((row for row in rows if row["series"] == name), key=lambda row: row["x"])
        xs = [row["x"] for row in points]
        style = styles[name]
        if any(row["lo"] is not None and row["hi"] is not None for row in points):
            ax.fill_between(xs, [row["lo"] if row["hi"] is not None and row["lo"] is not None else math.nan
                                 for row in points],
                            [row["hi"] if row["hi"] is not None and row["lo"] is not None else math.nan
                             for row in points],
                            color=style.colour, alpha=0.18, linewidth=0)
        ax.plot(xs, [row["y"] for row in points], color=style.colour, linestyle=style.linestyle,
                marker=style.marker, label=_mpl_text(name))


def _draw_bars(fig_id: str, ax: Any, rows: Sequence[Mapping[str, Any]], series: Sequence[str],
               groups: Sequence[str], styles: Mapping[str, SeriesStyle], wide: bool) -> None:
    cell: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        _need((row["group"], row["series"]) not in cell,
              f"figure {fig_id}: two bars for group {row['group']!r}, series {row['series']!r}")
        cell[row["group"], row["series"]] = row
    width = 0.8 / len(series)
    for index, name in enumerate(series):
        style = styles[name]
        placed = [(g, cell[group, name]) for g, group in enumerate(groups) if (group, name) in cell]
        xs = [g - 0.4 + width * (index + 0.5) for g, _ in placed]
        ys = [row["y"] for _, row in placed]
        ax.bar(xs, ys, width, color=style.colour, hatch=style.hatch, edgecolor="#333333", linewidth=0.4,
               label=_mpl_text(name))
        errors = [(x, row) for x, (_, row) in zip(xs, placed) if row["lo"] is not None and row["hi"] is not None]
        for _, row in errors:
            _need(row["lo"] <= row["y"] <= row["hi"],
                  f"figure {fig_id}: bar {row['group']!r}/{name!r} lies outside its interval")
        if errors:
            ax.errorbar([x for x, _ in errors], [row["y"] for _, row in errors],
                        yerr=[[row["y"] - row["lo"] for _, row in errors], [row["hi"] - row["y"] for _, row in errors]],
                        fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5)
    labels = [_mpl_text(group) for group in groups]
    crowded = sum(len(label) for label in labels) > (90 if wide else 40)
    ax.set_xticks(range(len(groups)), labels=labels, rotation=30 if crowded else 0, ha="right" if crowded else "center")
    ax.grid(False, axis="x")


# ---------------------------------------------------------------------- build
@dataclass
class Build:
    """A rendered build, still in memory."""

    mode: str  # "frozen" or "dev"
    files: dict[str, bytes]  # path relative to DIR -> bytes, build.json excluded
    record: dict[str, Any]  # build.json without removed_stale
    reasons: list[str]  # why the build is dev; empty when frozen


def _matplotlib_version() -> str | None:
    try:
        return metadata.version("matplotlib")
    except metadata.PackageNotFoundError:
        return None


def build(numbers_paths: Sequence[str | os.PathLike], *, out_dir: str | os.PathLike,
          allow_dev: bool = False) -> Build:
    """Validate, gate, merge and render in memory; raises RefusedError or BuildError, writes nothing."""
    paths = [Path(p) for p in numbers_paths]
    _need(bool(paths), "no numbers.json given")
    seen: set[str] = set()
    for path in paths:
        ident = os.path.normcase(str(path.resolve()))
        _need(ident not in seen, f"{path} is given twice")
        seen.add(ident)
    inputs = [load_input(path) for path in paths]
    reasons = dev_reasons(inputs)
    if reasons and not allow_dev:
        detail = "\n".join(f"  {item.path}: {why}" for item, why in reasons)
        raise RefusedError("the paper is built only from frozen test numbers; pass --allow-dev for a draft "
                           f"marked DEV NUMBERS:\n{detail}")
    merged = merge_inputs(inputs)
    out_dir = Path(out_dir)
    dev = bool(reasons)
    records: list[dict[str, Any]] = []
    for item in inputs:
        records.append({"kind": "numbers", "path": _rel(item.path, out_dir), "sha256": item.sha256,
                        "split": item.data["split"], "frozen": item.data["frozen"]})
        records += [{"kind": "figure_csv", "path": _rel(csv_path, out_dir), "sha256": sha, "figure": fig_id}
                    for fig_id, (csv_path, sha) in sorted(item.csvs.items())]
    records.sort(key=lambda record: (record["path"], record["kind"]))
    merged_sha = hashlib.sha256(protocol.canonical_json(merged.content()).encode("ascii")).hexdigest()
    files = {"numbers.tex": render_numbers_tex(merged, records, merged_sha, dev).encode("utf-8")}
    for table_id in sorted(merged.tables):
        files[f"tables/{table_id}.tex"] = render_table(table_id, merged.tables[table_id], dev).encode("utf-8")
    if merged.figures:
        slots = assign_series_slots(row["series"] for rows in merged.rows.values() for row in rows)
        for fig_id in sorted(merged.figures):
            files[f"figures/{fig_id}.pdf"] = render_figure(fig_id, merged.figures[fig_id], merged.rows[fig_id],
                                                           slots, dev)
    record = {
        "schema": BUILD_SCHEMA,
        "mode": "dev" if dev else "frozen",
        "dev_reasons": sorted(f"{_rel(item.path, out_dir)}: {why}" for item, why in reasons),
        "protocol_hash": merged.protocol_hash,
        "frozen_protocol_hash": protocol.FROZEN_PROTOCOL_HASH,
        "merged_sha256": merged_sha,
        "inputs": records,
        "keys": sorted(merged.numbers),
        "tables": sorted(merged.tables),
        "figures": sorted(merged.figures),
        "generated": {rel: hashlib.sha256(data).hexdigest() for rel, data in sorted(files.items())},
        "renderer": {"matplotlib": _matplotlib_version() if merged.figures else None},
    }
    return Build(mode=record["mode"], files=files, record=record,
                 reasons=[f"{item.path}: {why}" for item, why in reasons])


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    """Write through a temp file in the same folder, then replace: readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def stale_files(out_dir: Path, files: Collection[str]) -> list[str]:
    """Files in DIR/tables and DIR/figures that the build did not produce."""
    stale = []
    for sub in ("tables", "figures"):
        folder = out_dir / sub
        if folder.is_dir():
            stale += [f"{sub}/{p.name}" for p in sorted(folder.iterdir()) if p.is_file() and f"{sub}/{p.name}" not in files]
    return stale


def write_build(result: Build, out_dir: str | os.PathLike) -> list[str]:
    """Write the build into DIR (build.json last) and remove stale files; returns the removed paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = stale_files(out_dir, result.files)
    for rel, data in sorted(result.files.items()):
        atomic_write(out_dir / rel, data)
    for rel in stale:
        (out_dir / rel).unlink()
    atomic_write(out_dir / "build.json", _json_bytes(dict(result.record, removed_stale=stale)))
    return stale


def check_build(result: Build, out_dir: str | os.PathLike) -> list[str]:
    """Differences between the in-memory build and DIR ("stale: X", "missing: X", "extra: X"); writes nothing."""
    out_dir = Path(out_dir)
    previous: list[str] = []
    with contextlib.suppress(OSError, ValueError, AttributeError):
        recorded = json.loads((out_dir / "build.json").read_text(encoding="utf-8")).get("removed_stale", [])
        if isinstance(recorded, list) and all(isinstance(x, str) for x in recorded):
            previous = recorded  # the last real build's removals are history, not a difference
    expected = dict(result.files)
    expected["build.json"] = _json_bytes(dict(result.record, removed_stale=previous))
    problems = []
    for rel, data in sorted(expected.items()):
        path = out_dir / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
        elif path.read_bytes() != data:
            problems.append(f"stale: {rel}")
    problems += [f"extra: {rel}" for rel in stale_files(out_dir, result.files)]
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--numbers", nargs="+", required=True, type=Path, metavar="FILE",
                        help="numbers.json files written by the reducers")
    parser.add_argument("--out", required=True, type=Path, metavar="DIR", help="generated directory")
    parser.add_argument("--check", action="store_true", help="compare DIR with a fresh build; write nothing")
    parser.add_argument("--allow-dev", "--draft", dest="allow_dev", action="store_true",
                        help="build from development numbers; the output is marked DEV NUMBERS")
    args = parser.parse_args(argv)
    try:
        result = build(args.numbers, out_dir=args.out, allow_dev=args.allow_dev)
    except RefusedError as exc:
        print(f"build_paper: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except BuildError as exc:
        print(f"build_paper: error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    if result.reasons:
        print("build_paper: DEV build (--allow-dev); every output is marked DEV NUMBERS:", file=sys.stderr)
        for reason in result.reasons:
            print(f"  {reason}", file=sys.stderr)
    if args.check:
        problems = check_build(result, args.out)
        if problems:
            print(f"build_paper --check: {args.out} is out of date ({len(problems)} files):")
            for problem in problems:
                print(f"  {problem}")
            return EXIT_FAILED
        print(f"build_paper --check: {args.out} is up to date ({len(result.files) + 1} files, {result.mode})")
        return EXIT_OK
    try:
        removed = write_build(result, args.out)
    except OSError as exc:
        print(f"build_paper: error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    record = result.record
    print(f"build_paper: {len(record['keys'])} keys, {len(record['tables'])} tables, {len(record['figures'])} "
          f"figures -> {args.out} ({result.mode})")
    for rel in removed:
        print(f"  removed stale {rel}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
