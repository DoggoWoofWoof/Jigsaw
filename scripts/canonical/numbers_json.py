"""numbers.json: the one format between the reducers and the paper builder.

Every number the paper prints comes from a numbers.json written here and is
typeset by build_paper.py as \\jnum{KEY}. A file holds

  schema         "jigsaw-numbers-v1"
  protocol_hash  protocol.protocol_hash() of the runs it summarises
  frozen         True only for a frozen protocol (protocol.is_frozen())
  split          the query split of the runs ("test" for the reported numbers)
  generated_utc  when it was written
  sources        {RUN_ID: {path, manifest_sha256, experiment, dataset}}
  numbers        {KEY: {value, unit, digits?, ci, n, source: [RUN_ID], note?}}
  tables         {TABLE_ID: {columns, rows, notes?}}; a cell is a label string,
                 {"key": KEY} or {"key": KEY, "ci": true}
  figures        {FIG_ID: {type, csv, x, y, log_x?, notes?}}; the rows are in
                 figures/<FIG_ID>.csv next to numbers.json, with the columns
                 series,x,y,lo,hi (curve) or group,series,y,lo,hi (bars)

Keys, table ids and figure ids are dot-separated segments of [A-Za-z0-9_-].
Units: pct (0..100), pp (a signed difference in percentage points), frac
(0..1), number (any real, e.g. a mean cost in blocks or an effective rank),
count (integer), seconds, ms, bytes, gb (decimal gigabytes), ratio (a positive
multiplier), pvalue (0..1) and text.
A value, interval or n that breaks its unit is refused when it is added, so a
reducer bug stops at the reducer and never reaches the paper.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import protocol

SCHEMA = "jigsaw-numbers-v1"
UNITS = ("pct", "pp", "frac", "number", "count", "seconds", "ms", "bytes", "gb", "ratio", "pvalue", "text")
KEY_RE = re.compile(r"^[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*$")
FIGURE_COLUMNS = {"curve": ("series", "x", "y", "lo", "hi"), "bars": ("group", "series", "y", "lo", "hi")}
_BOUNDS = {"pct": (0.0, 100.0), "pp": (-100.0, 100.0), "frac": (0.0, 1.0), "pvalue": (0.0, 1.0),
           "number": (-math.inf, math.inf)}
_EPS = 1e-9


class NumbersError(ValueError):
    pass


def check_key(key: str, what: str = "key") -> str:
    if not isinstance(key, str) or not KEY_RE.match(key):
        raise NumbersError(f"bad {what} {key!r}: dot-separated segments of [A-Za-z0-9_-]")
    return key


def _finite(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NumbersError(f"{key}: {value!r} is not a number")
    if not math.isfinite(float(value)):
        raise NumbersError(f"{key}: {value!r} is not finite")
    return float(value)


def check_value(key: str, value: Any, unit: str) -> Any:
    """The value, checked against its unit."""
    if unit not in UNITS:
        raise NumbersError(f"{key}: unknown unit {unit!r}")
    if unit == "text":
        if not isinstance(value, str):
            raise NumbersError(f"{key}: a text value must be a string")
        return value
    x = _finite(value, key)
    if unit == "count":
        if x != int(x) or x < 0:
            raise NumbersError(f"{key}: a count must be a non-negative integer, got {value!r}")
        return int(x)
    lo, hi = _BOUNDS.get(unit, (0.0 if unit in ("seconds", "ms", "bytes", "gb", "ratio") else -math.inf, math.inf))
    if not lo - _EPS <= x <= hi + _EPS:
        raise NumbersError(f"{key}: {value!r} is outside [{lo}, {hi}] for unit {unit}")
    if unit == "ratio" and x <= 0:
        raise NumbersError(f"{key}: a ratio must be positive")
    return int(x) if isinstance(value, int) else x


class Numbers:
    """Collects numbers, tables and figures; write() produces numbers.json and the figure CSVs."""

    def __init__(self, *, split: str, frozen: bool | None = None, protocol_hash: str | None = None) -> None:
        self.split = str(split)
        self.frozen = protocol.is_frozen() if frozen is None else bool(frozen)
        self.protocol_hash = protocol_hash or protocol.protocol_hash()
        self.sources: dict[str, dict[str, Any]] = {}
        self.numbers: dict[str, dict[str, Any]] = {}
        self.tables: dict[str, dict[str, Any]] = {}
        self.figures: dict[str, dict[str, Any]] = {}
        self.figure_rows: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------ adding
    def source(self, run_id: str, *, path: str | os.PathLike, manifest_sha256: str | None, experiment: str,
               dataset: str) -> str:
        check_key(run_id, "run id")
        entry = {"path": str(path), "manifest_sha256": manifest_sha256, "experiment": str(experiment),
                 "dataset": str(dataset)}
        if run_id in self.sources and self.sources[run_id] != entry:
            raise NumbersError(f"source {run_id} added twice with different content")
        self.sources[run_id] = entry
        return run_id

    def number(self, key: str, value: Any, unit: str, *, ci: Sequence[float] | None = None, n: int | None = None,
               digits: int | None = None, source: Iterable[str] = (), note: str | None = None) -> str:
        check_key(key)
        if key in self.numbers:
            raise NumbersError(f"{key} added twice")
        entry: dict[str, Any] = {"value": check_value(key, value, unit), "unit": unit, "ci": None, "n": None,
                                 "source": sorted(set(source))}
        if ci is not None:
            lo, hi = (check_value(key, v, unit) for v in ci)
            if lo > hi + _EPS:
                raise NumbersError(f"{key}: interval [{lo}, {hi}] is reversed")
            entry["ci"] = [lo, hi]
        if n is not None:
            if isinstance(n, bool) or int(n) != n or n < 0:
                raise NumbersError(f"{key}: n must be a non-negative integer")
            entry["n"] = int(n)
        if digits is not None:
            if int(digits) != digits or not 0 <= digits <= 6:
                raise NumbersError(f"{key}: digits must be an integer in 0..6")
            entry["digits"] = int(digits)
        if note:
            entry["note"] = str(note)
        unknown = [s for s in entry["source"] if s not in self.sources]
        if unknown:
            raise NumbersError(f"{key}: unknown sources {unknown}")
        self.numbers[key] = entry
        return key

    def rate(self, key: str, successes: int, n: int, *, source: Iterable[str] = (), digits: int = 1,
             note: str | None = None) -> str:
        """A rate in percent with its Wilson 95% interval (protocol STATISTICS rate_interval)."""
        from . import stats

        r = stats.rate(successes, n)
        if n == 0:
            raise NumbersError(f"{key}: a rate over zero units")
        return self.number(key, 100.0 * r.rate, "pct", ci=(100.0 * r.lo, 100.0 * r.hi), n=n, digits=digits,
                           source=source, note=note)

    def table(self, table_id: str, columns: Sequence[str], rows: Sequence[Sequence[Any]], *,
              notes: str | None = None) -> str:
        check_key(table_id, "table id")
        if table_id in self.tables:
            raise NumbersError(f"table {table_id} added twice")
        columns = [str(c) for c in columns]
        out_rows = []
        for row in rows:
            if len(row) != len(columns):
                raise NumbersError(f"table {table_id}: a row has {len(row)} cells for {len(columns)} columns")
            cells = []
            for cell in row:
                if isinstance(cell, str):
                    cells.append(cell)
                elif isinstance(cell, Mapping) and set(cell) <= {"key", "ci"} and "key" in cell:
                    if cell["key"] not in self.numbers:
                        raise NumbersError(f"table {table_id}: unknown key {cell['key']}")
                    if cell.get("ci") and self.numbers[cell["key"]]["ci"] is None:
                        raise NumbersError(f"table {table_id}: {cell['key']} has no interval")
                    cells.append({"key": cell["key"], **({"ci": True} if cell.get("ci") else {})})
                else:
                    raise NumbersError(f"table {table_id}: a cell must be a string or {{'key': KEY}}")
            out_rows.append(cells)
        self.tables[table_id] = {"columns": columns, "rows": out_rows, **({"notes": notes} if notes else {})}
        return table_id

    def figure(self, fig_id: str, kind: str, rows: Sequence[Mapping[str, Any]], *, x: str, y: str,
               log_x: bool = False, notes: str | None = None) -> str:
        check_key(fig_id, "figure id")
        if fig_id in self.figures:
            raise NumbersError(f"figure {fig_id} added twice")
        if kind not in FIGURE_COLUMNS:
            raise NumbersError(f"figure {fig_id}: unknown type {kind!r}")
        columns = FIGURE_COLUMNS[kind]
        clean = []
        for row in rows:
            extra = set(row) - set(columns)
            if extra or not {"series", "y"} <= set(row) or ("x" if kind == "curve" else "group") not in row:
                raise NumbersError(f"figure {fig_id}: row {dict(row)} does not fit columns {columns}")
            out = {}
            for c in columns:
                v = row.get(c)
                if c in ("series", "group"):
                    out[c] = str(v)
                elif v is None or (c in ("lo", "hi") and v == ""):
                    if c not in ("lo", "hi"):
                        raise NumbersError(f"figure {fig_id}: {c} missing")
                    out[c] = None
                else:
                    out[c] = _finite(v, f"figure {fig_id} {c}")
            clean.append(out)
        if not clean:
            raise NumbersError(f"figure {fig_id} has no rows")
        self.figures[fig_id] = {"type": kind, "csv": f"figures/{fig_id}.csv", "x": str(x), "y": str(y),
                                **({"log_x": True} if log_x else {}), **({"notes": notes} if notes else {})}
        self.figure_rows[fig_id] = clean
        return fig_id

    def merge(self, other: "Numbers") -> None:
        if other.protocol_hash != self.protocol_hash or other.split != self.split or other.frozen != self.frozen:
            raise NumbersError("merging numbers of different protocols, splits or freeze states")
        for run_id, entry in other.sources.items():
            self.source(run_id, **entry)
        for name, own, theirs in (("key", self.numbers, other.numbers), ("table", self.tables, other.tables),
                                  ("figure", self.figures, other.figures)):
            clash = sorted(set(own) & set(theirs))
            if clash:
                raise NumbersError(f"{name}s defined twice: {clash[:5]}")
        self.numbers.update(other.numbers)
        self.tables.update(other.tables)
        self.figures.update(other.figures)
        self.figure_rows.update(other.figure_rows)

    # ------------------------------------------------------------------ output
    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "protocol_hash": self.protocol_hash, "frozen": self.frozen, "split": self.split,
                "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sources": self.sources, "numbers": self.numbers, "tables": self.tables, "figures": self.figures}

    def write(self, out_dir: str | os.PathLike) -> Path:
        out_dir = Path(out_dir)
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)
        for fig_id, rows in self.figure_rows.items():
            columns = FIGURE_COLUMNS[self.figures[fig_id]["type"]]
            buf = io.StringIO()
            writer = csv.writer(buf, lineterminator="\n")
            writer.writerow(columns)
            for row in rows:
                writer.writerow(["" if row[c] is None else (repr(row[c]) if isinstance(row[c], float) else row[c])
                                 for c in columns])
            _atomic_write(out_dir / "figures" / f"{fig_id}.csv", buf.getvalue().encode("utf-8"))
        path = out_dir / "numbers.json"
        _atomic_write(path, (json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8"))
        return path


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def load(path: str | os.PathLike) -> dict[str, Any]:
    """Read and validate a numbers.json (and the presence of its figure CSVs)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise NumbersError(f"{path}: schema {data.get('schema')!r}, expected {SCHEMA}")
    for key, entry in data.get("numbers", {}).items():
        check_key(key)
        check_value(key, entry.get("value"), entry.get("unit"))
        if entry.get("ci") is not None:
            lo, hi = (check_value(key, v, entry["unit"]) for v in entry["ci"])
            if lo > hi + _EPS:
                raise NumbersError(f"{path}: {key} has a reversed interval")
        missing = [s for s in entry.get("source", []) if s not in data.get("sources", {})]
        if missing:
            raise NumbersError(f"{path}: {key} names unknown sources {missing}")
    for table_id, table in data.get("tables", {}).items():
        check_key(table_id, "table id")
        for row in table["rows"]:
            for cell in row:
                if isinstance(cell, dict) and cell.get("key") not in data["numbers"]:
                    raise NumbersError(f"{path}: table {table_id} names unknown key {cell.get('key')}")
    for fig_id, fig in data.get("figures", {}).items():
        check_key(fig_id, "figure id")
        if not (path.parent / fig["csv"]).exists():
            raise NumbersError(f"{path}: figure {fig_id} has no {fig['csv']}")
    return data


def file_sha256(path: str | os.PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
