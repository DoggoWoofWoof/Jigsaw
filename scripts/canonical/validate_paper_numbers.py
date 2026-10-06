r"""Proof that the paper prints no hand-typed result.

    python -m scripts.canonical.validate_paper_numbers --tex PAPER.tex --generated DIR --allowlist ALLOW.json
        [--write-allowlist-draft FILE] [--allow-dev] [--verbose]

Checks; every finding is reported as file:line with a short context:
  keys      Every \jnum, \jci and \jn key used by the paper, or by a file it pulls in
            with \input or \include (resolved relative to the tex file), is defined in
            DIR/numbers.tex, with an interval for \jci and an n for \jn, and the paper
            inputs DIR/numbers.tex. Keys the paper never uses are information.
  build     DIR/build.json matches every numbers.json and figure CSV it names and every
            generated file it lists (sha256), and each generated table or figure the
            paper pulls in (\input{.../tables/X}, \includegraphics{.../figures/X.pdf})
            exists and belongs to that build. A build_paper output pulled in from outside
            DIR is an error; other graphics from outside DIR are information.
  numerals  Every numeral in the body (\begin{document} to \end{document}, comments
            stripped) outside masked text matches a reviewed allowlist entry.
  dev       A DIR built with --allow-dev fails unless --allow-dev is given here too.

Masked text, which cannot hold a hand-typed result: the arguments of citation,
reference, label, link and file commands (\cite..., \ref, \eqref, \label, \autoref,
\cref, \url, the URL of \href, \includegraphics, \input, \include, \bibliography...,
\jnum, \jci, \jn); environment names and the layout arguments of \begin (placement,
column specs, widths, the line-number step of algorithmic); the length, count and
colour arguments of layout commands (\vspace, \setlength, \resizebox, the span of
\multicolumn, \cmidrule, \\[2pt], \ccsdesc[500], \color ...); dimensions such as 2pt
or 0.5\linewidth; TeX parameter assignments; macro parameters (#1); \verb and
verbatim-like environments. Files inside DIR are generated: they are read for key
uses, never for numerals. Numerals in the preamble are not checked.

A numeral is a digit run with optional decimal or thousands parts (95.4, 1,800,
1{,}800). Its context is the text around it within its paragraph, whitespace
collapsed, at most CONTEXT_CHARS characters on each side cut back to whole words.
Contexts hold no line numbers, so an allowlist entry survives every edit outside
its own context.

Allowlist: {"schema": "jigsaw-paper-allowlist-v1", "notes": [...], "entries": [...]}
or a bare list of entries. An entry is either
  {"pattern": REGEX, "reason": TEXT}: the numeral is allowed when a match of REGEX that
      starts at or before the numeral in its context covers the whole numeral; a
      pattern that allows a bare numeral without context is refused; or
  {"context": SNIPPET, "numeral": N, "reason": TEXT}: allowed when the numeral is N and
      its context equals SNIPPET after whitespace normalisation.
An entry whose reason is "UNREVIEWED" never allows anything. --write-allowlist-draft
FILE writes one such entry per unmatched (numeral, context) for a human to triage.

Exit code 0 when every check passes, 1 otherwise (including an unreadable allowlist).
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import build_paper
from . import numbers_json as nj

ALLOWLIST_SCHEMA = "jigsaw-paper-allowlist-v1"
UNREVIEWED = "UNREVIEWED"
CHECKS = ("keys", "build", "numerals", "dev")
CONTEXT_CHARS = 40
MAX_INCLUDE_DEPTH = 16
NUMERAL_RE = re.compile(r"(?<![0-9])[0-9]+(?:(?:\{,\}|[.,])[0-9]+)*")
KEY_MACROS = ("jnum", "jci", "jn")
GRAPHIC_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".eps")
BARE_NUMERAL_PROBES = ("7", "95.4", "1,800", "1{,}800", "2026")

# One letter per argument: m {...} or a single token, o [...], p (...), * an optional
# star. Lower case masks the argument; upper case leaves it to the numeral scan.
_CITES = ("cite", "citep", "citet", "citealp", "citealt", "citeauthor", "citeyear", "citeyearpar", "citeN",
          "shortcite", "parencite", "textcite", "autocite")
_REFS = ("ref", "pageref", "autoref", "cref", "Cref", "nameref", "vref")
ARG_SIGNATURES: dict[str, str] = {
    **{name: "*oom" for name in _CITES},
    **{name: "*m" for name in _REFS},
    "nocite": "m", "citenum": "m", "eqref": "m", "crefrange": "mm", "Crefrange": "mm", "label": "om",
    "hyperref": "oM", "url": "m", "path": "m", "nolinkurl": "m", "href": "mM", "doi": "m", "email": "m",
    "orcid": "m", "includegraphics": "*oom", "input": "m", "include": "m", "includeonly": "m",
    "bibliography": "m", "bibliographystyle": "m", "addbibresource": "om", "end": "m",
    "jnum": "m", "jci": "m", "jn": "m",
    "vspace": "*m", "hspace": "*m", "addvspace": "m", "setlength": "mm", "addtolength": "mm",
    "settowidth": "mm", "settoheight": "mm", "settodepth": "mm", "setcounter": "mm", "addtocounter": "mm",
    "stepcounter": "m", "refstepcounter": "m", "resizebox": "*mmM", "scalebox": "moM", "rotatebox": "omM",
    "adjustbox": "mM", "rule": "omm", "raisebox": "mooM", "makebox": "ooM", "framebox": "ooM",
    "parbox": "ooomM", "fontsize": "mm", "linespread": "m", "multicolumn": "mmM", "multirow": "omomoM",
    "cmidrule": "opm", "cline": "m", "toprule": "o", "midrule": "o", "bottomrule": "o", "addlinespace": "o",
    "specialrule": "mmm", "enlargethispage": "*m", "linebreak": "o", "nolinebreak": "o", "pagebreak": "o",
    "nopagebreak": "o", "color": "om", "textcolor": "omM", "colorbox": "omM", "fcolorbox": "ommM",
    "pagecolor": "om", "definecolor": "ommm", "cellcolor": "om", "rowcolor": "om", "columncolor": "om",
    "newcommand": "*moOM", "renewcommand": "*moOM", "providecommand": "*moOM", "DeclareRobustCommand": "*moOM",
    "ccsdesc": "om", "pagestyle": "m", "thispagestyle": "m", "pagenumbering": "m", "hyphenation": "m",
    "ding": "m", "symbol": "m", "setcopyright": "m", "settopmatter": "m", "acmConference": "ommm",
    "acmYear": "m", "copyrightyear": "m", "acmDOI": "m", "acmISBN": "m", "acmPrice": "m", "acmVolume": "m",
    "acmNumber": "m", "acmArticle": "m", "acmMonth": "m", "acmBooktitle": "m", "acmSubmissionID": "m",
    "received": "om", "postcode": "m", "streetaddress": "m",
}
ENV_SIGNATURES: dict[str, str] = {
    "tabular": "om", "tabular*": "mom", "tabularx": "mom", "tabulary": "mom", "array": "om", "longtable": "om",
    "minipage": "ooom", "figure": "o", "figure*": "o", "table": "o", "table*": "o", "algorithm": "o",
    "algorithm*": "o", "algorithmic": "o", "wrapfigure": "oomom", "wraptable": "oomom", "multicols": "mO",
    "multicols*": "mO", "subfigure": "om", "subtable": "om", "tcolorbox": "o", "itemize": "o",
    "enumerate": "o", "description": "o", "thebibliography": "m", "list": "mm", "alignat": "m",
    "alignat*": "m", "adjustbox": "m",
}
VERBATIM_ENVS = ("verbatim", "verbatim*", "Verbatim", "lstlisting", "minted", "comment")

_NUM = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
_LENGTHS = ("textwidth", "linewidth", "columnwidth", "textheight", "paperwidth", "paperheight", "hsize", "vsize",
            "baselineskip", "parindent", "parskip", "tabcolsep", "arraycolsep", "columnsep", "unitlength", "fill",
            "stretch", "dimexpr")
_PARAMETERS = ("penalty", "hyphenpenalty", "exhyphenpenalty", "tolerance", "pretolerance", "looseness",
               "widowpenalty", "clubpenalty", "displaywidowpenalty", "brokenpenalty", "interlinepenalty",
               "linepenalty", "adjdemerits", "doublehyphendemerits", "finalhyphendemerits", "predisplaypenalty",
               "postdisplaypenalty", "floatpenalty", "hbadness", "vbadness", "emergencystretch", "hfuzz", "vfuzz",
               "parindent", "parskip", "baselineskip", "lineskip", "lineskiplimit", "tabcolsep", "arraycolsep",
               "columnsep", "abovecaptionskip", "belowcaptionskip", "textfloatsep", "floatsep", "intextsep",
               "dbltextfloatsep", "dblfloatsep", "abovedisplayskip", "belowdisplayskip", "itemsep", "topsep",
               "parsep", "partopsep", "leftmargin", "labelsep", "kern", "hskip", "vskip", "mskip", "mkern")
_RATIOS = ("arraystretch", "baselinestretch", "topfraction", "bottomfraction", "textfraction", "floatpagefraction",
           "dbltopfraction", "dblfloatpagefraction")
DIMENSION_RE = re.compile(rf"(?<![A-Za-z0-9.])-?{_NUM}(?:(?:pt|pc|in|bp|cm|mm|dd|cc|sp|em|ex|mu)(?![A-Za-z])"
                          rf"|\s*(?=\\(?:{'|'.join(_LENGTHS)})(?![A-Za-z])))")
PARAMETER_RE = re.compile(rf"\\(?:{'|'.join(_PARAMETERS)})(?![A-Za-z])\s*=?\s*(-?{_NUM})")
RATIO_RE = re.compile(rf"\\(?:renewcommand\*?|def)\s*\{{?\s*\\(?:{'|'.join(_RATIOS)})\s*\}}?\s*\{{\s*({_NUM})\s*\}}")
_DEFINITION_RE = re.compile(r"^\\expandafter\\def\\csname (jnum|jci|jn)@([^\s\\]+)\\endcsname\{", re.M)
_BEGIN_DOCUMENT_RE = re.compile(r"\\begin\s*\{document\}")
_END_DOCUMENT_RE = re.compile(r"\\end\s*\{document\}")


class AllowlistError(ValueError):
    """An allowlist that cannot be used."""


# ---------------------------------------------------------------------- report
@dataclass
class Finding:
    check: str  # one of CHECKS
    severity: str  # "error" or "info"
    file: str
    line: int  # 0 when the finding is about a whole file
    message: str
    context: str = ""

    def format(self) -> str:
        where = f"{self.file}:{self.line}" if self.line else self.file
        return f"[{self.check}] {where}: {self.message}" + (f" | {self.context}" if self.context else "")


@dataclass
class Occurrence:
    file: str
    line: int
    column: int
    numeral: str
    context: str
    offset: int  # position of the numeral in context
    allowed_by: str | None = None  # reason of the allowing entry


@dataclass
class KeyUse:
    macro: str
    key: str
    file: str
    line: int
    context: str


@dataclass
class Report:
    tex: str
    generated: str
    allow_dev: bool = False
    mode: str | None = None
    files: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    occurrences: list[Occurrence] = field(default_factory=list)
    key_uses: list[KeyUse] = field(default_factory=list)
    defined_keys: list[str] = field(default_factory=list)
    unused_keys: list[str] = field(default_factory=list)
    unreviewed_entries: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors()

    def errors(self, check: str | None = None) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error" and check in (None, f.check)]

    def infos(self, check: str | None = None) -> list[Finding]:
        return [f for f in self.findings if f.severity == "info" and check in (None, f.check)]

    def unmatched(self) -> list[Occurrence]:
        return [o for o in self.occurrences if o.allowed_by is None]

    def counts(self) -> dict[str, dict[str, int]]:
        return {check: {"errors": len(self.errors(check)), "info": len(self.infos(check))} for check in CHECKS}

    def summary(self, *, verbose: bool = False) -> str:
        allowed = len(self.occurrences) - len(self.unmatched())
        lines = [f"validate_paper_numbers: {self.tex} against {self.generated} (build mode {self.mode or 'unknown'})"]
        notes = {
            "keys": f"{len(self.key_uses)} uses of {len({u.key for u in self.key_uses})} keys; "
                    f"{len(self.unused_keys)} of {len(self.defined_keys)} defined keys unused",
            "build": f"{len(self.files)} source files read",
            "numerals": f"{len(self.occurrences)} numerals: {allowed} allowed, {len(self.unmatched())} unmatched; "
                        f"{self.unreviewed_entries} UNREVIEWED entries ignored",
            "dev": "dev build allowed" if self.allow_dev else "frozen build required",
        }
        for check in CHECKS:
            lines.append(f"  {check:<9} {len(self.errors(check))} errors, {len(self.infos(check))} info ({notes[check]})")
        if self.errors():
            lines.append("errors:")
            lines += [f"  {finding.format()}" for finding in self.errors()]
        if self.infos():
            lines.append("info:")
            lines += [f"  {finding.format()}" for finding in self.infos()]
        if self.unused_keys:
            shown = self.unused_keys if verbose else self.unused_keys[:10]
            more = "" if len(shown) == len(self.unused_keys) else f" (+{len(self.unused_keys) - len(shown)} more; --verbose)"
            lines.append(f"unused keys: {', '.join(shown)}{more}")
        if verbose:
            lines.append("allowed numerals:")
            lines += [f"  {o.file}:{o.line}: {o.numeral!r} ({o.allowed_by}) | {o.context}"
                      for o in self.occurrences if o.allowed_by is not None]
        lines.append("result: " + ("PASS" if self.ok else "FAIL"))
        return "\n".join(lines)


# ---------------------------------------------------------------------- allowlist
@dataclass(frozen=True)
class AllowEntry:
    reason: str
    pattern: str | None = None
    context: str | None = None
    numeral: str | None = None

    @property
    def reviewed(self) -> bool:
        return self.reason.strip() != UNREVIEWED


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def covers(regex: re.Pattern[str], text: str, start: int, end: int) -> bool:
    """True when a match of regex starting at or before `start` reaches `end`."""
    if regex.search(text) is None:
        return False
    for begin in range(start + 1):
        match = regex.match(text, begin)
        if match is not None and match.end() >= end:
            return True
    return False


def load_allowlist(source: str | os.PathLike | Mapping[str, Any] | list[Any] | None) -> list[AllowEntry]:
    """Entries from a file path, a parsed allowlist object or a bare list; refuses malformed entries."""
    if source is None:
        return []
    if isinstance(source, (str, os.PathLike)):
        try:
            source = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AllowlistError(f"cannot read allowlist {source}: {exc}") from None
    if isinstance(source, Mapping):
        extra = set(source) - {"schema", "notes", "entries"}
        if extra:
            raise AllowlistError(f"allowlist: unknown fields {sorted(extra)}")
        if source.get("schema", ALLOWLIST_SCHEMA) != ALLOWLIST_SCHEMA:
            raise AllowlistError(f"allowlist schema {source.get('schema')!r}, expected {ALLOWLIST_SCHEMA}")
        items = source.get("entries", [])
    else:
        items = source
    if not isinstance(items, list):
        raise AllowlistError("allowlist entries must be a list")
    entries = []
    for index, item in enumerate(items):
        where = f"allowlist entry {index}"
        if not isinstance(item, Mapping) or not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise AllowlistError(f"{where}: an entry is an object with a non-empty reason")
        if set(item) == {"pattern", "reason"} and isinstance(item["pattern"], str):
            try:
                regex = re.compile(item["pattern"])
            except re.error as exc:
                raise AllowlistError(f"{where}: bad pattern {item['pattern']!r}: {exc}") from None
            for probe in BARE_NUMERAL_PROBES:
                if covers(regex, probe, 0, len(probe)):
                    raise AllowlistError(f"{where}: pattern {item['pattern']!r} allows the bare numeral {probe!r}; "
                                         "a pattern must require context")
            entries.append(AllowEntry(reason=item["reason"], pattern=item["pattern"]))
        elif set(item) == {"context", "numeral", "reason"} and isinstance(item["context"], str) \
                and isinstance(item["numeral"], str):
            context = normalize_space(item["context"])
            if not NUMERAL_RE.fullmatch(item["numeral"]) or item["numeral"] not in context:
                raise AllowlistError(f"{where}: numeral {item['numeral']!r} must be a numeral inside its context")
            entries.append(AllowEntry(reason=item["reason"], context=context, numeral=item["numeral"]))
        else:
            raise AllowlistError(f'{where}: expected {{"pattern", "reason"}} or {{"context", "numeral", "reason"}}, '
                                 f"got {sorted(item)}")
    return entries


class _Matcher:
    def __init__(self, entries: Iterable[AllowEntry]) -> None:
        reviewed = [entry for entry in entries if entry.reviewed]
        self.patterns = [(re.compile(e.pattern), e) for e in reviewed if e.pattern is not None]
        self.contexts = {(e.numeral, e.context): e for e in reviewed if e.context is not None}

    def match(self, occurrence: Occurrence) -> AllowEntry | None:
        entry = self.contexts.get((occurrence.numeral, occurrence.context))
        if entry is not None:
            return entry
        end = occurrence.offset + len(occurrence.numeral)
        for regex, candidate in self.patterns:
            if covers(regex, occurrence.context, occurrence.offset, end):
                return candidate
        return None


def write_allowlist_draft(report: Report, path: str | os.PathLike) -> int:
    """Write every unmatched (numeral, context) as an UNREVIEWED entry; returns the entry count."""
    seen: set[tuple[str, str]] = set()
    entries = []
    for occurrence in report.unmatched():
        ident = (occurrence.numeral, occurrence.context)
        if ident not in seen:
            seen.add(ident)
            entries.append({"context": occurrence.context, "numeral": occurrence.numeral, "reason": UNREVIEWED})
    draft = {
        "schema": ALLOWLIST_SCHEMA,
        "notes": [f"Draft from validate_paper_numbers for {report.tex}: one entry per unmatched numeral.",
                  f'Entries with reason "{UNREVIEWED}" never allow anything. For each entry either replace the '
                  "numeral in the paper by \\jnum{KEY}, or write why it is structural (not a result) as its reason "
                  "and move it into the reviewed allowlist."],
        "entries": entries,
    }
    data = (json.dumps(draft, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    build_paper.atomic_write(Path(path), data)
    return len(entries)


# ---------------------------------------------------------------------- scanning
def _comment_start(line: str) -> int:
    i, n = 0, len(line)
    while i < n:
        char = line[i]
        if char == "\\":
            if line.startswith("verb", i + 1) and not (i + 5 < n and line[i + 5].isalpha()):
                j = i + 5 + (1 if line[i + 5:i + 6] == "*" else 0)
                if j < n:
                    close = line.find(line[j], j + 1)
                    i = n if close < 0 else close + 1
                    continue
            i += 2
            continue
        if char == "%":
            return i
        i += 1
    return -1


def strip_comments(text: str) -> str:
    """Comments blanked with spaces, so offsets and line numbers stay those of the source."""
    lines = []
    for line in text.split("\n"):
        cut = _comment_start(line)
        lines.append(line if cut < 0 else line[:cut] + " " * (len(line) - cut))
    return "\n".join(lines)


def _is_letter(char: str) -> bool:
    return "a" <= char <= "z" or "A" <= char <= "Z"


def _skip_space(text: str, pos: int) -> int:
    """Skip spaces and at most one line break (a blank line ends an argument search)."""
    breaks = 0
    while pos < len(text) and text[pos] in " \t\n":
        if text[pos] == "\n":
            breaks += 1
            if breaks > 1:
                break
        pos += 1
    return pos


def _group_end(text: str, start: int, opener: str, closer: str) -> int | None:
    """End (exclusive) of the group opened at text[start]; [..] and (..) end at the first closer outside braces."""
    depth = 0
    i = start
    while i < len(text):
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if opener == "{":
            depth += {"{": 1, "}": -1}.get(char, 0)
            if depth == 0:
                return i + 1
        else:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth < 0:
                    return None
            elif char == closer and depth == 0 and i > start:
                return i + 1
        i += 1
    return None


def _mask(mask: bytearray, start: int, end: int) -> None:
    mask[start:end] = b"\x01" * (end - start)


def _parse_args(text: str, pos: int, signature: str, mask: bytearray) -> tuple[int, list[tuple[int, int] | None]]:
    spans: list[tuple[int, int] | None] = []
    for spec in signature:
        k = _skip_space(text, pos)
        if spec == "*":
            if k < len(text) and text[k] == "*":
                pos = k + 1
            spans.append(None)
            continue
        if spec in "oOp":
            opener, closer = ("(", ")") if spec == "p" else ("[", "]")
            end = _group_end(text, k, opener, closer) if k < len(text) and text[k] == opener else None
            if end is not None:
                if spec != "O":
                    _mask(mask, k, end)
                pos = end
                spans.append((k, end))
            else:
                spans.append(None)
            continue
        if k >= len(text) or text[k] in "}] \t\n":
            spans.append(None)
            break
        if text[k] == "{":
            end = _group_end(text, k, "{", "}")
            if end is None:
                spans.append(None)
                break
        elif text[k] == "\\":
            end = k + 2
            while end < len(text) and _is_letter(text[k + 1]) and _is_letter(text[end]):
                end += 1
        else:
            end = k + 1
        if spec == "m":
            _mask(mask, k, end)
        pos = end
        spans.append((k, end))
    return pos, spans


@dataclass
class _Scan:
    text: str
    mask: bytearray
    keys: list[tuple[str, str, int]] = field(default_factory=list)  # (macro, key, position)
    malformed: list[tuple[str, int]] = field(default_factory=list)  # (macro, position)
    inputs: list[tuple[str, str, int]] = field(default_factory=list)  # (command, argument, position)
    graphics: list[tuple[str, int]] = field(default_factory=list)  # (argument, position)


def _command(scan: _Scan, name: str, start: int, end: int) -> int:
    """Handle one control word; returns where the scan continues (masked arguments are skipped by the mask)."""
    text, mask = scan.text, scan.mask
    if name == "verb":
        j = end + (1 if text[end:end + 1] == "*" else 0)
        if j < len(text):
            close = text.find(text[j], j + 1)
            stop = len(text) if close < 0 else close + 1
            _mask(mask, j, stop)
            return stop
        return end
    if name == "begin":
        k = _skip_space(text, end)
        close = _group_end(text, k, "{", "}") if k < len(text) and text[k] == "{" else None
        if close is None:
            return end
        env = text[k + 1:close - 1].strip()
        _mask(mask, k, close)
        if env in VERBATIM_ENVS:
            stop = re.compile(r"\\end\s*\{\s*" + re.escape(env) + r"\s*\}").search(text, close)
            _mask(mask, close, len(text) if stop is None else stop.start())
            return len(text) if stop is None else stop.start()
        _parse_args(text, close, ENV_SIGNATURES.get(env, ""), mask)
        return close
    if name in KEY_MACROS:
        k = _skip_space(text, end)
        close = _group_end(text, k, "{", "}") if k < len(text) and text[k] == "{" else None
        if close is None:
            scan.malformed.append((name, start))
            return end
        scan.keys.append((name, text[k + 1:close - 1], start))
        _mask(mask, k, close)
        return close
    if name in ("input", "include"):
        k = _skip_space(text, end)
        if k < len(text) and text[k] == "{":
            close = _group_end(text, k, "{", "}")
            if close is not None:
                scan.inputs.append((name, text[k + 1:close - 1].strip(), start))
                _mask(mask, k, close)
                return close
        elif name == "input":
            bare = re.compile(r"[^\s{}%\\]+").match(text, k)
            if bare is not None:
                scan.inputs.append((name, bare.group(0), start))
                _mask(mask, k, bare.end())
                return bare.end()
        return end
    signature = ARG_SIGNATURES.get(name)
    if signature:
        _, spans = _parse_args(text, end, signature, mask)
        if name == "includegraphics" and spans and spans[-1] is not None:
            first, last = spans[-1]
            scan.graphics.append((text[first + 1:last - 1].strip() if text[first] == "{" else text[first:last], start))
    return end


def scan_source(text: str) -> _Scan:
    """Mask the parts of comment-stripped text that cannot hold a hand-typed result and collect references."""
    mask = bytearray(len(text))
    for regex, group in ((DIMENSION_RE, 0), (PARAMETER_RE, 1), (RATIO_RE, 1)):
        for match in regex.finditer(text):
            _mask(mask, match.start(group), match.end(group))
    scan = _Scan(text=text, mask=mask)
    i, n = 0, len(text)
    while i < n:
        if mask[i]:
            i += 1
            continue
        char = text[i]
        if char == "#" and i + 1 < n and text[i + 1].isdigit():
            mask[i + 1] = 1  # a macro parameter
            i += 2
            continue
        if char != "\\" or i + 1 >= n:
            i += 1
            continue
        if not _is_letter(text[i + 1]):
            if text[i + 1].isdigit():
                mask[i + 1] = 1  # a control symbol such as \1
            elif text[i + 1] == "\\":
                _parse_args(text, i + 2, "*o", mask)  # \\[2pt]
            i += 2
            continue
        j = i + 1
        while j < n and _is_letter(text[j]):
            j += 1
        i = _command(scan, text[i + 1:j], i, j)
    return scan


def _blank_lines(raw: str) -> tuple[list[int], list[int]]:
    """Start and end offsets of the blank source lines (a comment-only line is not blank)."""
    starts, ends, pos = [], [], 0
    for line in raw.split("\n"):
        if not line.strip():
            starts.append(pos)
            ends.append(pos + len(line))
        pos += len(line) + 1
    return starts, ends


def context_of(text: str, start: int, end: int, low: int, high: int) -> tuple[str, int]:
    """(context, offset of the numeral) for text[start:end] inside the paragraph text[low:high]."""
    left = re.sub(r"\s+", " ", text[low:start])
    right = re.sub(r"\s+", " ", text[end:high])
    if len(left) > CONTEXT_CHARS:
        cut = len(left) - CONTEXT_CHARS
        if not left[cut - 1].isspace():
            space = left.find(" ", cut)
            cut = space + 1 if space >= 0 else cut
        left = left[cut:]
    if len(right) > CONTEXT_CHARS:
        cut = CONTEXT_CHARS
        if not right[cut].isspace():
            space = right.rfind(" ", 0, cut)
            cut = space if space >= 0 else cut
        right = right[:cut]
    left, right = left.lstrip(), right.rstrip()
    return left + text[start:end] + right, len(left)


# ---------------------------------------------------------------------- validation
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(path: Path, folder: Path) -> str | None:
    """path relative to folder (forward slashes), or None when it lies outside."""
    try:
        rel = Path(os.path.relpath(path.resolve(), folder.resolve())).as_posix()
    except ValueError:
        return None
    return None if rel == ".." or rel.startswith("../") else rel


def _candidates(base: Path, argument: str, command: str) -> list[Path]:
    if command == "includegraphics":
        if Path(argument).suffix.lower() in GRAPHIC_EXTENSIONS:
            return [base / argument]
        return [base / (argument + ext) for ext in GRAPHIC_EXTENSIONS] + [base / argument]
    if command == "include":
        return [base / (argument + ".tex")]
    return [base / argument] if argument.endswith(".tex") else [base / (argument + ".tex"), base / argument]


def _resolve(base: Path, argument: str, command: str) -> tuple[Path, bool]:
    options = _candidates(base, argument, command)
    for option in options:
        if option.is_file():
            return option, True
    return options[0], False


def _generated_elsewhere(path: Path) -> bool:
    """True for a build_paper output (by its header line or PDF creator); called for files outside DIR."""
    try:
        if path.suffix.lower() == ".pdf":
            return f"/Creator ({build_paper.PDF_METADATA['Creator']})".encode("ascii") in path.read_bytes()
        with path.open(encoding="utf-8", errors="replace") as handle:
            first = handle.readline()
    except OSError:
        return False
    return first.startswith("%") and "generated by scripts/canonical/build_paper.py" in first.lower()


class _Validator:
    def __init__(self, tex: Path, generated: Path, allow_dev: bool) -> None:
        self.tex, self.generated, self.base = tex, generated, tex.parent
        self.report = Report(tex=self._display(tex), generated=self._display(generated), allow_dev=allow_dev)
        self.record: dict[str, Any] | None = None
        self.numbers_tex_included = False
        self.visited: set[str] = set()

    def _display(self, path: Path) -> str:
        try:
            return Path(os.path.relpath(path.resolve(), Path.cwd().resolve())).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def add(self, check: str, severity: str, path: Path | str, line: int, message: str, context: str = "") -> None:
        name = path if isinstance(path, str) else self._display(path)
        self.report.findings.append(Finding(check, severity, name, line, message, context))

    # ---- build.json and dev mode
    def check_build(self) -> None:
        path = self.generated / "build.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.add("build", "error", path, 0, f"cannot read build.json ({exc}); run build_paper first")
            return
        if not isinstance(record, dict) or record.get("schema") != build_paper.BUILD_SCHEMA:
            self.add("build", "error", path, 0, f"not a {build_paper.BUILD_SCHEMA} record")
            return
        inputs, generated = record.get("inputs"), record.get("generated")
        if not (isinstance(inputs, list) and isinstance(generated, dict)
                and all(isinstance(i, dict) and isinstance(i.get("path"), str) and isinstance(i.get("sha256"), str)
                        for i in inputs)):
            self.add("build", "error", path, 0, "malformed build.json (inputs or generated); rebuild the paper")
            return
        self.record = record
        self.report.mode = record.get("mode")
        for item in inputs:
            source = Path(item["path"])
            source = source if source.is_absolute() else self.generated / source
            if not source.is_file():
                self.add("build", "error", path, 0, f"input {item['path']} named by build.json is missing")
            elif _sha256(source) != item["sha256"]:
                self.add("build", "error", path, 0, f"stale: {item['path']} changed since the build; rebuild the paper")
        for rel, sha in sorted(generated.items()):
            target = self.generated / rel
            if not target.is_file():
                self.add("build", "error", path, 0, f"generated file {rel} is missing")
            elif _sha256(target) != sha:
                self.add("build", "error", path, 0, f"generated file {rel} differs from build.json "
                                                    "(edited by hand, or a partial build)")
        for rel in build_paper.stale_files(self.generated, generated):
            self.add("build", "info", self.generated / rel, 0, "not produced by the current build")
        if record.get("mode") == "dev" and not self.report.allow_dev:
            self.add("dev", "error", path, 0, "DIR was built from development numbers (--allow-dev); pass "
                                              "--allow-dev to validate a draft, or build from frozen test numbers")
        elif record.get("mode") not in ("frozen", "dev"):
            self.add("dev", "error", path, 0, f"unknown build mode {record.get('mode')!r}")

    def defined_keys(self) -> dict[str, set[str]] | None:
        path = self.generated / "numbers.tex"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            self.add("keys", "error", path, 0, f"cannot read numbers.tex ({exc})")
            return None
        defined: dict[str, set[str]] = {macro: set() for macro in KEY_MACROS}
        for match in _DEFINITION_RE.finditer(text):
            defined[match.group(1)].add(match.group(2))
        return defined

    # ---- sources
    def visit(self, path: Path, body: bool | None, depth: int, generated: bool) -> None:
        ident = os.path.normcase(str(path.resolve()))
        if ident in self.visited:
            return
        self.visited.add(ident)
        name = self._display(path)
        self.report.files.append(name)
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = strip_comments(raw)
        scan = scan_source(text)
        line_starts = [0] + [m.end() for m in re.finditer("\n", text)]

        def line_of(pos: int) -> int:
            return bisect.bisect_right(line_starts, pos)

        def snippet(pos: int) -> str:
            line = line_of(pos)
            end = line_starts[line] - 1 if line < len(line_starts) else len(text)
            return normalize_space(text[line_starts[line - 1]:end])[:100]

        if body is None:  # the main file: the body lies between \begin{document} and \end{document}
            begin = _BEGIN_DOCUMENT_RE.search(text)
            if begin is None:
                self.add("numerals", "error", path, 0, r"no \begin{document}: the body cannot be checked")
                ranges: list[tuple[int, int]] = []
            else:
                end = _END_DOCUMENT_RE.search(text, begin.end())
                ranges = [(begin.end(), len(text) if end is None else end.start())]
        else:
            ranges = [(0, len(text))] if body else []

        def in_body(pos: int) -> bool:
            return any(low <= pos < high for low, high in ranges)

        for macro, key, pos in scan.keys:
            self.report.key_uses.append(KeyUse(macro, key, name, line_of(pos), snippet(pos)))
        for macro, pos in scan.malformed:
            self.add("keys", "error", path, line_of(pos), f"\\{macro} needs a braced key", snippet(pos))
        for argument, pos in scan.graphics:
            self.graphic(path, argument, line_of(pos), snippet(pos))
        if not generated:
            self.numerals(path, name, raw, text, scan, ranges, line_starts)
        for command, argument, pos in scan.inputs:
            self.pull(path, command, argument, in_body(pos), depth, line_of(pos), snippet(pos))

    def pull(self, parent: Path, command: str, argument: str, body: bool, depth: int, line: int,
             context: str) -> None:
        target, exists = _resolve(self.base, argument, command)
        rel = _inside(target, self.generated)
        if rel is not None:
            if not exists:
                self.add("build", "error", parent, line, f"generated file {argument} is missing from the build",
                         context)
                return
            if self.record is not None and rel not in self.record.get("generated", {}):
                self.add("build", "error", parent, line, f"{rel} was not produced by the current build", context)
            if rel == "numbers.tex":
                self.numbers_tex_included = True
                return
            self.visit(target, False, depth + 1, generated=True)
            return
        if not exists:
            self.add("build", "error", parent, line, f"\\{command}{{{argument}}}: file not found", context)
            return
        if _generated_elsewhere(target):
            self.add("build", "error", parent, line, f"{argument} was generated by build_paper outside "
                                                     f"{self.report.generated}; validate the directory the paper "
                                                     "inputs", context)
            return
        if depth + 1 > MAX_INCLUDE_DEPTH:
            self.add("build", "error", parent, line, f"\\{command} nesting deeper than {MAX_INCLUDE_DEPTH}", context)
            return
        self.visit(target, body, depth + 1, generated=False)

    def graphic(self, parent: Path, argument: str, line: int, context: str) -> None:
        target, exists = _resolve(self.base, argument, "includegraphics")
        rel = _inside(target, self.generated)
        if rel is None:
            if exists and _generated_elsewhere(target):
                self.add("build", "error", parent, line, f"graphic {argument} was generated by build_paper outside "
                                                         f"{self.report.generated}", context)
            elif exists:
                self.add("build", "info", parent, line, f"graphic {argument} is not generated from numbers.json",
                         context)
            else:
                self.add("build", "error", parent, line, f"graphic {argument} not found", context)
            return
        if Path(argument).suffix.lower() not in GRAPHIC_EXTENSIONS:
            self.add("build", "error", parent, line, f"write the extension: graphicx reads the dots of a figure id "
                                                     f"as one (\\includegraphics{{{argument}.pdf}})", context)
        if not exists:
            self.add("build", "error", parent, line, f"generated figure {argument} is missing from the build", context)
        elif self.record is not None and rel not in self.record.get("generated", {}):
            self.add("build", "error", parent, line, f"{rel} was not produced by the current build", context)

    def numerals(self, path: Path, name: str, raw: str, text: str, scan: _Scan, ranges: list[tuple[int, int]],
                 line_starts: list[int]) -> None:
        visible = "".join(" " if flag else char for char, flag in zip(text, scan.mask))
        blank_starts, blank_ends = _blank_lines(raw)
        for low, high in ranges:
            for match in NUMERAL_RE.finditer(visible, low, high):
                start, end = match.span()
                before = bisect.bisect_right(blank_ends, start) - 1
                after = bisect.bisect_left(blank_starts, start)
                paragraph = (blank_ends[before] if before >= 0 else 0,
                             blank_starts[after] if after < len(blank_starts) else len(text))
                context, offset = context_of(text, start, end, *paragraph)
                line = bisect.bisect_right(line_starts, start)
                self.report.occurrences.append(Occurrence(name, line, start - line_starts[line - 1] + 1,
                                                          text[start:end], context, offset))

    # ---- keys
    def check_keys(self, defined: dict[str, set[str]] | None) -> None:
        report = self.report
        for use in report.key_uses:
            if not nj.KEY_RE.match(use.key):
                self.add("keys", "error", use.file, use.line, f"malformed key {use.key!r} in \\{use.macro}", use.context)
            elif defined is None:
                continue
            elif use.key not in defined["jnum"]:
                self.add("keys", "error", use.file, use.line, f"undefined key {use.key} (\\{use.macro})", use.context)
            elif use.macro == "jci" and use.key not in defined["jci"]:
                self.add("keys", "error", use.file, use.line, f"\\jci{{{use.key}}}: the key has no interval",
                         use.context)
            elif use.macro == "jn" and use.key not in defined["jn"]:
                self.add("keys", "error", use.file, use.line, f"\\jn{{{use.key}}}: the key has no n", use.context)
        if report.key_uses and not self.numbers_tex_included:
            self.add("keys", "error", self.tex, 0, f"the paper uses \\jnum/\\jci/\\jn but never inputs "
                                                   f"{self.report.generated}/numbers.tex")
        if defined is not None:
            report.defined_keys = sorted(defined["jnum"])
            report.unused_keys = sorted(defined["jnum"] - {use.key for use in report.key_uses})
            if report.unused_keys:
                self.add("keys", "info", self.generated / "numbers.tex", 0,
                         f"{len(report.unused_keys)} defined keys are not used by the paper")


def validate(tex_path: str | os.PathLike, generated_dir: str | os.PathLike,
             allowlist: str | os.PathLike | Mapping[str, Any] | list[Any] | None, *,
             allow_dev: bool = False) -> Report:
    """Run the four checks; raises AllowlistError for a bad allowlist and OSError for an unreadable paper."""
    entries = load_allowlist(allowlist)
    validator = _Validator(Path(tex_path), Path(generated_dir), allow_dev)
    report = validator.report
    report.unreviewed_entries = sum(1 for entry in entries if not entry.reviewed)
    validator.check_build()
    defined = validator.defined_keys()
    validator.visit(Path(tex_path), None, 0, generated=False)
    validator.check_keys(defined)
    matcher = _Matcher(entries)
    for occurrence in report.occurrences:
        entry = matcher.match(occurrence)
        if entry is None:
            report.findings.append(Finding("numerals", "error", occurrence.file, occurrence.line,
                                           f"hand-typed numeral {occurrence.numeral!r}", occurrence.context))
        else:
            occurrence.allowed_by = entry.reason
    order = {check: index for index, check in enumerate(CHECKS)}
    report.findings.sort(key=lambda f: (order[f.check], f.severity, f.file, f.line))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tex", required=True, type=Path, help="the paper's main .tex file")
    parser.add_argument("--generated", required=True, type=Path, metavar="DIR", help="build_paper output directory")
    parser.add_argument("--allowlist", required=True, type=Path, help="reviewed allowlist JSON")
    parser.add_argument("--write-allowlist-draft", type=Path, metavar="FILE",
                        help="write every unmatched numeral as an UNREVIEWED entry")
    parser.add_argument("--allow-dev", "--draft", dest="allow_dev", action="store_true",
                        help="accept a DIR built with --allow-dev")
    parser.add_argument("--verbose", action="store_true", help="list allowed numerals and every unused key")
    args = parser.parse_args(argv)
    try:
        report = validate(args.tex, args.generated, args.allowlist, allow_dev=args.allow_dev)
    except (AllowlistError, OSError) as exc:
        print(f"validate_paper_numbers: error: {exc}", file=sys.stderr)
        return 1
    print(report.summary(verbose=args.verbose))
    if args.write_allowlist_draft is not None:
        if args.write_allowlist_draft.resolve() == args.allowlist.resolve():
            print("validate_paper_numbers: error: the draft must not overwrite the reviewed allowlist", file=sys.stderr)
            return 1
        count = write_allowlist_draft(report, args.write_allowlist_draft)
        print(f"wrote {count} {UNREVIEWED} entries to {args.write_allowlist_draft}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
