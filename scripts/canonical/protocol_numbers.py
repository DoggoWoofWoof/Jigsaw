r"""The protocol's settings as paper numbers, so the paper types none of them.

    python -m scripts.canonical.protocol_numbers --out DIR [--list]

Every numeric setting of protocol.protocol_dict() becomes the key protocol.<SECTION>.<path> of a
numbers.json (numbers_json.py). Dict keys and list positions (0, 1, ...) name the path; in a name, a
decimal point between digits becomes p and any other character outside [A-Za-z0-9_-] becomes _:

  protocol.TIME.primary_T_seconds              30\,s
  protocol.TIME.run_T_max_seconds.E3_extension 600\,s
  protocol.QUERY_SIZES.2                       100
  protocol.DATASETS.mag.partitions             2{,}000
  protocol.SERVING.memory_caps_gb.0            2\,GiB
  protocol.SEEDS.training.0                    7301

Units follow the names on the path. A name ending in _seconds gives seconds. A name with a gb part
(memory_caps_gb, host_memory_gb_min) gives bytes at 2^30 bytes per unit, since the launcher reads a
cap of 2 as 2G, 2 GiB (memprobe.parse_bytes). A name ending in _pp gives pp. A seed (the SEEDS section,
a name seed or ..._seed) is text, digit strings included, so it prints without thousands separators.
Other integers are counts and other reals numbers. digits is the fewest decimals that print the setting
exactly; a real that needs more than MAX_DIGITS prints with ROUNDED_DIGITS and says so in its note.
Booleans and other strings are skipped.

Derived keys: protocol.STATISTICS.<test>.resamples, the resample count a statistics setting names
(sign_flip_randomization_100k is 100,000); protocol.hash and protocol.hash_short, the protocol hash in
full and its first 12 characters.

The file carries protocol.protocol_hash(), protocol.is_frozen() and split "test": a setting holds for
every split, and build_paper accepts only test numbers. It merges with the reducers' numbers.json under
the same hash, so a paper built after the freeze prints the frozen settings. --list prints each key with
its LaTeX text.
"""

from __future__ import annotations

import argparse
import re
import sys
from decimal import Decimal
from typing import Any, Iterator, Mapping, Sequence

from . import numbers_json, protocol

MAX_DIGITS = 6
ROUNDED_DIGITS = 3
GIB = 1024 ** 3
SHORT_HASH = 12
_RESAMPLES = re.compile(r"_(\d+)k$")


def segment(name: Any) -> str:
    """A key segment for a dict key or list position."""
    text = re.sub(r"(?<=\d)\.(?=\d)", "p", str(name))
    text = re.sub(r"[^A-Za-z0-9_-]", "_", text)
    if not text:
        raise numbers_json.NumbersError(f"protocol name {name!r} gives an empty key segment")
    return text


def exact_digits(value: int | float) -> int | None:
    """The fewest decimals, at most MAX_DIGITS, that print `value` exactly; None when more are needed."""
    d = Decimal(repr(value))
    for digits in range(MAX_DIGITS + 1):
        if d == d.quantize(Decimal(1).scaleb(-digits)):
            return digits
    return None


def _is_seed(names: Sequence[str]) -> bool:
    return names[0] == "SEEDS" or any(n == "seed" or n.endswith("_seed") for n in names)


def unit_of(names: Sequence[str], value: Any) -> str:
    """The unit of a setting from the names on its path (list positions excluded)."""
    if _is_seed(names):
        return "text"
    if any(n.endswith("_seconds") for n in names):
        return "seconds"
    if any(re.search(r"(^|_)gb(_|$)", n) for n in names):
        return "bytes"
    if any(n.endswith("_pp") for n in names):
        return "pp"
    if isinstance(value, int):
        return "count" if value >= 0 else "number"
    return "number"


def leaves(value: Any, path: tuple[str, ...] = (),
           names: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], tuple[str, ...], Any]]:
    """(key path, the dict names on it, value) of every leaf; list positions extend only the path."""
    if isinstance(value, Mapping):
        for name, item in value.items():
            yield from leaves(item, path + (segment(name),), names + (segment(name),))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from leaves(item, path + (str(index),), names)
    else:
        yield path, names, value


def add_setting(nums: numbers_json.Numbers, path: Sequence[str], names: Sequence[str], value: Any) -> str | None:
    """Adds one setting; returns its key, or None for a setting that is not a number."""
    if isinstance(value, bool) or value is None:
        return None
    key = "protocol." + ".".join(path)
    unit = unit_of(names, value)
    if unit == "text":
        if isinstance(value, str) and not value.isdigit():
            return None
        if isinstance(value, float) and value != int(value):
            raise numbers_json.NumbersError(f"{key}: a seed must be an integer, got {value!r}")
        return nums.number(key, str(int(value)) if isinstance(value, float) else str(value), "text")
    if not isinstance(value, (int, float)):
        return None
    digits = exact_digits(value)
    note = None
    if digits is None:
        digits, note = ROUNDED_DIGITS, f"rounded to {ROUNDED_DIGITS} decimals; the protocol holds {value!r}"
    if unit == "count":
        return nums.number(key, value, "count")
    if unit == "bytes":
        if value * GIB != int(value * GIB):
            raise numbers_json.NumbersError(f"{key}: {value!r} GiB is not a whole number of bytes")
        return nums.number(key, int(value * GIB), "bytes", digits=digits, note=note or "GiB (2^30 bytes)")
    return nums.number(key, value, unit, digits=digits, note=note)


def protocol_numbers() -> numbers_json.Numbers:
    """Every protocol setting and the derived keys, under the current protocol hash."""
    spec = protocol.protocol_dict()
    nums = numbers_json.Numbers(split="test")
    for path, names, value in leaves(spec):
        add_setting(nums, path, names, value)
    statistics = spec.get("STATISTICS") or {}
    for name in sorted(statistics):
        match = _RESAMPLES.search(str(statistics[name])) if isinstance(statistics[name], str) else None
        if match:
            nums.number(f"protocol.STATISTICS.{segment(name)}.resamples", int(match.group(1)) * 1000, "count")
    nums.number("protocol.hash", nums.protocol_hash, "text")
    nums.number("protocol.hash_short", nums.protocol_hash[:SHORT_HASH], "text")
    return nums


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="directory for numbers.json")
    parser.add_argument("--list", action="store_true", help="print every key with its LaTeX text")
    args = parser.parse_args(argv)
    if args.out is None and not args.list:
        parser.error("give --out, --list or both")
    nums = protocol_numbers()
    if args.list:
        from .build_paper import format_value

        for key, entry in sorted(nums.numbers.items()):
            print(f"{key}\t{format_value(entry['value'], entry['unit'], entry.get('digits'), key=key)}")
    if args.out is not None:
        path = nums.write(args.out)
        print(f"{len(nums.numbers)} protocol numbers -> {path} (protocol {nums.protocol_hash[:SHORT_HASH]}, "
              f"frozen {nums.frozen})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
