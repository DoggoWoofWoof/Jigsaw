"""Sealed test seeds: commit before development, reveal after the sealed run.

    python -m scripts.canonical.seal generate   # writes internal/sealed/SEALED_SEEDS.json once
    python -m scripts.canonical.seal commit     # writes the commitment into protocol.SEAL_COMMITMENT
    python -m scripts.canonical.seal verify     # checks the file against the commitment
    python -m scripts.canonical.seal reveal     # prints the file for publication (after the sealed run)

The seeds file lives under internal/, which .git/info/exclude keeps out of git.
Back it up outside the repository: losing it means the committed test split can
never be regenerated. The commitment is SHA-256 over the canonical JSON of the
salt and the three masters, so publishing it reveals nothing about the seeds.

load_test_masters() is the only way the pipeline obtains the test masters. It
requires an explicit unseal flag, a frozen protocol and clean tracked code, and
appends every unseal to internal/sealed/UNSEAL_LOG.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import sys
from pathlib import Path
from typing import Any

from . import manifest, protocol

SEALED_DIR = manifest.REPO_ROOT / "internal" / "sealed"
SEEDS_FILE = SEALED_DIR / "SEALED_SEEDS.json"
UNSEAL_LOG = SEALED_DIR / "UNSEAL_LOG.jsonl"
PROTOCOL_FILE = Path(__file__).resolve().with_name("protocol.py")
MASTER_NAMES = ("test", "reserve_a", "reserve_b")


class SealError(RuntimeError):
    pass


def _committed_part(sealed: dict[str, Any]) -> dict[str, Any]:
    return {"version": int(sealed["version"]), "salt": str(sealed["salt"]), "masters": dict(sealed["masters"])}


def commitment_of(sealed: dict[str, Any]) -> str:
    payload = protocol.canonical_json(_committed_part(sealed)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def generate(path: Path = SEEDS_FILE) -> dict[str, Any]:
    if path.exists():
        raise SealError(f"{path} already exists; refusing to overwrite sealed seeds")
    sealed = {
        "version": 1,
        "created_utc": manifest.utc_now(),
        "salt": secrets.token_hex(32),
        "masters": {name: secrets.token_hex(32) for name in MASTER_NAMES},
        "note": "Jigsaw canonical rerun test split. Do not use before the protocol freeze.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(sealed, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return sealed


def read_sealed(path: Path = SEEDS_FILE) -> dict[str, Any]:
    if not path.exists():
        raise SealError(f"sealed seeds file not found: {path}")
    sealed = json.loads(path.read_text(encoding="utf-8"))
    if set(sealed.get("masters", {})) != set(MASTER_NAMES):
        raise SealError("sealed seeds file has an unexpected set of masters")
    return sealed


def write_commitment(value: str, protocol_file: Path = PROTOCOL_FILE, *, force: bool = False) -> None:
    text = protocol_file.read_text(encoding="utf-8")
    pattern = re.compile(r'^SEAL_COMMITMENT: str \| None = (None|"[0-9a-f]{64}")$', re.M)
    match = pattern.search(text)
    if not match:
        raise SealError("could not find the SEAL_COMMITMENT line in protocol.py")
    current = match.group(1)
    if current != "None" and current.strip('"') != value and not force:
        raise SealError("protocol.py already holds a different commitment; pass --force to replace it")
    new_text = pattern.sub(f'SEAL_COMMITMENT: str | None = "{value}"', text, count=1)
    protocol_file.write_text(new_text, encoding="utf-8")


def verify(path: Path = SEEDS_FILE) -> str:
    if protocol.SEAL_COMMITMENT is None:
        raise SealError("protocol.SEAL_COMMITMENT is not set")
    value = commitment_of(read_sealed(path))
    if value != protocol.SEAL_COMMITMENT:
        raise SealError("sealed seeds do not match protocol.SEAL_COMMITMENT")
    return value


def load_test_masters(*, unseal: bool, reason: str, path: Path = SEEDS_FILE) -> dict[str, str]:
    """Return the test masters. Refuses unless the protocol is frozen and code is clean."""
    if not unseal:
        raise SealError("the test split requires an explicit --unseal")
    verify(path)
    if not protocol.is_frozen():
        raise SealError(
            "protocol is not frozen: set FROZEN_PROTOCOL_HASH to protocol_hash() at the freeze "
            f"(current hash {protocol.protocol_hash()})"
        )
    git = manifest.git_info()
    if not git.get("available") or git.get("dirty_code"):
        raise SealError("tracked code differs from HEAD; commit the frozen code before unsealing")
    sealed = read_sealed(path)
    entry = {
        "utc": manifest.utc_now(),
        "reason": reason,
        "protocol_hash": protocol.protocol_hash(),
        "git_head": git.get("head"),
        "argv": manifest.scrub_argv(sys.argv),
    }
    UNSEAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(UNSEAL_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return dict(sealed["masters"])


def authorize_sealed_run(*, unseal: bool, reason: str) -> dict[str, Any]:
    """Gate for running methods on sealed query sets that were already generated.

    The same checks as load_test_masters() without reading the seeds: an explicit
    unseal flag, a frozen protocol and clean tracked code. Every authorised run
    is appended to the unseal log.
    """
    if not unseal:
        raise SealError("sealed query sets run only with an explicit --unseal")
    if not protocol.is_frozen():
        raise SealError("protocol is not frozen: set FROZEN_PROTOCOL_HASH to protocol_hash() at the freeze "
                        f"(current hash {protocol.protocol_hash()})")
    git = manifest.git_info()
    if not git.get("available") or git.get("dirty_code"):
        raise SealError("tracked code differs from HEAD; commit the frozen code before a sealed run")
    entry = {"utc": manifest.utc_now(), "reason": reason, "event": "sealed_run",
             "protocol_hash": protocol.protocol_hash(), "git_head": git.get("head"),
             "argv": manifest.scrub_argv(sys.argv)}
    UNSEAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(UNSEAL_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def split_masters(split: str, *, unseal: bool = False, reason: str = "") -> list[str]:
    """Masters for a split: dev and validation are public; test is sealed."""
    if split in ("dev", "validation"):
        return [str(seed) for seed in protocol.SEEDS[split]]
    if split in MASTER_NAMES:
        return [load_test_masters(unseal=unseal, reason=reason or split)[split]]
    raise SealError(f"unknown split {split!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["generate", "commit", "verify", "reveal", "status"])
    parser.add_argument("--force", action="store_true", help="replace an existing commitment in protocol.py")
    parser.add_argument("--i-have-finished-the-sealed-run", action="store_true", dest="finished")
    args = parser.parse_args(argv)

    if args.command == "generate":
        generate()
        print(f"wrote {SEEDS_FILE} (back it up outside the repository)")
        return 0
    if args.command == "commit":
        value = commitment_of(read_sealed())
        write_commitment(value, force=args.force)
        print(f"SEAL_COMMITMENT = {value}")
        return 0
    if args.command == "verify":
        print(f"commitment verified: {verify()}")
        return 0
    if args.command == "status":
        print(f"seeds file: {'present' if SEEDS_FILE.exists() else 'absent'} ({SEEDS_FILE})")
        print(f"commitment in protocol.py: {protocol.SEAL_COMMITMENT or 'not set'}")
        print(f"protocol hash: {protocol.protocol_hash()} frozen={protocol.is_frozen()}")
        if UNSEAL_LOG.exists():
            print(f"unseal events: {len(UNSEAL_LOG.read_text(encoding='utf-8').splitlines())}")
        return 0
    if not args.finished:
        print("reveal publishes the test seeds; pass --i-have-finished-the-sealed-run", file=sys.stderr)
        return 2
    verify()
    print(json.dumps(_committed_part(read_sealed()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
