"""Run manifests: what ran, on which inputs, with which code, on which machine.

Every canonical run directory holds manifest.json. It records the git state, the
protocol hash, SHA-256 of every input artifact, solver binaries, library
versions, the machine (hostname hashed), an allow-listed environment, start and
end times, and the hashes of every output file. It never records credentials:
environment variables are allow-listed, and argv values that follow a
secret-looking flag are redacted.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata as _metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_NAME = "manifest.json"
CODE_PATHS = ("scripts", "src", "tests", "external/patches")

_SECRET_NAME = re.compile(r"(TOKEN|SECRET|PASSW|PASSWD|CREDENTIAL|AUTH|API_?KEY|PRIVATE|COOKIE|SESSION)", re.I)
_ENV_ALLOW = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "JIGSAW_DATA_ROOT",
    "JIGSAW_STORE_ROOT",
    "JIGSAW_RUNS_ROOT",
    "GLASGOW_SOLVER_BIN",
    "SUBGRAPHMATCHING_BIN",
    "GNNPE_BIN",
    "GPMETIS_BIN",
    "KAHIP_BIN",
    "RX_JOB_ID",
)
_LIBRARIES = (
    "numpy",
    "scipy",
    "torch",
    "torch-geometric",
    "torch-sparse",
    "torch-scatter",
    "faiss-cpu",
    "faiss-gpu",
    "networkx",
    "psutil",
    "pymetis",
    "ogb",
    "pandas",
)
FAST_FINGERPRINT_BYTES = 8 << 20
FULL_HASH_LIMIT_BYTES = 2 << 30


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 30.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, str(exc)
    return proc.returncode, proc.stdout.decode("utf-8", "replace")


def code_fingerprint(root: Path = REPO_ROOT, code_paths: Iterable[str] = CODE_PATHS) -> dict[str, Any]:
    """SHA-256 of every file under code_paths with CRLF folded to LF, and one digest of that list.

    Folding makes a Windows checkout (core.autocrlf) and a copy of it on another machine hash
    like the git blobs, so code_check can compare a run made without git against a commit.
    """
    root = Path(root)
    files: dict[str, str] = {}
    for base in code_paths:
        top = root / base
        found = [top] if top.is_file() else sorted(p for p in top.rglob("*") if p.is_file()) if top.is_dir() else []
        for path in found:
            rel = path.relative_to(root).as_posix()
            if "__pycache__" in rel.split("/") or path.suffix in (".pyc", ".pyo"):
                continue
            files[rel] = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    listing = "".join(f"{rel}\0{sha}\n" for rel, sha in sorted(files.items()))
    return {"files": files, "tree_sha256": hashlib.sha256(listing.encode("utf-8")).hexdigest(),
            "crlf_folded": True, "code_paths": list(code_paths)}


def git_info(repo_root: Path = REPO_ROOT, code_paths: Iterable[str] = CODE_PATHS) -> dict[str, Any]:
    """HEAD, branch, and whether tracked or untracked code differs from HEAD.

    Without git (a copy of the code on another machine) it records code_fingerprint instead;
    code_check compares that against a commit.
    """
    code, sha = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if code != 0:
        return {"available": False, "code_fingerprint": code_fingerprint(repo_root, code_paths)}
    _, branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    paths = [p for p in code_paths if (repo_root / p).exists()]
    _, status = _run(["git", "status", "--porcelain", "--untracked-files=all", "--", *paths], cwd=repo_root)
    _, diff = _run(["git", "diff", "HEAD", "--binary", "--", *paths], cwd=repo_root, timeout=120.0)
    untracked = sorted(line[3:] for line in status.splitlines() if line.startswith("?? "))
    digest = hashlib.sha256(diff.encode("utf-8", "replace"))
    for rel in untracked:
        path = repo_root / rel
        digest.update(rel.encode("utf-8"))
        if path.is_file():
            digest.update(sha256_file(path)["sha256"].encode("ascii"))
    return {
        "available": True,
        "head": sha.strip(),
        "branch": branch.strip(),
        "dirty_code": bool(status.strip()),
        "dirty_entries": len(status.splitlines()),
        "untracked_code_files": untracked[:200],
        "code_diff_sha256": digest.hexdigest(),
    }


def sha256_file(path: str | os.PathLike, *, fast: bool = False) -> dict[str, Any]:
    """Full SHA-256, or for fast=True a fingerprint of size plus head, middle and tail."""
    path = Path(path)
    size = path.stat().st_size
    digest = hashlib.sha256()
    mode = "sha256"
    with open(path, "rb") as handle:
        if fast and size > 3 * FAST_FINGERPRINT_BYTES:
            mode = "fingerprint_head_mid_tail_8MiB"
            digest.update(str(size).encode("ascii"))
            for offset in (0, size // 2, size - FAST_FINGERPRINT_BYTES):
                handle.seek(offset)
                digest.update(handle.read(FAST_FINGERPRINT_BYTES))
        else:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return {"path": str(path), "bytes": size, mode if mode == "sha256" else "fingerprint": digest.hexdigest(), "mode": mode}


def hash_inputs(inputs: Mapping[str, str | os.PathLike | None], *, fast: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, path in inputs.items():
        if path is None:
            out[name] = None
            continue
        path = Path(path)
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file())
            entries = [sha256_file(p, fast=fast) for p in files]
            combined = hashlib.sha256()
            for entry in entries:
                combined.update(str(Path(entry["path"]).relative_to(path)).encode("utf-8"))
                combined.update(str(entry.get("sha256") or entry.get("fingerprint")).encode("ascii"))
            out[name] = {"path": str(path), "files": len(entries), "tree_sha256": combined.hexdigest(),
                         "mode": "fast" if fast else "full"}
        elif path.is_file():
            out[name] = sha256_file(path, fast=fast)
        else:
            out[name] = {"path": str(path), "missing": True}
    return out


def machine_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname_sha256_12": hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()[:12],
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "logical_cpus": os.cpu_count(),
        "cpu_model": platform.processor() or "",
    }
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import psutil

        info["ram_bytes"] = int(psutil.virtual_memory().total)
        info["physical_cpus"] = psutil.cpu_count(logical=False)
    except Exception:
        pass
    try:
        affinity = os.sched_getaffinity(0)  # type: ignore[attr-defined]
        info["cpu_affinity_count"] = len(affinity)
    except (AttributeError, OSError):
        pass
    cgroup_max = Path("/sys/fs/cgroup/memory.max")
    if cgroup_max.exists():
        try:
            info["cgroup_memory_max"] = cgroup_max.read_text().strip()
        except OSError:
            pass
    code, out = _run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        timeout=10.0,
    )
    if code == 0 and out.strip():
        info["gpus"] = [line.strip() for line in out.strip().splitlines()]
    return info


def library_versions(names: Iterable[str] = _LIBRARIES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = _metadata.version(name)
        except _metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def allowed_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    environ = os.environ if environ is None else environ
    return {
        name: environ[name]
        for name in _ENV_ALLOW
        if name in environ and not _SECRET_NAME.search(name)
    }


def scrub_argv(argv: Iterable[str]) -> list[str]:
    """Redact any value that follows, or is attached to, a secret-looking flag."""
    out: list[str] = []
    redact_next = False
    for arg in argv:
        if redact_next:
            out.append("<redacted>")
            redact_next = False
            continue
        if arg.startswith("-") and _SECRET_NAME.search(arg):
            if "=" in arg:
                out.append(arg.split("=", 1)[0] + "=<redacted>")
            else:
                out.append(arg)
                redact_next = True
            continue
        out.append(arg)
    return out


def solver_info(path: str | os.PathLike | None, pins_file: str | os.PathLike | None = None) -> dict[str, Any]:
    if not path:
        return {"available": False}
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "available": False}
    info: dict[str, Any] = {"path": str(path), "available": True, **sha256_file(path)}
    pins = Path(pins_file) if pins_file else REPO_ROOT / "external" / "PINS.json"
    if pins.exists():
        try:
            info["pins"] = json.loads(pins.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            info["pins"] = "unreadable"
    return info


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    os.replace(tmp, path)


def write_manifest(
    run_dir: str | os.PathLike,
    *,
    experiment: str,
    config: Mapping[str, Any],
    inputs: Mapping[str, str | os.PathLike | None] | None = None,
    solvers: Mapping[str, str | os.PathLike | None] | None = None,
    split: str = "dev",
    fast_input_hashes: bool = False,
    argv: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Create run_dir/manifest.json with status 'running'. Returns the manifest."""
    from . import protocol

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "jigsaw-canonical-manifest-v1",
        "experiment": experiment,
        "split": split,
        "status": "running",
        "started_utc": utc_now(),
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_hash": protocol.protocol_hash(),
        "protocol_frozen": protocol.is_frozen(),
        "seal_commitment": protocol.SEAL_COMMITMENT,
        "git": git_info(),
        "argv": scrub_argv(sys.argv if argv is None else argv),
        "config": dict(config),
        "inputs": hash_inputs(inputs or {}, fast=fast_input_hashes),
        "solvers": {name: solver_info(path) for name, path in (solvers or {}).items()},
        "machine": machine_info(),
        "libraries": library_versions(),
        "environment": allowed_environment(),
    }
    _atomic_write_json(run_dir / MANIFEST_NAME, manifest)
    return manifest


def output_hashes(run_dir: Path) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        if path.name in {MANIFEST_NAME, MANIFEST_NAME + ".tmp"}:
            continue
        rel = str(path.relative_to(run_dir)).replace(os.sep, "/")
        fast = path.stat().st_size > FULL_HASH_LIMIT_BYTES
        entries[rel] = sha256_file(path, fast=fast)
        entries[rel].pop("path", None)
    return entries


def finalize_manifest(
    run_dir: str | os.PathLike,
    *,
    status: str = "complete",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    path = run_dir / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = status
    manifest["finished_utc"] = utc_now()
    if extra:
        manifest.setdefault("extra", {}).update(dict(extra))
    manifest["outputs"] = output_hashes(run_dir)
    _atomic_write_json(path, manifest)
    return manifest


class RunManifest:
    """Context manager: writes the manifest on entry and finalizes it on exit."""

    def __init__(self, run_dir: str | os.PathLike, **kwargs: Any) -> None:
        self.run_dir = Path(run_dir)
        self.kwargs = kwargs
        self.extra: dict[str, Any] = {}
        self.manifest: dict[str, Any] | None = None
        self._t0 = 0.0

    def __enter__(self) -> "RunManifest":
        self._t0 = time.perf_counter()
        self.manifest = write_manifest(self.run_dir, **self.kwargs)
        return self

    def note(self, **values: Any) -> None:
        self.extra.update(values)

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.extra["wall_seconds"] = time.perf_counter() - self._t0
        if exc_type is None:
            finalize_manifest(self.run_dir, status="complete", extra=self.extra)
        elif exc_type is KeyboardInterrupt:
            finalize_manifest(self.run_dir, status="interrupted", extra=self.extra)
        else:
            self.extra["error"] = f"{exc_type.__name__}: {exc}"
            finalize_manifest(self.run_dir, status="failed", extra=self.extra)
        return False


def verify_manifest(run_dir: str | os.PathLike, *, expect_protocol_hash: str | None = None) -> list[str]:
    """Re-hash outputs and compare with the manifest. Returns a list of problems."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    problems: list[str] = []
    if manifest.get("status") != "complete":
        problems.append(f"status={manifest.get('status')}")
    if expect_protocol_hash and manifest.get("protocol_hash") != expect_protocol_hash:
        problems.append("protocol hash differs from the expected one")
    recorded = manifest.get("outputs", {})
    current = output_hashes(run_dir)
    for rel, entry in recorded.items():
        now = current.get(rel)
        if now is None:
            problems.append(f"missing output {rel}")
        elif now.get("sha256", now.get("fingerprint")) != entry.get("sha256", entry.get("fingerprint")):
            problems.append(f"changed output {rel}")
    for rel in sorted(set(current) - set(recorded)):
        problems.append(f"unrecorded output {rel}")
    return problems


def code_check(path: str | os.PathLike, rev: str = "HEAD", repo_root: Path = REPO_ROOT) -> list[str]:
    """Problems when the code a run recorded without git differs from commit `rev` of this repository.

    path is a run directory or its manifest.json. Every file of the run's code_fingerprint must
    exist in rev with the same content (CRLF folded to LF on both sides).
    """
    path = Path(path)
    data = json.loads((path / MANIFEST_NAME if path.is_dir() else path).read_text(encoding="utf-8"))
    git = data.get("git") or {}
    fp = git.get("code_fingerprint")
    if git.get("available"):
        return [] if not git.get("dirty_code") else [f"run recorded dirty code at {git.get('head')}"]
    if not fp:
        return ["the run recorded neither git state nor a code fingerprint"]
    code, full = _run(["git", "rev-parse", "--verify", f"{rev}^{{commit}}"], cwd=repo_root)
    if code != 0:
        return [f"unknown revision {rev}"]
    problems = []
    for rel, digest in sorted(fp["files"].items()):
        try:
            blob = subprocess.run(["git", "show", f"{full.strip()}:{rel}"], cwd=str(repo_root), capture_output=True,
                                  timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            problems.append(f"{rel}: {exc}")
            continue
        if blob.returncode != 0:
            problems.append(f"{rel}: not in {rev}")
        elif hashlib.sha256(blob.stdout.replace(b"\r\n", b"\n")).hexdigest() != digest:
            problems.append(f"{rel}: differs from {rev}")
    return problems


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("verify", help="re-hash a run directory against its manifest")
    check.add_argument("run_dir")
    check.add_argument("--protocol-hash", default=None)
    sub.add_parser("machine", help="print machine, library and git info")
    same = sub.add_parser("code-check", help="compare a run's recorded code (made without git) with a commit")
    same.add_argument("run_dir", help="run directory or manifest.json")
    same.add_argument("--rev", default="HEAD")
    args = parser.parse_args()
    if args.cmd == "verify":
        issues = verify_manifest(args.run_dir, expect_protocol_hash=args.protocol_hash)
        print("OK" if not issues else "\n".join(issues))
        raise SystemExit(1 if issues else 0)
    if args.cmd == "code-check":
        issues = code_check(args.run_dir, args.rev)
        print("OK" if not issues else "\n".join(issues))
        raise SystemExit(1 if issues else 0)
    print(json.dumps({"machine": machine_info(), "libraries": library_versions(), "git": git_info()}, indent=2))
