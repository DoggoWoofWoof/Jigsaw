"""Memory, I/O and time measurement for a whole process tree.

run_measured(cmd, ...) runs a command in its own session and measures it from
outside: it samples the summed RSS of the process tree (and the summed PSS on
Linux), kills the tree at a wall-clock limit, optionally runs it under a cgroup
v2 memory cap with swap disabled, and returns the wait4 rusage (CPU time, largest
single-process RSS, block I/O). This is the measurement used for every system
in E2, so the driver, the solver subprocesses and any classical binary are
measured the same way.

Under a cap the kernel's OOM killer takes one process of the tree (usually a
solver) and the rest runs on, under both mechanisms: the systemd scope sets
OOMPolicy=continue, since systemd's default would stop the whole scope after the
first kill. run_measured counts the kills from the cgroup's memory.events
(oom_kills; oom_killed when there was one). An uncapped tree shares memprobe's
own cgroup, so it reports the kills there during the run and the tightest
memory.max above it (a campaign's safety limit, for example).

TreeSampler samples an existing tree from a background thread; the cascade uses
it in-process for stage-level peaks.

evict_files() drops file pages from the page cache (posix_fadvise DONTNEED;
pages still mapped by a live process are not dropped, so release mappings first
with release_mapping()), drop_caches() tries the kernel-wide drop when running
as root, and resident_fraction() reports how much of a file is cached (mincore),
which lets a cold run prove that it started cold.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a hard requirement on measurement machines
    psutil = None

IS_LINUX = sys.platform.startswith("linux")
IS_POSIX = os.name == "posix"
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names else 4096


# ----------------------------------------------------------------------------
# Sizes
# ----------------------------------------------------------------------------
_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?)(i?B?)?\s*$", re.I)


def parse_bytes(text: str | int | float) -> int:
    if isinstance(text, (int, float)):
        return int(text)
    match = _SIZE.match(str(text))
    if not match:
        raise ValueError(f"cannot parse size {text!r}")
    value = float(match.group(1))
    power = " KMGT".index((match.group(2) or " ").upper())
    return int(value * (1024 ** power))


# ----------------------------------------------------------------------------
# Process-tree sampling
# ----------------------------------------------------------------------------
def _read_pss_bytes(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/smaps_rollup", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


@dataclass
class TreeStats:
    samples: int = 0
    peak_rss_bytes: int = 0
    peak_pss_bytes: int | None = None
    peak_process_count: int = 0
    peak_single_rss_bytes: int = 0
    peak_single_name: str = ""
    read_bytes: int = 0
    write_bytes: int = 0
    read_chars: int | None = None
    duration_seconds: float = 0.0
    timeline: list[tuple[float, int]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["timeline"] = [[round(t, 4), rss] for t, rss in self.timeline]
        return out


class TreeSampler:
    """Background sampler of RSS/PSS and I/O counters for a process and its descendants."""

    def __init__(self, root_pid: int, *, interval: float = 0.02, pss_every: int = 10,
                 max_timeline: int = 4000, include_root: bool = True) -> None:
        if psutil is None:
            raise RuntimeError("psutil is required for process-tree sampling")
        self.root_pid = int(root_pid)
        self.interval = float(interval)
        self.pss_every = max(1, int(pss_every))
        self.max_timeline = int(max_timeline)
        self.include_root = include_root
        self.stats = TreeStats()
        self._io_last: dict[int, tuple[int, int, int | None]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._timeline_stride = 1

    def _processes(self) -> list[Any]:
        try:
            root = psutil.Process(self.root_pid)
            procs = root.children(recursive=True)
            if self.include_root:
                procs = [root] + procs
            return procs
        except psutil.Error:
            return []

    def sample_once(self) -> None:
        procs = self._processes()
        total = 0
        count = 0
        for proc in procs:
            try:
                rss = int(proc.memory_info().rss)
            except psutil.Error:
                continue
            total += rss
            count += 1
            if rss > self.stats.peak_single_rss_bytes:
                self.stats.peak_single_rss_bytes = rss
                try:
                    self.stats.peak_single_name = proc.name()
                except psutil.Error:
                    pass
            try:
                io = proc.io_counters()
                self._io_last[proc.pid] = (
                    int(getattr(io, "read_bytes", 0)),
                    int(getattr(io, "write_bytes", 0)),
                    int(io.read_chars) if hasattr(io, "read_chars") else None,
                )
            except (psutil.Error, AttributeError, NotImplementedError):
                pass
        self.stats.samples += 1
        self.stats.peak_rss_bytes = max(self.stats.peak_rss_bytes, total)
        self.stats.peak_process_count = max(self.stats.peak_process_count, count)
        if IS_LINUX and self.stats.samples % self.pss_every == 1:
            pss_values = [_read_pss_bytes(proc.pid) for proc in procs]
            known = [value for value in pss_values if value is not None]
            if known:
                pss = sum(known)
                self.stats.peak_pss_bytes = max(self.stats.peak_pss_bytes or 0, pss)
        if self.stats.samples % self._timeline_stride == 0:
            self.stats.timeline.append((time.perf_counter() - self._t0, total))
            if len(self.stats.timeline) >= self.max_timeline:
                self.stats.timeline = self.stats.timeline[::2]
                self._timeline_stride *= 2

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval)

    def start(self) -> "TreeSampler":
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, name="memprobe-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> TreeStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.stats.duration_seconds = time.perf_counter() - self._t0
        self.stats.read_bytes = sum(v[0] for v in self._io_last.values())
        self.stats.write_bytes = sum(v[1] for v in self._io_last.values())
        chars = [v[2] for v in self._io_last.values() if v[2] is not None]
        self.stats.read_chars = sum(chars) if chars else None
        return self.stats

    def __enter__(self) -> "TreeSampler":
        return self.start()

    def __exit__(self, *exc: Any) -> bool:
        self.stop()
        return False


def self_tree_sampler(interval: float = 0.02) -> TreeSampler:
    """Sampler for the current process and its children (in-process stage peaks)."""
    return TreeSampler(os.getpid(), interval=interval)


# ----------------------------------------------------------------------------
# Page cache control
# ----------------------------------------------------------------------------
def release_mapping(array: Any) -> bool:
    """madvise(DONTNEED) on a numpy memmap so its pages can then be evicted."""
    mm = getattr(array, "_mmap", None)
    if mm is None:
        base = getattr(array, "base", None)
        mm = getattr(base, "_mmap", None) if base is not None else None
        if mm is None and base is not None and base.__class__.__name__ == "mmap":
            mm = base
    if mm is None or not hasattr(mm, "madvise"):
        return False
    try:
        import mmap as _mmap

        mm.madvise(_mmap.MADV_DONTNEED)
        return True
    except (AttributeError, OSError, ValueError):
        return False


def evict_files(paths: Iterable[str | os.PathLike]) -> dict[str, Any]:
    """Drop the page-cache pages of files (clean, unmapped pages only)."""
    report: dict[str, Any] = {"method": "posix_fadvise_dontneed", "files": 0, "bytes": 0, "errors": []}
    if not hasattr(os, "posix_fadvise"):
        report["method"] = "unsupported"
        return report
    for path in paths:
        path = Path(path)
        files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
        for file in files:
            try:
                fd = os.open(file, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
                report["files"] += 1
                report["bytes"] += file.stat().st_size
            except OSError as exc:
                report["errors"].append(f"{file}: {exc}")
    return report


def drop_caches() -> dict[str, Any]:
    """Kernel-wide page-cache drop. Needs root; never prompts for a password."""
    if not IS_LINUX:
        return {"dropped": False, "reason": "not linux"}
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w", encoding="ascii") as handle:
            handle.write("1\n")
        return {"dropped": True}
    except OSError as exc:
        return {"dropped": False, "reason": str(exc)}


def resident_fraction(path: str | os.PathLike) -> float | None:
    """Fraction of a file's pages in the page cache (Linux mincore), or None."""
    if not IS_LINUX:
        return None
    path = Path(path)
    size = path.stat().st_size
    if size == 0:
        return 0.0
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    prot_read, map_shared = 0x1, 0x01
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, prot_read, map_shared, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            return None
        try:
            pages = (size + PAGE_SIZE - 1) // PAGE_SIZE
            vec = (ctypes.c_ubyte * pages)()
            if libc.mincore(ctypes.c_void_p(addr), size, vec) != 0:
                return None
            resident = sum(1 for byte in vec if byte & 1)
            return resident / pages
        finally:
            libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)


# ----------------------------------------------------------------------------
# Memory caps
# ----------------------------------------------------------------------------
def _cgroup_root() -> Path | None:
    if not IS_LINUX:
        return None
    try:
        text = Path("/proc/self/cgroup").read_text(encoding="ascii")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("0::"):
            rel = line[3:].strip().lstrip("/")
            base = Path("/sys/fs/cgroup") / rel
            return base if base.exists() else None
    return None


def detect_cap_mechanism() -> str:
    """'systemd', 'cgroup', or 'none'. rlimit_as is never chosen automatically."""
    if not IS_LINUX:
        return "none"
    if shutil.which("systemd-run"):
        probe = subprocess.run(
            ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=64M", "true"],
            capture_output=True,
            timeout=20,
            check=False,
        )
        if probe.returncode == 0:
            return "systemd"
    root = _cgroup_root()
    if root is not None and os.access(root, os.W_OK):
        controllers = root / "cgroup.subtree_control"
        try:
            if "memory" in controllers.read_text(encoding="ascii") or os.access(controllers, os.W_OK):
                return "cgroup"
        except OSError:
            pass
    return "none"


class _CgroupCap:
    """A child cgroup under the current (delegated) cgroup with memory.max set."""

    def __init__(self, cap_bytes: int) -> None:
        root = _cgroup_root()
        if root is None:
            raise RuntimeError("no writable cgroup v2 hierarchy")
        self.path = root / f"jigsaw-cap-{uuid.uuid4().hex[:12]}"
        self.path.mkdir()
        (self.path / "memory.max").write_text(str(int(cap_bytes)), encoding="ascii")
        swap = self.path / "memory.swap.max"
        if swap.exists():
            swap.write_text("0", encoding="ascii")

    def preexec(self) -> None:
        with open(self.path / "cgroup.procs", "w", encoding="ascii") as handle:
            handle.write(str(os.getpid()))

    def read(self) -> dict[str, Any]:
        out: dict[str, Any] = {"cgroup": str(self.path)}
        for name in ("memory.peak", "memory.max", "memory.swap.max"):
            file = self.path / name
            if file.exists():
                out[name] = file.read_text(encoding="ascii").strip()
        events = self.path / "memory.events"
        if events.exists():
            for line in events.read_text(encoding="ascii").splitlines():
                key, value = line.split()
                out[f"events.{key}"] = int(value)
        return out

    def remove(self) -> None:
        for _ in range(50):
            try:
                self.path.rmdir()
                return
            except OSError:
                time.sleep(0.1)


def _pid_cgroup(pid: int | str = "self") -> Path | None:
    """The cgroup v2 directory of a process, or None."""
    try:
        text = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("0::"):
            path = Path("/sys/fs/cgroup") / line[3:].strip().lstrip("/")
            return path if path.exists() else None
    return None


def _oom_kills(cgroup: Path | None) -> int | None:
    """The oom_kill counter of a cgroup's memory.events, or None when unreadable."""
    if cgroup is None:
        return None
    try:
        for line in (cgroup / "memory.events").read_text(encoding="ascii").splitlines():
            key, value = line.split()
            if key == "oom_kill":
                return int(value)
    except (OSError, ValueError):
        return None
    return None


def effective_memory_max(cgroup: Path | None, root: Path = Path("/sys/fs/cgroup")) -> int | None:
    """The tightest memory.max of a cgroup and its ancestors in bytes, or None when none is set."""
    best = None
    path = cgroup
    while path is not None and root in path.parents:
        try:
            text = (path / "memory.max").read_text(encoding="ascii").strip()
        except OSError:
            text = "max"
        if text.isdigit():
            best = int(text) if best is None else min(best, int(text))
        path = path.parent
    return best


class _ScopeWatch:
    """Polls the cgroup of a process that systemd-run moves into a scope of its own.

    The scope is removed with its last process, so its memory.events and memory.peak are read while the tree
    runs: every `interval` seconds, from the moment the process has left memprobe's own cgroup. A kill in the
    last interval before the root exits can go unseen; a root killed by the OOM killer is caught by its signal."""

    def __init__(self, pid: int, own: Path | None, interval: float = 0.02) -> None:
        self.pid = int(pid)
        self.own = own
        self.interval = float(interval)
        self.path: Path | None = None
        self.oom_kills: int | None = None
        self.memory_peak: int | None = None
        self.memory_max: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll(self) -> None:
        path = _pid_cgroup(self.pid)
        if path is None or path == self.own:
            return
        self.path = path
        kills = _oom_kills(path)
        if kills is not None:
            self.oom_kills = max(self.oom_kills or 0, kills)
        try:
            self.memory_peak = max(self.memory_peak or 0, int((path / "memory.peak").read_text(encoding="ascii")))
            self.memory_max = (path / "memory.max").read_text(encoding="ascii").strip()
        except (OSError, ValueError):
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.poll()
            self._stop.wait(self.interval)

    def start(self) -> "_ScopeWatch":
        self._thread = threading.Thread(target=self._loop, name="memprobe-scope", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return {"cgroup": str(self.path) if self.path else None, "events.oom_kill": self.oom_kills,
                "memory.peak": self.memory_peak, "memory.max": self.memory_max}


def _rlimit_preexec(cap_bytes: int):
    def apply() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (cap_bytes, cap_bytes))

    return apply


# ----------------------------------------------------------------------------
# Measured execution
# ----------------------------------------------------------------------------
def _kill_tree(proc: subprocess.Popen) -> None:
    if IS_POSIX:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
    if psutil is not None:
        try:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
            return
        except psutil.Error:
            pass
    try:
        proc.kill()
    except OSError:
        pass


def run_measured(
    cmd: Sequence[str],
    *,
    timeout: float | None = None,
    memory_cap: str | int | None = None,
    cap_mechanism: str = "auto",
    evict: Iterable[str | os.PathLike] = (),
    drop_all_caches: bool = False,
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike | None = None,
    stdout_path: str | os.PathLike | None = None,
    stderr_path: str | os.PathLike | None = None,
    interval: float = 0.02,
) -> dict[str, Any]:
    """Run cmd and measure its whole process tree. Returns a JSON-serialisable dict."""
    result: dict[str, Any] = {"cmd": list(cmd), "timeout_seconds": timeout}
    evict = list(evict)
    if drop_all_caches:
        result["drop_caches"] = drop_caches()
    if evict:
        result["evict"] = evict_files(evict)
        fractions = {}
        for path in evict:
            path = Path(path)
            files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
            for file in files[:64]:
                fractions[str(file)] = resident_fraction(file)
        result["resident_fraction_after_evict"] = fractions

    cap_bytes = parse_bytes(memory_cap) if memory_cap else None
    mechanism = "none"
    preexec = None
    cgroup: _CgroupCap | None = None
    full_cmd = list(cmd)
    if cap_bytes:
        mechanism = detect_cap_mechanism() if cap_mechanism == "auto" else cap_mechanism
        if mechanism == "systemd":
            full_cmd = [
                "systemd-run", "--user", "--scope", "--quiet",
                "-p", f"MemoryMax={cap_bytes}", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue", "--",
            ] + full_cmd
        elif mechanism == "cgroup":
            cgroup = _CgroupCap(cap_bytes)
            preexec = cgroup.preexec
        elif mechanism == "rlimit_as":
            preexec = _rlimit_preexec(cap_bytes)
        else:
            raise RuntimeError("a memory cap was requested but no cap mechanism is available")
    result["memory_cap"] = {"bytes": cap_bytes, "mechanism": mechanism,
                            "canonical": mechanism in ("systemd", "cgroup") if cap_bytes else None}

    stdout = open(stdout_path, "wb") if stdout_path else subprocess.DEVNULL
    stderr = open(stderr_path, "wb") if stderr_path else subprocess.DEVNULL
    popen_kwargs: dict[str, Any] = {
        "stdout": stdout,
        "stderr": stderr,
        "env": dict(env) if env is not None else None,
        "cwd": str(cwd) if cwd else None,
    }
    if IS_POSIX:
        popen_kwargs["start_new_session"] = True
        if preexec is not None:
            popen_kwargs["preexec_fn"] = preexec
    else:
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    own_cgroup = _pid_cgroup() if IS_LINUX else None
    own_kills = _oom_kills(own_cgroup)
    timed_out = threading.Event()
    t0 = time.perf_counter()
    proc = subprocess.Popen(full_cmd, **popen_kwargs)
    sampler = TreeSampler(proc.pid, interval=interval).start()
    scope = _ScopeWatch(proc.pid, own_cgroup, interval).start() if mechanism == "systemd" and IS_LINUX else None

    def watchdog() -> None:
        if timeout is None:
            return
        if not done.wait(timeout):
            timed_out.set()
            _kill_tree(proc)

    done = threading.Event()
    guard = threading.Thread(target=watchdog, name="memprobe-watchdog", daemon=True)
    guard.start()
    rusage = None
    try:
        if hasattr(os, "wait4"):
            _, status, rusage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
        else:
            proc.wait()
    finally:
        wall = time.perf_counter() - t0
        done.set()
        stats = sampler.stop()
        scope_info = scope.stop() if scope is not None else None
        for handle in (stdout, stderr):
            if hasattr(handle, "close"):
                handle.close()

    result.update({
        "exit_code": proc.returncode,
        "signal": -proc.returncode if proc.returncode is not None and proc.returncode < 0 else None,
        "timed_out": timed_out.is_set(),
        "wall_seconds": wall,
        "tree": stats.as_dict(),
    })
    if rusage is not None:
        maxrss_unit = 1024 if IS_LINUX else 1
        result["rusage"] = {
            "user_seconds": rusage.ru_utime,
            "system_seconds": rusage.ru_stime,
            "max_single_process_rss_bytes": int(rusage.ru_maxrss) * maxrss_unit,
            "block_input_bytes": int(rusage.ru_inblock) * 512,
            "block_output_bytes": int(rusage.ru_oublock) * 512,
            "major_faults": int(rusage.ru_majflt),
            "minor_faults": int(rusage.ru_minflt),
        }
    # a SIGKILL of the root that the watchdog did not send is taken for the OOM killer's
    root_killed = bool(IS_POSIX and result.get("signal") == signal.SIGKILL and not result["timed_out"])
    if cgroup is not None:
        info = cgroup.read()
        result["cgroup"] = info
        result["oom_kills"] = int(info.get("events.oom_kill", 0))
        result["oom_killed"] = result["oom_kills"] > 0
        cgroup.remove()
    elif scope_info is not None:
        result["cgroup"] = scope_info
        result["oom_kills"] = scope_info["events.oom_kill"]
        result["oom_killed"] = bool(result["oom_kills"]) or root_killed
    elif cap_bytes:
        result["oom_killed"] = root_killed if IS_POSIX else None
    elif own_cgroup is not None:
        # uncapped: the tree ran in memprobe's own cgroup, under whatever limit an enclosing one sets
        after = _oom_kills(own_cgroup)
        kills = after - own_kills if after is not None and own_kills is not None else None
        result["enclosing_cgroup"] = {"cgroup": str(own_cgroup), "memory_max_bytes": effective_memory_max(own_cgroup),
                                      "oom_kills": kills}
        result["oom_kills"] = kills
        result["oom_killed"] = bool(kills) or root_killed
    result["peak_rss_bytes"] = max(
        stats.peak_rss_bytes,
        result.get("rusage", {}).get("max_single_process_rss_bytes", 0),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a command under measurement: memprobe run [opts] -- cmd ...")
    run.add_argument("--out", required=True)
    run.add_argument("--timeout", type=float, default=None)
    run.add_argument("--memory-cap", default=None, help="e.g. 4G")
    run.add_argument("--cap-mechanism", default="auto", choices=["auto", "systemd", "cgroup", "rlimit_as"])
    run.add_argument("--evict", action="append", default=[], help="file or directory to evict first")
    run.add_argument("--drop-caches", action="store_true")
    run.add_argument("--stdout", default=None)
    run.add_argument("--stderr", default=None)
    run.add_argument("--interval", type=float, default=0.02)
    run.add_argument("cmd", nargs=argparse.REMAINDER)
    ev = sub.add_parser("evict", help="evict files from the page cache")
    ev.add_argument("paths", nargs="+")
    rf = sub.add_parser("resident", help="report the cached fraction of files")
    rf.add_argument("paths", nargs="+")
    sub.add_parser("caps", help="report the available memory-cap mechanism")
    args = parser.parse_args(argv)

    if args.command == "run":
        cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
        if not cmd:
            parser.error("no command given")
        result = run_measured(
            cmd,
            timeout=args.timeout,
            memory_cap=args.memory_cap,
            cap_mechanism=args.cap_mechanism,
            evict=args.evict,
            drop_all_caches=args.drop_caches,
            stdout_path=args.stdout,
            stderr_path=args.stderr,
            interval=args.interval,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"exit={result['exit_code']} timed_out={result['timed_out']} "
              f"wall={result['wall_seconds']:.3f}s peak_rss={result['peak_rss_bytes'] / 2**20:.1f} MiB")
        return 0
    if args.command == "evict":
        print(json.dumps(evict_files(args.paths), indent=2))
        return 0
    if args.command == "resident":
        for path in args.paths:
            print(f"{path}\t{resident_fraction(path)}")
        return 0
    print(detect_cap_mechanism())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
