"""memprobe's OOM accounting: the cgroup readers, the scope policy of a systemd cap, and (on Linux with a systemd
user manager) a solver-like child killed at the cap while its parent runs on."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.canonical import memprobe


def _cgroup(path: Path, **files: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (path / name.replace("_", ".")).write_text(text, encoding="ascii")
    return path


def test_effective_memory_max_takes_the_tightest_ancestor(tmp_path):
    root = tmp_path / "cg"
    _cgroup(root / "user.slice", memory_max="max")
    _cgroup(root / "user.slice" / "campaign.scope", memory_max=str(16 << 30))
    leaf = _cgroup(root / "user.slice" / "campaign.scope" / "inner", memory_max=str(32 << 30))
    assert memprobe.effective_memory_max(leaf, root) == 16 << 30
    assert memprobe.effective_memory_max(root / "user.slice", root) is None
    assert memprobe.effective_memory_max(None, root) is None


def test_oom_kills_reads_memory_events(tmp_path):
    cg = _cgroup(tmp_path / "a", memory_events="low 0\nhigh 0\nmax 12\noom 3\noom_kill 2\noom_group_kill 0\n")
    assert memprobe._oom_kills(cg) == 2
    assert memprobe._oom_kills(tmp_path / "missing") is None
    assert memprobe._oom_kills(None) is None


def test_scope_watch_reads_the_scope_once_the_process_has_left_memprobes_cgroup(tmp_path, monkeypatch):
    own = _cgroup(tmp_path / "own", memory_events="oom_kill 5\n")
    scope = _cgroup(tmp_path / "run-1.scope", memory_events="oom_kill 0\n", memory_peak="100", memory_max="4096")
    where = {"path": own}
    monkeypatch.setattr(memprobe, "_pid_cgroup", lambda pid="self": where["path"])
    watch = memprobe._ScopeWatch(123, own)
    watch.poll()
    assert watch.path is None and watch.oom_kills is None  # still in memprobe's cgroup: its kills are not the tree's
    where["path"] = scope
    watch.poll()
    (scope / "memory.events").write_text("oom_kill 1\n", encoding="ascii")
    (scope / "memory.peak").write_text("4096", encoding="ascii")
    watch.poll()
    info = watch.stop()
    assert info == {"cgroup": str(scope), "events.oom_kill": 1, "memory.peak": 4096, "memory.max": "4096"}


def test_systemd_cap_keeps_the_tree_running_after_an_oom_kill(monkeypatch):
    seen = {}

    class Stop(Exception):
        pass

    def popen(cmd, **kwargs):
        seen["cmd"] = cmd
        raise Stop

    monkeypatch.setattr(memprobe.subprocess, "Popen", popen)
    with pytest.raises(Stop):
        memprobe.run_measured([sys.executable, "-c", "pass"], memory_cap="1G", cap_mechanism="systemd")
    cmd = seen["cmd"]
    assert cmd[:4] == ["systemd-run", "--user", "--scope", "--quiet"]
    assert "OOMPolicy=continue" in cmd and "MemorySwapMax=0" in cmd and f"MemoryMax={1 << 30}" in cmd
    assert cmd[cmd.index("--") + 1:] == [sys.executable, "-c", "pass"]


def _systemd_scopes() -> bool:
    if not sys.platform.startswith("linux") or not shutil.which("systemd-run"):
        return False
    probe = subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "-p", "OOMPolicy=continue", "true"],
                           capture_output=True, timeout=20, check=False)
    return probe.returncode == 0


@pytest.mark.skipif(not _systemd_scopes(), reason="needs Linux with a systemd user manager")
def test_child_oom_kill_is_counted_and_the_parent_finishes(tmp_path):
    hog = "b = b'x' * (512 << 20)"
    # the parent outlives the kill by a moment, as a driver that records the failed query does
    parent = ("import subprocess, sys, time; "
              f"r = subprocess.run([sys.executable, '-c', {hog!r}]); "
              "print('child', r.returncode, flush=True); time.sleep(0.5)")
    out = tmp_path / "stdout.txt"
    result = memprobe.run_measured([sys.executable, "-c", parent], memory_cap="160M", cap_mechanism="systemd",
                                   stdout_path=out, timeout=120)
    assert result["exit_code"] == 0, result
    assert out.read_text(encoding="utf-8").strip() == "child -9"
    assert result["oom_kills"] >= 1 and result["oom_killed"] is True
    assert result["cgroup"]["memory.max"] == str(160 << 20)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux cgroups")
def test_uncapped_run_reports_its_enclosing_cgroup():
    result = memprobe.run_measured([sys.executable, "-c", "pass"])
    assert result["exit_code"] == 0 and result["memory_cap"]["mechanism"] == "none"
    if memprobe._pid_cgroup() is not None:
        enclosing = result["enclosing_cgroup"]
        assert enclosing["oom_kills"] in (0, None) and result["oom_killed"] is False
