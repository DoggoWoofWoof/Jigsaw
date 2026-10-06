"""lowprio: scheduler arithmetic against rx's, contention, yielding, requeue through a waiter, code pinning."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

psutil = pytest.importorskip("psutil")

from scripts.canonical.launch import lowprio as lp  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CAP = {"v": 1, "cpus": 32, "mem_gb": 127.7, "gpus": [{"index": 0, "name": "GPU", "mem_gb": 24.0}],
       "reserve_cpus": 2, "reserve_mem_gb": 8, "hold_after_s": 1800}
GPU_ASSIGN = {"cpus": 8.0, "mem_gb": 32.0, "gpus": [0], "gpu_shares": {"0": 1.0}}
FAKE_AGENT = r'''
import json, os, sys, time
home = os.environ["RX_HOME"]
req = json.loads(sys.stdin.readline())
with open(os.path.join(home, "rpc.jsonl"), "a", encoding="utf-8") as f:
    f.write(json.dumps(req) + "\n")
if req["op"] == "launch":
    spec = req["spec"]
    jd = os.path.join(home, "projects", spec["project"], "jobs", spec["id"])
    os.makedirs(jd, exist_ok=True)
    json.dump(spec, open(os.path.join(jd, "spec.json"), "w"))
    state = os.environ.get("FAKE_RX_STATE", "running")
    json.dump({"id": spec["id"], "state": state}, open(os.path.join(jd, "status.json"), "w"))
    print(json.dumps({"ok": True, "existing": False, "job": {"id": spec["id"], "state": state}}))
elif req["op"] == "cancel":
    jd = os.path.join(home, "projects", req["project"], "jobs", req["id"])
    json.dump({"id": req["id"], "state": "cancelled"}, open(os.path.join(jd, "status.json"), "w"))
    print(json.dumps({"ok": True, "job": {"id": req["id"], "state": "cancelled"}}))
else:
    print(json.dumps({"ok": False, "error": "unsupported", "kind": "protocol"}))
'''
CHILD = r'''
import pathlib, sys, time
d = pathlib.Path(sys.argv[1]); seconds = float(sys.argv[2])
(d / "started").write_text("1")
time.sleep(seconds)
(d / "finished").write_text("1")
sys.exit(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
'''
GPU_XML = """<?xml version="1.0" ?>
<nvidia_smi_log>
  <gpu id="00000000:01:00.0"><processes>
    <process_info><pid>111</pid><type>C</type><process_name>C:\\x\\python.exe</process_name></process_info>
    <process_info><pid>222</pid><type>C+G</type><process_name>C:\\Windows\\explorer.exe</process_name></process_info>
    <process_info><pid>333</pid><type>G</type><process_name>dwm.exe</process_name></process_info>
  </processes></gpu>
  <gpu id="00000000:02:00.0"><processes>
    <process_info><pid>444</pid><type>C</type><process_name>/usr/bin/python3</process_name></process_info>
    <process_info><pid>bad</pid><type>C</type><process_name>x</process_name></process_info>
  </processes></gpu>
</nvidia_smi_log>"""


# ───────────────────────────── fixtures ─────────────────────────────

def entry(project, jid, req, *, home, kind, assign=None, enq=None, admitted=None, alive=True):
    jd = home / "projects" / project / "jobs" / jid
    jd.mkdir(parents=True, exist_ok=True)
    now = time.time()
    (jd / "heartbeat.json").write_text(json.dumps({"t": now if alive else now - 600}), encoding="utf-8")
    e = {"key": f"{project}--{jid}", "project": project, "id": jid, "jd": str(jd), "req": req,
         "enq": now - 100 if enq is None else enq}
    if kind == "active":
        e.update(assign=assign or {"cpus": req["cpus"], "mem_gb": req["mem_gb"], "gpus": [], "gpu_shares": {}},
                 admitted=now - 50 if admitted is None else admitted)
    (home / kind).mkdir(parents=True, exist_ok=True)
    (home / kind / f"{project}--{jid}.json").write_text(json.dumps(e), encoding="utf-8")
    return e


def req(cpus, mem, gpus=0.0):
    return {"cpus": float(cpus), "mem_gb": float(mem), "gpus": float(gpus)}


@pytest.fixture()
def host(tmp_path, monkeypatch):
    """An rx home with this job (jigsaw, 1 GPU) running, a fake agent, a workspace and a campaign dir."""
    home = tmp_path / "rxhome"
    (home / "agent").mkdir(parents=True)
    (home / "host.json").write_text(json.dumps(CAP), encoding="utf-8")
    (home / "agent" / "rx_agent-fake00000000.py").write_text(FAKE_AGENT, encoding="utf-8")
    ws = tmp_path / "ws"
    (ws / "scripts").mkdir(parents=True)
    (ws / "scripts" / "code.py").write_bytes(b"x = 1\n")
    (ws / "child.py").write_text(CHILD, encoding="utf-8")
    monkeypatch.chdir(ws)
    monkeypatch.setenv("JIGSAW_RX_PYTHON", sys.executable)
    monkeypatch.setattr(lp, "lower_priority", lambda: "unchanged (test)")
    jid = "260929-120000-cal-ab12"
    me = entry("jigsaw", jid, req(8, 32, 1), home=home, kind="active", assign=GPU_ASSIGN)
    argv = ["python", "-m", "scripts.canonical.launch.lowprio", "run", "--requeue", "--", "python", "child.py"]
    spec = {"project": "jigsaw", "id": jid, "name": "cal", "ws": "ws", "shell": "exec", "argv": argv,
            "command": None, "env": "jigsaw-cu128", "cpus": 8.0, "mem_gb": 32.0, "gpus": 1.0, "mem_hard_gb": None,
            "outputs": ["outputs/**"], "outputs_exclude": [], "env_vars": {"PYTHONHASHSEED": "0"}, "cwd": ".",
            "timeout_s": None, "display": " ".join(argv), "code": {"available": True}, "created": time.time(),
            "agent": "fake00000000", "_fetch": True}
    jd = Path(me["jd"])
    (jd / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (jd / "status.json").write_text(json.dumps({"state": "running", "assign": GPU_ASSIGN}), encoding="utf-8")
    environ = {"RX_HOME": str(home), "RX_JOB_DIR": str(jd), "RX_PROJECT": "jigsaw", "RX_JOB_ID": jid, "RX_GPUS": "0"}
    return {"home": home, "ws": ws, "jd": jd, "spec": spec, "environ": environ, "state": tmp_path / "campaigns",
            "tmp": tmp_path}


def rpc_log(home):
    path = home / "rpc.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def records(state, campaign="cal"):
    path = state / f"{campaign}.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def events(state, campaign="cal"):
    """The campaign's events without the periodic `host` records."""
    return [r["event"] for r in records(state, campaign) if r["event"] != "host"]


def run_args(h, *extra, child=("5", "0")):
    return lp.build_parser().parse_args(
        ["run", "--requeue", "--state-dir", str(h["state"]), "--poll-s", "0.05", "--confirm-s", "0.2",
         "--after-progress-s", "0.05", *extra, "--", sys.executable, "child.py", str(h["tmp"]), *child])


def later(seconds, fn):
    t = threading.Timer(seconds, fn)
    t.daemon = True
    t.start()
    return t


def after_event(state, name, fn, timeout_s=30.0):
    """Run fn (in a thread) once the campaign log holds the event, so a slow or loaded machine cannot let a
    fixed delay pass before lowprio gets there."""
    def watch():
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            try:
                seen = name in events(state)
            except ValueError:  # a line being written
                seen = False
            if seen:
                fn()
                return
            time.sleep(0.02)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    return t


# ───────────────────────────── scheduler arithmetic ─────────────────────────────

def test_fit_take_mirror_rx():
    gfree = {0: 1.0, 1: 0.5}
    assert lp.fit(req(4, 8, 1), 10, 20, gfree)["gpu_shares"] == {"0": 1.0}
    assert lp.fit(req(4, 8, 0.5), 10, 20, gfree)["gpu_shares"] == {"1": 0.5}  # best fit keeps whole GPUs whole
    assert lp.fit(req(4, 8, 2), 10, 20, gfree) is None
    assert lp.fit(req(11, 8), 10, 20, gfree) is None and lp.fit(req(4, 21), 10, 20, gfree) is None
    a = lp.fit(req(4, 8, 1), 10, 20, gfree)
    pool = lp.take((10.0, 20.0, gfree), a)
    assert pool == (6.0, 12.0, {0: 0.0, 1: 0.5})
    assert lp.take(pool, a, sign=-1.0) == (10.0, 20.0, gfree)
    total = lp.capacity_pool(CAP)
    assert total[0] == 30 and total[1] == pytest.approx(119.7) and total[2] == {0: 1.0}
    assert lp.job_request({"cpus": 0.1, "mem_gb": 0.25}) == req(0.1, 0.25)


def test_rx_contention(tmp_path):
    home = tmp_path
    now = time.time()
    me = entry("jigsaw", "me", req(8, 32, 1), home=home, kind="active", assign=GPU_ASSIGN, admitted=now - 500)
    check = lambda: lp.rx_contention(CAP, lp.live_entries(home, "active", now, 0), lp.live_entries(home, "queue", now, 0),
                                     project="jigsaw", key=me["key"], now=now)
    assert check() is None
    entry("mpr", "cpu24", req(24, 40), home=home, kind="queue", enq=now - 30)  # 22 CPUs free, 30 without us
    got = check()
    assert got and got["job"] == "cpu24" and got["project"] == "mpr" and got["kind"] == "rx"
    entry("mpr", "fits", req(20, 64), home=home, kind="queue", enq=now - 100)  # older, fits, goes first (FIFO)
    assert check() is None  # after it, cpu24 would not fit even without us
    for name in ("cpu24", "fits"):
        (home / "queue" / f"mpr--{name}.json").unlink()
    entry("mpr", "gpu", req(8, 32, 1), home=home, kind="queue")
    assert check()["job"] == "gpu"
    (home / "queue" / "mpr--gpu.json").unlink()
    entry("mpr", "huge", req(64, 32), home=home, kind="queue")  # impossible on this host: ignored
    entry("jigsaw", "sibling", req(8, 32, 1), home=home, kind="queue")  # our own project never counts
    entry("mpr", "dead", req(8, 32, 1), home=home, kind="queue", alive=False)  # its supervisor is gone
    assert check() is None


def test_youngest_jigsaw_job_yields_first(tmp_path):
    home = tmp_path
    now = time.time()
    old = entry("jigsaw", "old", req(8, 16), home=home, kind="active", admitted=now - 900)
    young = entry("jigsaw", "young", req(8, 16), home=home, kind="active", admitted=now - 60)
    entry("mpr", "other", req(8, 16), home=home, kind="active")
    entry("mpr", "cpu12", req(12, 16), home=home, kind="queue")  # 6 CPUs free; the young job's 8 are enough
    active, queue = lp.live_entries(home, "active", now, 0), lp.live_entries(home, "queue", now, 0)
    assert lp.rx_contention(CAP, active, queue, project="jigsaw", key=young["key"], now=now)
    assert lp.rx_contention(CAP, active, queue, project="jigsaw", key=old["key"], now=now) is None
    (home / "queue" / "mpr--cpu12.json").unlink()
    entry("mpr", "cpu20", req(20, 16), home=home, kind="queue")  # needs both of ours
    queue = lp.live_entries(home, "queue", now, 0)
    assert lp.rx_contention(CAP, active, queue, project="jigsaw", key=old["key"], now=now)
    assert lp.rx_contention(CAP, active, queue, project="jigsaw", key=young["key"], now=now)


def test_only_jobs_the_blocked_job_needs_yield(tmp_path):
    home = tmp_path
    now = time.time()
    train = entry("jigsaw", "train", req(8, 32, 1), home=home, kind="active", assign=GPU_ASSIGN, admitted=now - 900)
    build = entry("jigsaw", "build", req(4, 16), home=home, kind="active", admitted=now - 300)
    wait = entry("jigsaw", "wait", dict(lp.WAITER_REQ), home=home, kind="active", admitted=now - 10)
    entry("mpr", "cpu14", req(14, 16), home=home, kind="active")  # 4 CPUs free
    entry("mpr", "gpu", req(8, 32, 1), home=home, kind="queue")

    def yields(job):
        active, queue = lp.live_entries(home, "active", now, 0), lp.live_entries(home, "queue", now, 0)
        return lp.rx_contention(CAP, active, queue, project="jigsaw", key=job["key"], now=now) is not None

    # it waits for the GPU: the CPU build and the waiter, both younger, keep running
    assert yields(train) and not yields(build) and not yields(wait)
    (home / "queue" / "mpr--gpu.json").unlink()
    entry("mpr", "cpu10", req(10, 16), home=home, kind="queue")  # the build's 4 CPUs are not enough, the trainer's are
    assert yields(train) and not yields(build) and not yields(wait)
    (home / "queue" / "mpr--cpu10.json").unlink()
    entry("mpr", "cpu4", req(4, 16), home=home, kind="queue")  # exactly the free CPUs: the waiter is below tolerance
    assert not any(yields(j) for j in (train, build, wait))
    entry("mpr", "cpu3", req(3, 16), home=home, kind="queue")  # after cpu4 takes the free CPUs, the build is enough
    assert yields(build) and not yields(train) and not yields(wait)
    (home / "queue" / "mpr--cpu3.json").unlink()
    entry("mpr", "cpu8", req(8, 16), home=home, kind="queue")  # the build's 4 are not: only the trainer yields
    assert yields(train) and not yields(build) and not yields(wait)


CAP_P = dict(CAP, priority={"mpr": 10.0})  # the lab host's map: mpr first, every other project 0


def rx_agent_module():
    """rx's own scheduler: $JIGSAW_RX_AGENT, the sibling message-passing-retrieval checkout, or the agent
    that $RX_HOME deploys on the host."""
    import glob
    import importlib.util

    cands = [os.environ.get("JIGSAW_RX_AGENT") or "",
             str(REPO.parent / "message-passing-retrieval" / "tools" / "rx" / "rx_agent.py")]
    if os.environ.get("RX_HOME"):
        cands += sorted(glob.glob(os.path.join(os.environ["RX_HOME"], "agent", "rx_agent-*.py")),
                        key=os.path.getmtime, reverse=True)
    path = next((c for c in cands if c and os.path.isfile(c)), None)
    if path is None:
        pytest.skip("no rx_agent.py to compare with")
    spec = importlib.util.spec_from_file_location("rx_agent_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decide_matches_rx_agent():
    """Random hosts, running jobs and queues: lowprio's decide admits exactly what rx's decide admits."""
    import random

    rx = rx_agent_module()
    rng = random.Random(20260929)
    now = 1_790_000_000.0
    projects = ["jigsaw", "mpr", "crag", "other"]
    checked = admitted = 0
    for trial in range(400):
        cap = dict(CAP_P if trial % 2 else CAP, gpus=[{"index": i} for i in range(rng.choice([1, 1, 2]))],
                   hold_after_s=rng.choice([600, 1800]))
        pool = lp.capacity_pool(cap)
        active = []
        for i in range(rng.randint(0, 5)):
            r = req(rng.choice([1, 2, 4, 8]), rng.choice([4, 16, 32]), rng.choice([0, 0, 0.25, 0.5, 1]))
            placed = lp.fit(r, *pool)
            if placed is None:
                continue
            pool = lp.take(pool, placed)
            p = rng.choice(projects)
            active.append({"key": f"{p}--a{i}", "project": p, "id": f"a{i}", "req": r, "assign": placed,
                           "admitted": now - rng.uniform(0, 5000)})
        queue = []
        for i in range(rng.randint(1, 6)):
            p = rng.choice(projects)
            queue.append({"key": f"{p}--q{i}", "project": p, "id": f"q{i}", "enq": now - rng.uniform(0, 4000),
                          "req": req(rng.choice([1, 4, 8, 16, 24, 40]), rng.choice([2, 16, 48, 100]),
                                     rng.choice([0, 0, 0.3, 0.75, 1, 2]))})
        for me in queue:
            ours = lp.decide(me, cap, active, queue, now)
            theirs = rx.decide(me, cap, active, queue, now)
            assert (ours[0], ours[1]) == (theirs[0], theirs[1]), (trial, me["key"], ours, theirs)
            checked += 1
            admitted += bool(ours[0])
    assert checked > 1000 and 0.1 < admitted / checked < 0.9  # both outcomes are well exercised


def test_rx_contention_follows_rx_priority_and_peer_wait(tmp_path):
    home = tmp_path
    now = time.time()
    me = entry("jigsaw", "me", req(8, 32, 1), home=home, kind="active", assign=GPU_ASSIGN, admitted=now - 500)

    def check(cap=CAP, **policy):
        return lp.rx_contention(cap, lp.live_entries(home, "active", now, 0), lp.live_entries(home, "queue", now, 0),
                                project="jigsaw", key=me["key"], now=now, **policy)

    peer = entry("crag", "cpu24", req(24, 40), home=home, kind="queue", enq=now - 100)  # 22 CPUs free, 30 without us
    assert check()["job"] == "cpu24"  # without --yield-to every other project comes first
    assert check(yield_to=["mpr"]) is None  # a peer waits peer_wait_s (default hold_after_s) first
    assert check(yield_to=["mpr"], peer_wait_s=60)["job"] == "cpu24"
    peer["enq"] = now - 1900
    (home / "queue" / "crag--cpu24.json").write_text(json.dumps(peer), encoding="utf-8")
    assert check(yield_to=["mpr"])["job"] == "cpu24"
    (home / "queue" / "crag--cpu24.json").unlink()
    entry("mpr", "cpu24", req(24, 40), home=home, kind="queue", enq=now - 5)
    assert check(yield_to=["mpr"])["project"] == "mpr"  # a named project at once
    assert check(yield_to=["crag"]) is None
    assert check(CAP_P, yield_to=["crag"])["project"] == "mpr"  # and a project rx ranks above ours
    # rx's order: mpr's job goes ahead of an older peer job that fits, and then it needs our CPUs
    entry("crag", "fits", req(20, 64), home=home, kind="queue", enq=now - 200)
    assert check() is None  # first in, first out: after crag's job takes 20 CPUs, mpr's cannot fit even without us
    assert check(CAP_P, yield_to=["mpr"])["job"] == "cpu24"


def test_launch_blocker_mirrors_rx_admission(tmp_path):
    home = tmp_path
    now = time.time()
    entry("jigsaw", "lane", req(8, 32), home=home, kind="active", admitted=now - 300)

    def blocker(cap=CAP, target=req(8, 32, 1), **policy):
        return lp.launch_blocker(cap, lp.live_entries(home, "active", now, 0), lp.live_entries(home, "queue", now, 0),
                                 target, project="jigsaw", now=now, **policy)

    assert blocker() is None  # 22 CPUs free
    assert "waiting for resources" in blocker(target=req(24, 32))
    entry("mpr", "cpu26", req(26, 40), home=home, kind="queue", enq=now - 30)  # needs the lane's CPUs too
    assert "yield to mpr job cpu26" in blocker()  # it would fit before the lane's; our new job would be in its way
    assert "higher-priority job cpu26" in blocker(CAP_P)  # rx itself sets its request aside
    (home / "queue" / "mpr--cpu26.json").unlink()
    # a peer's job blocked on its own project's GPU does not keep a CPU job out
    entry("crag", "gpu", req(2, 8, 1), home=home, kind="active",
          assign={"cpus": 2.0, "mem_gb": 8.0, "gpus": [0], "gpu_shares": {"0": 1.0}})
    entry("crag", "gpu2", req(2, 8, 1), home=home, kind="queue", enq=now - 100)
    assert blocker(target=req(8, 32)) is None and blocker(target=req(8, 32), yield_to=["mpr"]) is None
    assert "waiting for resources" in blocker()  # the GPU job itself cannot start
    (home / "queue" / "crag--gpu2.json").unlink()
    # a peer's CPU job that the new job would keep out: backfill until it has waited peer_wait_s
    entry("crag", "cpu20", req(20, 32), home=home, kind="queue", enq=now - 100)  # 20 CPUs free: it fits first
    assert "waiting behind older queued jobs" in blocker(target=req(8, 32))
    (home / "queue" / "crag--cpu20.json").unlink()
    entry("crag", "cpu26", req(26, 32), home=home, kind="queue", enq=now - 100)
    assert blocker(target=req(8, 32), yield_to=["mpr"]) is None
    assert "yield to crag job cpu26" in blocker(target=req(8, 32), yield_to=["mpr"], peer_wait_s=60)
    assert "yield to crag job cpu26" in blocker(target=req(8, 32))  # without --yield-to: at once
    (home / "queue" / "crag--cpu26.json").unlink()
    entry("crag", "old", req(26, 32), home=home, kind="queue", enq=now - 2000)  # rx's hold rule: nothing overtakes it
    assert "holding for older job old" in blocker(target=req(1, 1), yield_to=["mpr"])


def test_gpu_processes_of_rx_jobs(tmp_path, monkeypatch):
    home = tmp_path
    jobs = {"jigsaw": ("sib", 500), "crag": ("gen", 600), "mpr": ("probe", 700)}
    for project, (jid, pid) in jobs.items():
        e = entry(project, jid, req(2, 8, 0.25), home=home, kind="active",
                  assign={"cpus": 2.0, "mem_gb": 8.0, "gpus": [0], "gpu_shares": {"0": 0.25}})
        (Path(e["jd"]) / "status.json").write_text(json.dumps({"supervisor_pid": pid, "pid": pid + 1}),
                                                  encoding="utf-8")
    lineage = {502: [502, 501, 500], 601: [601, 600], 702: [702, 701, 700, 4], 999: [999, 4]}
    monkeypatch.setattr(lp, "process_lineage", lambda pid: lineage.get(pid, [pid]))
    procs = [{"gpu": 0, "pid": pid, "type": "C", "name": "python.exe"} for pid in (502, 601, 702, 999)]
    active = lp.live_entries(home, "active", time.time(), 0)
    owners = lp.gpu_process_jobs(procs, active)
    assert {pid: job["project"] for pid, job in owners.items()} == {502: "jigsaw", 601: "crag", 702: "mpr"}
    shared = lambda cap=CAP, **kw: lp.shared_gpu_pids(procs, cap, active, project="jigsaw", **kw)
    assert shared(yield_to=None) == {502}  # without --yield-to only our own jobs share the GPU
    assert shared(yield_to=["mpr"]) == {502, 601}  # a peer's rx job shares it as rx assigned it
    assert shared(CAP_P, yield_to=["crag"]) == {502}  # mpr is ranked above us; crag is named
    assert lp.gpu_process_jobs(procs, []) == {}


def test_run_shares_the_gpu_with_peers_and_yields_to_mpr(host, monkeypatch):
    for project, jid, pid in (("crag", "gen", 600), ("mpr", "probe", 700)):
        e = entry(project, jid, req(2, 8), home=host["home"], kind="active")
        (Path(e["jd"]) / "status.json").write_text(json.dumps({"supervisor_pid": pid}), encoding="utf-8")
    monkeypatch.setattr(lp, "process_lineage", lambda pid: [pid, pid - 1])
    gpu = lambda pid: (lambda: [{"gpu": 0, "pid": pid, "type": "C", "name": "python.exe"}])
    assert lp.cmd_run(run_args(host, "--yield-to", "mpr", child=("0.1", "0")), lp.JobContext(host["environ"]),
                      gpu_query=gpu(601)) == 0
    assert lp.cmd_run(run_args(host, "--yield-to", "mpr"), lp.JobContext(host["environ"]),
                      gpu_query=gpu(701)) == lp.EXIT_YIELD
    assert lp.cmd_run(run_args(host), lp.JobContext(host["environ"]), gpu_query=gpu(601)) == lp.EXIT_YIELD
    launched = rpc_log(host["home"])[-1]["spec"]
    assert launched["lowprio"]["yield_to"] is None and launched["lowprio"]["last_yield"]["kind"] == "gpu"


def test_policy_options():
    args = lp.build_parser().parse_args(["run", "--yield-to", "mpr, other", "--peer-wait-s", "900", "--", "x"])
    assert args.yield_to == ["mpr", "other"] and args.peer_wait_s == 900.0
    meta = lp._meta_for(lp.JobContext({}), args)
    assert meta["yield_to"] == ["mpr", "other"] and meta["peer_wait_s"] == 900.0
    assert lp.build_parser().parse_args(["run", "--", "x"]).yield_to is None
    for bad in ("", "a b", ",,"):
        with pytest.raises(SystemExit):
            lp.build_parser().parse_args(["run", "--yield-to", bad, "--", "x"])


def test_gpu_processes_parse_and_filter():
    procs = lp.parse_gpu_processes(GPU_XML)
    assert [(p["gpu"], p["pid"], p["type"]) for p in procs] == [(0, 111, "C"), (0, 222, "C+G"), (0, 333, "G"),
                                                                  (1, 444, "C")]
    pick = lambda **kw: [p["pid"] for p in lp.foreign_gpu_processes(procs, **{"gpus": [0], "mine": set(),
                                                                              "types": ["C"], "ignore": [], **kw})]
    assert pick() == [111]
    assert pick(mine={111}) == []
    assert pick(types=["C", "C+G"]) == [111, 222]
    assert pick(ignore=["python.exe"]) == []
    assert pick(gpus=[0, 1]) == [111, 444]


def test_requeue_specs(host):
    spec = host["spec"]
    meta = {"campaign": "cal", "gen": 1, "origin": spec["id"], "requeue": True}
    target = lp.job_template(spec, meta)
    assert "id" not in target and "created" not in target and "agent" not in target and target["_fetch"] is False
    assert target["argv"] == spec["argv"] and target["gpus"] == 1.0 and target["cpus"] == 8.0
    assert target["lowprio"] == dict(meta, role="job") and target["rerun_of"] == spec["id"]
    waiter = lp.waiter_spec(spec, target, meta)
    assert lp.job_request(waiter) == lp.WAITER_REQ and waiter["argv"][-1] == "wait"
    assert waiter["lowprio_target"] == target and waiter["lowprio"]["role"] == "waiter"
    assert waiter["ws"] == spec["ws"] and waiter["env"] == spec["env"] and waiter["outputs"] == ["outputs/lowprio/**"]
    relaunch = lp.launched_spec(target)
    for jid in (waiter["id"], relaunch["id"]):
        assert lp.NAME_RE.match(jid) and jid != spec["id"]
    assert relaunch["id"].split("-")[2] == "cal" and waiter["name"] == "cal-wait"


def test_agent_resolution(host):
    ctx = lp.JobContext(host["environ"])
    assert ctx.under_rx and ctx.key == f"jigsaw--{ctx.job_id}" and ctx.gpus == [0]
    assert lp.agent_path(ctx).endswith("rx_agent-fake00000000.py")
    ctx.spec["agent"] = "missing"
    assert lp.agent_path(ctx).endswith("rx_agent-fake00000000.py")  # the newest agent otherwise
    assert lp.rx_python(ctx) == sys.executable
    res = lp.agent_call(ctx, {"op": "launch", "spec": {"project": "jigsaw", "id": "x-1"}})
    assert res["job"]["state"] == "running" and rpc_log(host["home"])[0]["op"] == "launch"
    with pytest.raises(lp.LowprioError):
        lp.agent_call(ctx, {"op": "nope"})


# ───────────────────────────── run ─────────────────────────────

def test_run_completes_without_contention(host):
    ctx = lp.JobContext(host["environ"])
    rc = lp.cmd_run(run_args(host, child=("0.2", "0")), ctx, gpu_query=lambda: [])
    assert rc == 0 and (host["tmp"] / "finished").exists()
    assert events(host["state"]) == ["start", "code", "child", "done"]
    assert rpc_log(host["home"]) == []
    assert lp.cmd_run(run_args(host, child=("0.1", "5")), lp.JobContext(host["environ"]), gpu_query=lambda: []) == 5


def test_run_yields_on_foreign_queue_after_progress(host):
    ctx = lp.JobContext(host["environ"])
    progress = host["tmp"] / "runs"
    progress.mkdir()
    later(0.4, lambda: entry("mpr", "cpu24", req(24, 40), home=host["home"], kind="queue"))
    later(1.2, lambda: (progress / "ckpt_epoch005.pt").write_bytes(b"x"))
    t0 = time.time()
    rc = lp.cmd_run(run_args(host, "--grace-s", "30", "--progress", str(progress / "*.pt"), child=("20", "0")),
                    ctx, gpu_query=lambda: [])
    assert rc == lp.EXIT_YIELD and time.time() - t0 < 15
    assert (host["tmp"] / "started").exists() and not (host["tmp"] / "finished").exists()
    ev = events(host["state"])
    assert ev[:3] == ["start", "code", "child"] and ev[-3:] == ["yielding", "yielded", "requeued"]
    launched = rpc_log(host["home"])
    assert len(launched) == 1 and launched[0]["op"] == "launch"
    waiter = launched[0]["spec"]
    assert waiter["lowprio_target"]["argv"] == host["spec"]["argv"] and waiter["lowprio"]["gen"] == 1
    assert waiter["lowprio_target"]["lowprio"]["campaign"] == "cal" and waiter["cpus"] == lp.WAITER_REQ["cpus"]
    log = [json.loads(x) for x in (host["state"] / "cal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert next(e for e in log if e["event"] == "yielded")["on"] == "progress"


def test_run_yields_before_start(host):
    entry("mpr", "gpu", req(8, 32, 1), home=host["home"], kind="queue")
    rc = lp.cmd_run(run_args(host), lp.JobContext(host["environ"]), gpu_query=lambda: [])
    assert rc == lp.EXIT_YIELD and not (host["tmp"] / "started").exists()
    assert events(host["state"])[-2:] == ["yield", "requeued"]


def test_run_yields_on_grace_without_progress(host):
    later(0.3, lambda: entry("mpr", "gpu", req(8, 32, 1), home=host["home"], kind="queue"))
    rc = lp.cmd_run(run_args(host, "--grace-s", "0.8", "--progress", str(host["tmp"] / "none" / "*.pt"),
                             child=("20", "0")), lp.JobContext(host["environ"]), gpu_query=lambda: [])
    assert rc == lp.EXIT_YIELD
    log = [json.loads(x) for x in (host["state"] / "cal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert next(e for e in log if e["event"] == "yielded")["on"] == "grace"


def test_yield_cancelled_when_contention_ends(host):
    later(0.3, lambda: entry("mpr", "gpu", req(8, 32, 1), home=host["home"], kind="queue"))
    after_event(host["state"], "yielding", lambda: (host["home"] / "queue" / "mpr--gpu.json").unlink())
    rc = lp.cmd_run(run_args(host, "--grace-s", "60", child=("8", "0")), lp.JobContext(host["environ"]),
                    gpu_query=lambda: [])
    assert rc == 0 and (host["tmp"] / "finished").exists()
    assert "yield_cancelled" in events(host["state"]) and rpc_log(host["home"]) == []


def wall_clock(monkeypatch, fn):
    """lowprio's time module with time() replaced (its sleep and monotonic stay real)."""
    shim = types.SimpleNamespace(**{k: getattr(time, k) for k in dir(time) if not k.startswith("_")})
    shim.time = fn
    monkeypatch.setattr(lp, "time", shim)


def test_yield_timing_ignores_the_wall_clock(host, monkeypatch):
    """Confirmation, cancellation and grace are durations on the monotonic clock: a wall clock that stops or
    steps (the host's WSL steps it by 31.7 s every few seconds) neither delays nor hastens a yield."""
    frozen = time.time()
    wall_clock(monkeypatch, lambda: frozen)
    later(0.3, lambda: entry("mpr", "gpu", req(8, 32, 1), home=host["home"], kind="queue"))
    after_event(host["state"], "yielding", lambda: (host["home"] / "queue" / "mpr--gpu.json").unlink())
    rc = lp.cmd_run(run_args(host, "--grace-s", "60", child=("8", "0")), lp.JobContext(host["environ"]),
                    gpu_query=lambda: [])
    assert rc == 0 and (host["tmp"] / "finished").exists()
    assert events(host["state"]).count("yielding") == 1 and "yield_cancelled" in events(host["state"])


def test_foreign_gpu_process_yields_and_desktop_apps_do_not(host):
    desktop = [{"gpu": 0, "pid": 222, "type": "C+G", "name": "C:\\Windows\\explorer.exe"}]
    rc = lp.cmd_run(run_args(host, child=("0.1", "0")), lp.JobContext(host["environ"]), gpu_query=lambda: desktop)
    assert rc == 0
    other = desktop + [{"gpu": 0, "pid": 999999, "type": "C", "name": "C:\\someone\\python.exe"}]
    rc = lp.cmd_run(run_args(host), lp.JobContext(host["environ"]), gpu_query=lambda: other)
    assert rc == lp.EXIT_YIELD and rpc_log(host["home"])[-1]["spec"]["lowprio"]["last_yield"]["kind"] == "gpu"
    rc = lp.cmd_run(run_args(host, "--ignore-gpu-process", "python.exe", child=("0.1", "0")),
                    lp.JobContext(host["environ"]), gpu_query=lambda: other)
    assert rc == 0


def test_stop_file_prevents_requeue(host):
    assert lp.main(["stop", "--campaign", "cal", "--state-dir", str(host["state"])]) == 2  # no such campaign yet
    assert not (host["state"] / "cal.STOP").exists()
    assert lp.main(["stop", "--campaign", "cal", "--state-dir", str(host["state"]), "--force"]) == 0
    rc = lp.cmd_run(run_args(host), lp.JobContext(host["environ"]), gpu_query=lambda: [])
    assert rc == 0 and events(host["state"])[-1] == "stopped" and not (host["tmp"] / "started").exists()
    assert lp.main(["stop", "--campaign", "cal", "--state-dir", str(host["state"]), "--clear"]) == 0
    (host["state"] / "cal.STOP").write_text("x", encoding="utf-8")
    ctx = lp.JobContext(host["environ"])
    meta = lp._meta_for(ctx, run_args(host))
    assert lp.requeue(ctx, meta, host["state"], None) is None
    assert events(host["state"])[-1] == "no_requeue" and rpc_log(host["home"]) == []


def test_code_is_pinned_across_hops(host):
    assert lp.cmd_run(run_args(host, child=("0.1", "0")), lp.JobContext(host["environ"]), gpu_query=lambda: []) == 0
    spec = dict(host["spec"], lowprio={"campaign": "cal", "gen": 1, "origin": host["spec"]["id"], "requeue": True,
                                       "state_dir": str(host["state"]), "max_requeues": 5})
    (host["jd"] / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    assert lp.cmd_run(run_args(host, child=("0.1", "0")), lp.JobContext(host["environ"]), gpu_query=lambda: []) == 0
    (host["ws"] / "scripts" / "code.py").write_bytes(b"x = 2\n")
    rc = lp.cmd_run(run_args(host, child=("0.1", "0")), lp.JobContext(host["environ"]), gpu_query=lambda: [])
    assert rc == lp.EXIT_CODE_CHANGED
    (host["ws"] / "scripts" / "code.py").write_bytes(b"x = 1\r\n")  # CRLF folds to the same code
    assert lp.cmd_run(run_args(host, child=("0.1", "0")), lp.JobContext(host["environ"]), gpu_query=lambda: []) == 0


def test_max_requeues(host):
    spec = dict(host["spec"], lowprio={"campaign": "cal", "gen": 2, "origin": host["spec"]["id"], "requeue": True,
                                       "state_dir": str(host["state"]), "max_requeues": 2})
    (host["jd"] / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (host["state"]).mkdir()
    (host["state"] / "cal.code.json").write_text(json.dumps(lp.manifest.code_fingerprint(host["ws"])), encoding="utf-8")
    entry("mpr", "gpu", req(8, 32, 1), home=host["home"], kind="queue")
    assert lp.cmd_run(run_args(host), lp.JobContext(host["environ"]), gpu_query=lambda: []) == lp.EXIT_YIELD
    assert events(host["state"])[-1] == "no_requeue" and rpc_log(host["home"]) == []


# ───────────────────────────── the waiter ─────────────────────────────

def make_waiter(host, *, settle="0.1", max_queue="0.4", last_yield=None, target_req=None, **policy):
    """Turn the fixture's job into the waiter that its yield would launch, holding 0.1 CPU (a share above
    rx's fit tolerance) so that a job can wait for it. target_req overrides the job's cpus/mem_gb/gpus."""
    target = lp.job_template(dict(host["spec"], **(target_req or {})),
                             {"campaign": "cal", "gen": 1, "origin": host["spec"]["id"]})
    meta = {"campaign": "cal", "gen": 1, "origin": host["spec"]["id"], "state_dir": str(host["state"]),
            "poll_s": 0.05, "settle_s": float(settle), "gpu_quiet_s": 0.5, "max_queue_s": float(max_queue),
            "gpu_policy": "yield", "gpu_types": "C", "ignore_gpu_process": [], **policy,
            **({"last_yield": last_yield} if last_yield else {})}
    waiter = lp.waiter_spec(host["spec"], target, meta)
    waiter["id"] = host["spec"]["id"]  # the fixture's job directory plays the waiter
    waiter["agent"] = "fake00000000"
    (host["jd"] / "spec.json").write_text(json.dumps(waiter), encoding="utf-8")
    active = host["home"] / "active" / f"jigsaw--{waiter['id']}.json"
    e = json.loads(active.read_text(encoding="utf-8"))
    e.update(req=lp.job_request(waiter), assign={"cpus": 0.1, "mem_gb": 0.25, "gpus": [], "gpu_shares": {}})
    active.write_text(json.dumps(e), encoding="utf-8")
    host["state"].mkdir(exist_ok=True)
    (host["state"] / "cal.code.json").write_text(json.dumps(lp.manifest.code_fingerprint(host["ws"])), encoding="utf-8")
    env = dict(host["environ"], RX_GPUS="")
    return lp.JobContext(env)


def wait_args(give_up="3"):
    return lp.build_parser().parse_args(["wait", "--give-up-s", give_up])


def test_waiter_relaunches_when_the_host_has_room(host):
    ctx = make_waiter(host)
    assert lp.cmd_wait(wait_args(), ctx, gpu_query=lambda: []) == 0
    calls = rpc_log(host["home"])
    assert [c["op"] for c in calls] == ["launch"]
    spec = calls[0]["spec"]
    assert spec["argv"] == host["spec"]["argv"] and spec["gpus"] == 1.0 and spec["lowprio"]["gen"] == 1
    assert spec["id"] != host["spec"]["id"] and "lowprio_target" not in spec
    assert events(host["state"])[-2:] == ["relaunched", "handed_over"]


def test_waiter_waits_while_others_need_the_host(host):
    ctx = make_waiter(host)
    entry("mpr", "gpujob", req(8, 32, 1), home=host["home"], kind="active",
          assign={"cpus": 8.0, "mem_gb": 32.0, "gpus": [0], "gpu_shares": {"0": 1.0}})
    assert lp.cmd_wait(wait_args("0.6"), ctx, gpu_query=lambda: []) == lp.EXIT_GAVE_UP
    (host["home"] / "active" / "mpr--gpujob.json").unlink()
    entry("mpr", "queued", req(8, 32, 1), home=host["home"], kind="queue")  # older and fits: it takes the GPU first
    assert lp.cmd_wait(wait_args("0.6"), ctx, gpu_query=lambda: []) == lp.EXIT_GAVE_UP
    (host["home"] / "queue" / "mpr--queued.json").unlink()
    other = [{"gpu": 0, "pid": 999999, "type": "C", "name": "python.exe"}]
    assert lp.cmd_wait(wait_args("0.6"), ctx, gpu_query=lambda: other) == lp.EXIT_GAVE_UP
    ctx = make_waiter(host, last_yield={"kind": "gpu", "t": time.time()})  # the GPU must stay quiet a while
    assert lp.cmd_wait(wait_args("0.3"), ctx, gpu_query=lambda: []) == lp.EXIT_GAVE_UP
    assert rpc_log(host["home"]) == []


def test_waiter_quiet_period_survives_a_clock_step(host):
    """The yield's wall time came from a clock 31.7 s ahead that has since stepped back: the waiter still
    waits the whole quiet period, and only that."""
    ctx = make_waiter(host, last_yield={"kind": "gpu", "t": time.time() + 31.7})
    assert lp.cmd_wait(wait_args("0.3"), ctx, gpu_query=lambda: []) == lp.EXIT_GAVE_UP
    assert lp.cmd_wait(wait_args("3"), ctx, gpu_query=lambda: []) == 0
    assert [c["op"] for c in rpc_log(host["home"])] == ["launch"]


def test_waiter_launches_beside_jobs_blocked_elsewhere(host):
    """A queued job that the relaunch would not keep out does not hold the waiter (the old rule idled the host
    whenever another project had anything queued), and one it would keep out does."""
    entry("crag", "gpu", req(2, 8, 1), home=host["home"], kind="active",
          assign={"cpus": 2.0, "mem_gb": 8.0, "gpus": [0], "gpu_shares": {"0": 1.0}})
    entry("crag", "gpu2", req(2, 8, 1), home=host["home"], kind="queue")  # waits for crag's own GPU job
    ctx = make_waiter(host, target_req={"gpus": 0.0}, yield_to=["mpr"])
    assert lp.cmd_wait(wait_args(), ctx, gpu_query=lambda: []) == 0
    assert [c["op"] for c in rpc_log(host["home"])] == ["launch"] and events(host["state"])[-1] == "handed_over"
    entry("mpr", "cpu26", req(26, 40), home=host["home"], kind="queue")  # would fit now, not beside the relaunch
    ctx = make_waiter(host, target_req={"gpus": 0.0}, yield_to=["mpr"])
    assert lp.cmd_wait(wait_args("0.6"), ctx, gpu_query=lambda: []) == lp.EXIT_GAVE_UP
    assert len(rpc_log(host["home"])) == 1


def test_waiter_cancels_a_relaunch_stuck_in_the_queue(host, monkeypatch):
    monkeypatch.setenv("FAKE_RX_STATE", "queued")
    ctx = make_waiter(host, max_queue="0.3")
    assert lp.cmd_wait(wait_args("2.5"), ctx, gpu_query=lambda: []) == lp.EXIT_GAVE_UP
    ops = [c["op"] for c in rpc_log(host["home"])]
    assert ops[:2] == ["launch", "cancel"] and ops.count("launch") >= 2
    assert "relaunch_cancelled" in events(host["state"])


def test_waiter_steps_aside_for_a_foreign_job(host):
    ctx = make_waiter(host)
    entry("mpr", "all", req(29.95, 64), home=host["home"], kind="queue")  # fits only without the waiter's 0.1 CPU
    assert lp.cmd_wait(wait_args(), ctx, gpu_query=lambda: []) == 0
    calls = rpc_log(host["home"])
    assert len(calls) == 1 and calls[0]["spec"]["argv"][-1] == "wait" and "lowprio_target" in calls[0]["spec"]


def make_deferred(host):
    """The fixture's job as the small first hop of `run --defer`, holding 0.1 CPU as make_waiter's does."""
    small = dict(host["spec"], cpus=0.1, mem_gb=0.25, gpus=0.0)
    (host["jd"] / "spec.json").write_text(json.dumps(small), encoding="utf-8")
    active = host["home"] / "active" / f"jigsaw--{small['id']}.json"
    e = json.loads(active.read_text(encoding="utf-8"))
    e.update(req=lp.job_request(small), assign={"cpus": 0.1, "mem_gb": 0.25, "gpus": [], "gpu_shares": {}})
    active.write_text(json.dumps(e), encoding="utf-8")
    return lp.JobContext(dict(host["environ"], RX_GPUS=""))


DEFER = ("--defer", "8,32,1", "--settle-s", "0.1", "--max-queue-s", "0.4", "--gpu-quiet-s", "0.5")


def test_deferred_start_launches_the_full_request(host):
    assert lp.cmd_run(run_args(host, *DEFER), make_deferred(host), gpu_query=lambda: []) == 0
    assert not (host["tmp"] / "started").exists()  # the small hop never runs the command
    calls = rpc_log(host["home"])
    assert [c["op"] for c in calls] == ["launch"]
    spec = calls[0]["spec"]
    assert (spec["cpus"], spec["mem_gb"], spec["gpus"]) == (8.0, 32.0, 1.0)
    assert spec["argv"] == host["spec"]["argv"] and spec["lowprio"]["gen"] == 1 and "lowprio_target" not in spec
    assert events(host["state"]) == ["start", "code", "deferred", "waiting", "relaunched", "handed_over"]
    # the launched job runs the command: a later hop never defers, and its code must equal the first hop's
    (host["jd"] / "spec.json").write_text(json.dumps(dict(spec, id=host["spec"]["id"])), encoding="utf-8")
    assert lp.cmd_run(run_args(host, *DEFER, child=("0.1", "0")), lp.JobContext(host["environ"]),
                      gpu_query=lambda: []) == 0
    assert (host["tmp"] / "finished").exists() and events(host["state"])[-3:] == ["code", "child", "done"]


def test_deferred_start_steps_aside_and_needs_rx(host):
    entry("mpr", "all", req(29.95, 64), home=host["home"], kind="queue")  # fits only without the 0.1 CPU
    assert lp.cmd_run(run_args(host, *DEFER), make_deferred(host), gpu_query=lambda: []) == 0
    calls = rpc_log(host["home"])
    assert len(calls) == 1 and calls[0]["spec"]["argv"][-1] == "wait"
    assert calls[0]["spec"]["cpus"] == lp.WAITER_REQ["cpus"]
    target = calls[0]["spec"]["lowprio_target"]
    assert (target["cpus"], target["mem_gb"], target["gpus"]) == (8.0, 32.0, 1.0) and target["lowprio"]["gen"] == 1
    assert lp.cmd_run(run_args(host, "--campaign", "local", *DEFER), lp.JobContext({}), gpu_query=lambda: []) == 2
    assert lp.parse_request("4, 12") == {"cpus": 4.0, "mem_gb": 12.0, "gpus": 0.0}
    for bad in ("8", "0,4", "8,32,-1", "a,b"):
        with pytest.raises(SystemExit):
            lp.build_parser().parse_args(["run", "--defer", bad, "--", "x"])


# ───────────────────────────── command line ─────────────────────────────

def test_host_events_record_what_shares_the_host(host):
    entry("mpr", "cpu4", req(4, 8), home=host["home"], kind="active")
    entry("jigsaw", "260929-120500-q-cd34", req(2, 4), home=host["home"], kind="active")
    rc = lp.cmd_run(run_args(host, "--load-log-s", "0.2", child=("0.8", "0")), lp.JobContext(host["environ"]),
                    gpu_query=lambda: [])
    assert rc == 0 and events(host["state"]) == ["start", "code", "child", "done"]
    logged = [r for r in records(host["state"]) if r["event"] == "host"]
    assert len(logged) >= 2 and records(host["state"])[-2]["event"] == "host"  # the last one closes the run
    first = logged[0]
    assert [(o["key"], o["cpus"]) for o in first["others"]] == [("jigsaw--260929-120500-q-cd34", 2.0),
                                                                ("mpr--cpu4", 4.0)]
    for r in logged:
        assert 0.0 <= r["cpu_ours_pct"] <= 100.0 and r["cpu_other_pct"] >= 0.0 and r["interval_s"] >= 0.0
    assert lp.cmd_run(run_args(host, "--campaign", "quiet", "--load-log-s", "0", child=("0.1", "0")),
                      lp.JobContext(host["environ"]), gpu_query=lambda: []) == 0
    assert [r for r in records(host["state"], "quiet") if r["event"] == "host"] == []


def test_status_and_probe(host, capsys, monkeypatch):
    assert lp.cmd_run(run_args(host, child=("0.1", "0")), lp.JobContext(host["environ"]), gpu_query=lambda: []) == 0
    capsys.readouterr()
    assert lp.main(["status", "--campaign", "cal", "--state-dir", str(host["state"])]) == 0
    out = capsys.readouterr().out.splitlines()
    summary = json.loads(out[-1])
    assert summary["host_events"] >= 1 and summary["events"] == 4 + summary["host_events"]
    assert summary["stop_requested"] is False and summary["code"] and summary["host_events_shared"] == 0
    for k, v in host["environ"].items():
        monkeypatch.setenv(k, v)
    assert lp.cmd_probe(lp.build_parser().parse_args(["probe"]), gpu_query=lambda: []) == 0
    probe = json.loads(capsys.readouterr().out)
    assert probe["rx_contention"] is None and probe["project"] == "jigsaw" and len(probe["active"]) == 1
    with pytest.raises(lp.LowprioError):
        lp.campaign_paths(host["state"], "bad name")


def test_status_and_stop_find_the_campaign_without_job_variables(host, capsys, monkeypatch):
    """`rx exec` passes no job variables: status and stop fall back to rx.toml's and refuse an unknown campaign."""
    assert lp.cmd_run(run_args(host, child=("0.1", "0")), lp.JobContext(host["environ"]), gpu_query=lambda: []) == 0
    monkeypatch.delenv("JIGSAW_LOWPRIO_ROOT", raising=False)
    toml = host["tmp"] / "rx.toml"
    toml.write_text("[run]\nenv_vars = { JIGSAW_LOWPRIO_ROOT = '%s' }\n" % host["state"], encoding="utf-8")
    monkeypatch.setattr(lp, "RX_TOML", host["tmp"] / "absent.toml")
    capsys.readouterr()
    assert lp.main(["status", "--campaign", "cal"]) == 2  # outputs/lowprio of the workspace holds nothing
    assert "no campaign cal under" in capsys.readouterr().err
    assert lp.main(["stop", "--campaign", "cal"]) == 2 and not Path("outputs", "lowprio", "cal.STOP").exists()
    monkeypatch.setattr(lp, "RX_TOML", toml)
    assert lp.campaign_state_dir(None) == host["state"] and lp.campaign_state_dir("elsewhere") == Path("elsewhere")
    assert lp.main(["status", "--campaign", "cal", "--last", "1"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["events"] >= 4
    assert lp.main(["stop", "--campaign", "cal"]) == 0 and (host["state"] / "cal.STOP").exists()
    assert lp.main(["status", "--campaign", "other"]) == 2
    monkeypatch.setenv("JIGSAW_LOWPRIO_ROOT", str(host["tmp"] / "env-root"))
    assert lp.campaign_state_dir(None) == host["tmp"] / "env-root"  # a job's own variable comes first
    toml.write_text("[run\n", encoding="utf-8")
    monkeypatch.delenv("JIGSAW_LOWPRIO_ROOT")
    assert lp.rx_toml_var("JIGSAW_LOWPRIO_ROOT") is None and lp.campaign_state_dir(None) == Path("outputs", "lowprio")


def test_child_runs_below_normal_priority(tmp_path):
    out = tmp_path / "nice.txt"
    code = f"import psutil; open({str(out)!r}, 'w').write(str(int(psutil.Process().nice())))"
    env = {k: v for k, v in os.environ.items() if not k.startswith("RX_")}
    env["PYTHONPATH"] = str(REPO)  # run from an empty directory: the campaign fingerprints its code paths
    proc = subprocess.run([sys.executable, "-m", "scripts.canonical.launch.lowprio", "run", "--campaign", "prio",
                           "--state-dir", str(tmp_path / "state"), "--poll-s", "0.05", "--gpu-policy", "ignore",
                           "--", sys.executable, "-c", code], cwd=str(tmp_path), env=env, capture_output=True,
                          timeout=120)
    assert proc.returncode == 0, proc.stdout.decode() + proc.stderr.decode()
    nice = int(out.read_text())
    if os.name == "nt":
        assert nice == psutil.BELOW_NORMAL_PRIORITY_CLASS
    else:
        assert nice >= 10
