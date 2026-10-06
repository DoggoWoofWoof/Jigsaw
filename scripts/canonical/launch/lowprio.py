"""Run a long job on the shared rx host at lower priority than every other tenant.

    rx run --cpus 1e-11 --mem 1e-11 --ws train1 --name cal-corafull -- python -m scripts.canonical.launch.lowprio run \\
        --campaign cal1 --defer 8,32,1 --requeue --progress "RUNS/train/**/ckpt_epoch*.pt" -- \\
        python -m scripts.canonical.launch.train_matrix --execute ...

rx (tools/rx of the message-passing-retrieval checkout) admits jobs by project priority
(host.json's `priority` map), then first in, first out with backfill, and never preempts.
This wrapper adds the preemption, with Jigsaw's jobs the ones that give way:

1. The job and its children run at below-normal CPU priority and low I/O priority.
2. The job yields when another project's queued rx job is blocked and would fit if Jigsaw
   jobs returned their resources, and this job is among the fewest that must (taken
   youngest first; a job holding nothing the blocked job lacks keeps running), or while it
   holds a GPU on which a foreign compute process is running. Contention must persist for
   --confirm-s; a job that meets it before its child starts yields at once.
   Without --yield-to every other project comes first. With --yield-to mpr only mpr's
   queued jobs make it yield at once; another project's queued job does once it has waited
   --peer-wait-s (default rx's hold_after_s), so no project waits on Jigsaw without bound.
   A GPU process is foreign unless it runs under an active rx job of this project or, with
   --yield-to, of a project not named there: processes outside rx (other users) and those of
   the named projects' jobs are foreign. Jigsaw jobs therefore share a GPU as rx assigns it.
3. A yield waits up to --grace-s for new progress (a file matching --progress, such as
   the next checkpoint), stops the child tree and exits 75. The trainer is kill-safe and
   resumes from its last checkpoint, so a yield loses at most the work since then.
4. With --requeue the yielding job launches a waiter through the host's rx agent: a job
   that asks for no GPU and less CPU and memory than rx's fit tolerance (WAITER_REQ), so
   it never keeps another job out. The waiter relaunches the full job once rx would admit
   it (older queued jobs that fit go first, as in rx's decide), no queued job it must make
   room for (rule 2) is waiting and, for a GPU job, no foreign compute process has run for
   --gpu-quiet-s. A relaunch still queued after --max-queue-s is
   cancelled and retried later, so a waiting Jigsaw job never reaches rx's hold rule
   (hold_after_s), under which an old blocked job stops younger jobs from starting.
5. With --defer CPUS,MEM_GB[,GPUS] the first hop is such a waiter too: launched at the
   waiter's size, it records the code and launches the job with that request once the
   host has room, so no Jigsaw job ever waits in rx's queue at full size.

Every hop checks that the workspace code equals the campaign's first hop
(manifest.code_fingerprint); a campaign never mixes code. Events go to
<state-dir>/<campaign>.jsonl. While the child runs, `host` events record what shares the
machine (the other active rx jobs and the CPU time spent outside this job's process tree,
see LoadLog), so every wall time can be checked against a quiet host. To stop a campaign, cancel its current job with rx
(the running job or its waiter), or create <state-dir>/<campaign>.STOP: the current job
finishes or yields and nothing new is launched.

    python -m scripts.canonical.launch.lowprio status --campaign cal1 [--state-dir DIR]
    python -m scripts.canonical.launch.lowprio stop --campaign cal1 [--clear] [--force]
    python -m scripts.canonical.launch.lowprio probe      # this host's view, as a job sees it

status and stop look in --state-dir, else $JIGSAW_LOWPRIO_ROOT, else the directory rx.toml
gives rx jobs (`rx exec` passes none of rx.toml's job variables), and exit 2 when the campaign
has no log there (stop --force stops a campaign that has not started).

The scheduler arithmetic mirrors rx_agent.py (_fit, _take, _hold_back, project_priority,
decide, heartbeat_alive); this module reads rx's state files and never writes them.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .. import manifest

EXIT_YIELD = 75  # EX_TEMPFAIL: the job stepped aside and did not finish
EXIT_CODE_CHANGED = 3
EXIT_GAVE_UP = 4
LOST_AFTER_S = 120.0  # rx_agent.LOST_AFTER_S: a heartbeat older than this is a dead entry
NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,120}$")  # rx_agent.NAME_RE
# A waiter asks for less than rx's fit tolerance (1e-9 in rx_agent._fit), so up to 100 waiters never keep
# another job out: 0.1 CPU would block a job that asks for exactly the whole CPUs left. Its process (a poll
# every 30 s) runs inside the host reserve, like rx's own job supervisors.
WAITER_REQ = {"cpus": 1e-11, "mem_gb": 1e-11, "gpus": 0.0}
LAUNCH_FIELDS = ("created", "agent", "_fetch")  # set by rx at launch; a relaunch gets its own
IS_WIN = os.name == "nt"
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
CREATE_NO_WINDOW = 0x08000000
RX_TOML = manifest.REPO_ROOT / "rx.toml"


class LowprioError(RuntimeError):
    pass


# ───────────────────────────── scheduler arithmetic (mirrors rx) ─────────────────────────────

Pool = tuple[float, float, dict[int, float]]


def job_request(spec: Mapping[str, Any]) -> dict[str, float]:
    return {"cpus": float(spec.get("cpus") or 1), "mem_gb": float(spec.get("mem_gb") or 1),
            "gpus": float(spec.get("gpus") or 0)}


def fit(req: Mapping[str, float], cpu: float, mem: float, gfree: Mapping[int, float]) -> dict[str, Any] | None:
    """rx_agent._fit: the assignment of req in the pool, or None."""
    if req["cpus"] > cpu + 1e-9 or req["mem_gb"] > mem + 1e-9:
        return None
    g = req.get("gpus") or 0
    shares: dict[int, float] = {}
    if g > 0:
        if g < 1:
            cands = sorted((v, i) for i, v in gfree.items() if v >= g - 1e-9)
            if not cands:
                return None
            shares = {cands[0][1]: g}
        else:
            n = int(round(g))
            full = sorted(i for i, v in gfree.items() if v >= 1 - 1e-9)
            if len(full) < n:
                return None
            shares = {i: 1.0 for i in full[:n]}
    return {"cpus": req["cpus"], "mem_gb": req["mem_gb"], "gpus": sorted(shares),
            "gpu_shares": {str(i): s for i, s in shares.items()}}


def take(pool: Pool, assign: Mapping[str, Any], sign: float = 1.0) -> Pool:
    """rx_agent._take; sign -1 returns the assignment to the pool."""
    cpu, mem, g = pool
    g = dict(g)
    for i, s in (assign.get("gpu_shares") or {}).items():
        g[int(i)] = g.get(int(i), 0) - sign * s
    return cpu - sign * assign["cpus"], mem - sign * assign["mem_gb"], g


def capacity_pool(cap: Mapping[str, Any]) -> Pool:
    return (cap["cpus"] - cap.get("reserve_cpus", 0), cap["mem_gb"] - cap.get("reserve_mem_gb", 0),
            {int(g["index"]): 1.0 for g in cap.get("gpus") or []})


def entry_assign(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    return entry.get("assign") or {"cpus": entry["req"]["cpus"], "mem_gb": entry["req"]["mem_gb"]}


def free_pool(cap: Mapping[str, Any], active: Sequence[Mapping[str, Any]]) -> Pool:
    pool = capacity_pool(cap)
    for a in active:
        pool = take(pool, entry_assign(a))
    return pool


def project_priority(cap: Mapping[str, Any], project: str | None) -> float:
    """rx_agent.project_priority: host.json's `priority` map; a project not in it is 0."""
    try:
        return float((cap.get("priority") or {}).get(project) or 0)
    except (TypeError, ValueError, AttributeError):
        return 0.0


def hold_back(pool: Pool, req: Mapping[str, float]) -> Pool:
    """rx_agent._hold_back: the pool once a waiting higher-priority job's request is set aside."""
    cpu, mem, g = pool
    g = dict(g)
    need = float(req.get("gpus") or 0)
    for i in sorted(g, key=lambda i: (-g[i], i)):
        if need <= 1e-9:
            break
        part = min(max(g[i], 0.0), need)
        g[i] -= part
        need -= part
    return max(0.0, cpu - req["cpus"]), max(0.0, mem - req["mem_gb"]), g


def queue_order(cap: Mapping[str, Any], q: Mapping[str, Any]) -> tuple[float, float, str]:
    """rx's queue order: a higher-priority project first, then first in, first out."""
    return -project_priority(cap, q.get("project")), float(q.get("enq") or 0), str(q.get("key") or "")


def decide(me: Mapping[str, Any], cap: Mapping[str, Any], active: Sequence[Mapping[str, Any]],
           queue: Sequence[Mapping[str, Any]], now: float) -> tuple[bool, dict[str, Any] | None, str | None]:
    """rx_agent.decide: (admit, assignment, reason) for the queued job `me`."""
    req = me["req"]
    if fit(req, *capacity_pool(cap)) is None:
        return False, None, "impossible on this host"
    pool = free_pool(cap, active)
    if fit(req, *pool) is None:
        return False, None, "waiting for resources"
    hold = float(cap.get("hold_after_s", 1800))
    mine_p = project_priority(cap, me.get("project"))
    mine_key = queue_order(cap, me)
    higher = None
    for q in sorted(queue, key=lambda q: queue_order(cap, q)):
        if queue_order(cap, q) >= mine_key:
            continue
        above = project_priority(cap, q.get("project")) > mine_p
        higher = higher or (q if above else None)
        placed = fit(q["req"], *pool)
        if placed is not None:
            pool = take(pool, placed)  # it goes first
        elif above:
            pool = hold_back(pool, q["req"])  # what it needs is not ours to take
        elif now - float(q.get("enq") or now) > hold:
            return False, None, f"holding for older job {q.get('id')} (queued {(now - float(q['enq'])) / 60:.0f} min)"
    mine = fit(req, *pool)
    if mine is None:
        if higher is not None:
            return False, None, f"waiting for higher-priority job {higher.get('id')} (project {higher.get('project')})"
        return False, None, "waiting behind older queued jobs"
    return True, mine, None


def must_yield_to(cap: Mapping[str, Any], q: Mapping[str, Any], *, project: str, now: float,
                  yield_to: Sequence[str] | None = None, peer_wait_s: float | None = None) -> bool:
    """Whether a queued job may make a Jigsaw job yield. Without yield_to, any other project's job may.
    With it, a job of a named project or of one that rx ranks above this project may at once, and any
    other project's job may once it has waited peer_wait_s (default rx's hold_after_s)."""
    other = q.get("project")
    if other == project:
        return False
    if yield_to is None or other in yield_to or project_priority(cap, other) > project_priority(cap, project):
        return True
    wait = float(cap.get("hold_after_s", 1800)) if peer_wait_s is None else float(peer_wait_s)
    return now - float(q.get("enq") or now) >= wait


def rx_contention(cap: Mapping[str, Any], active: Sequence[Mapping[str, Any]], queue: Sequence[Mapping[str, Any]],
                  *, project: str, key: str, now: float, yield_to: Sequence[str] | None = None,
                  peer_wait_s: float | None = None) -> dict[str, Any] | None:
    """The first live queued job of another project that waits for resources this job holds.

    Queued jobs are taken in rx's order, and decide() says whether each is admitted. For a refused
    job that may make Jigsaw yield (must_yield_to), this project's active jobs are returned youngest
    first until decide() admits it, and then every returned job it can do without is given back,
    youngest first. This job yields when it is in the set that remains: a job whose resources the
    blocked job does not need (CPUs when it waits for the GPU, a waiter's share) keeps running.
    """
    ours = sorted((a for a in active if a.get("project") == project),
                  key=lambda a: (a.get("admitted") or 0, a.get("key") or ""), reverse=True)
    if not any(a.get("key") == key for a in ours):
        return None

    def admitted(q: Mapping[str, Any], released: Sequence[Mapping[str, Any]]) -> bool:
        gone = {a.get("key") for a in released}
        return decide(q, cap, [a for a in active if a.get("key") not in gone], queue, now)[0]

    total = capacity_pool(cap)
    for q in sorted(queue, key=lambda q: queue_order(cap, q)):
        if not must_yield_to(cap, q, project=project, now=now, yield_to=yield_to, peer_wait_s=peer_wait_s):
            continue
        if fit(q["req"], *total) is None or admitted(q, []):
            continue
        chosen: list[Mapping[str, Any]] = []
        for a in ours:
            chosen.append(a)
            if admitted(q, chosen):
                break
        else:
            continue  # it is refused even without all of ours
        for a in list(chosen):
            rest = [c for c in chosen if c is not a]
            if admitted(q, rest):
                chosen = rest
        if any(a.get("key") == key for a in chosen):
            return {"kind": "rx", "job": q.get("id"), "project": q.get("project"), "req": dict(q["req"]),
                    "queued_s": round(now - float(q.get("enq") or now), 1)}
    return None


NEXT_ID = "next-hop"  # the id of the job a waiter is about to launch, in launch_blocker's what-if


def launch_blocker(cap: Mapping[str, Any], active: Sequence[Mapping[str, Any]], queue: Sequence[Mapping[str, Any]],
                   req: Mapping[str, float], *, project: str, now: float, yield_to: Sequence[str] | None = None,
                   peer_wait_s: float | None = None) -> str | None:
    """Why a waiter should not launch a job with request `req` now, or None. rx must admit the job as
    one queued now (decide), and once running it must not be a job that rx_contention would make
    yield, so a waiter never launches what the next contention check stops."""
    me = {"key": f"{project}--{NEXT_ID}", "project": project, "id": NEXT_ID, "req": dict(req), "enq": now}
    ok, assign, why = decide(me, cap, active, queue, now)
    if not ok:
        return why
    running = list(active) + [dict(me, assign=assign, admitted=now)]
    blocked = rx_contention(cap, running, queue, project=project, key=me["key"], now=now, yield_to=yield_to,
                            peer_wait_s=peer_wait_s)
    if blocked:
        return f"it would have to yield to {blocked['project']} job {blocked['job']}"
    return None


# ───────────────────────────── reading rx's state (read only) ─────────────────────────────

def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def boot_time() -> float:
    try:
        import psutil

        return float(psutil.boot_time())
    except Exception:
        return 0.0


def heartbeat_alive(jd: str, now: float, boot: float) -> bool:
    hb = read_json(os.path.join(jd, "heartbeat.json")) if jd else None
    if not hb:
        return False
    t = float(hb.get("t") or 0)
    return now - t <= LOST_AFTER_S and not (boot and t < boot - 5)


def live_entries(home: str | os.PathLike, kind: str, now: float, boot: float) -> list[dict[str, Any]]:
    """rx's active/ or queue/ entries whose supervisor still beats; stale files are left to rx."""
    out = []
    d = Path(home) / kind
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return out
    for name in names:
        if not name.endswith(".json"):
            continue
        entry = read_json(d / name)
        if isinstance(entry, dict) and "req" in entry and heartbeat_alive(entry.get("jd", ""), now, boot):
            out.append(entry)
    return out


class HostView:
    """One consistent-enough snapshot of host.json, active/ and queue/."""

    def __init__(self, home: str | os.PathLike, now: float | None = None) -> None:
        self.home = Path(home)
        self.now = time.time() if now is None else now
        boot = boot_time()
        self.cap = read_json(self.home / "host.json") or {}
        self.active = live_entries(self.home, "active", self.now, boot)
        self.queue = live_entries(self.home, "queue", self.now, boot)

    @property
    def usable(self) -> bool:
        return bool(self.cap.get("cpus")) and bool(self.cap.get("mem_gb"))


# ───────────────────────────── GPU processes ─────────────────────────────

def nvidia_smi_argv() -> list[str] | None:
    exe = os.environ.get("JIGSAW_NVIDIA_SMI") or shutil.which("nvidia-smi")
    if not exe and IS_WIN:
        cand = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
        exe = cand if os.path.exists(cand) else None
    if not exe:
        return None
    return [sys.executable, exe] if exe.endswith(".py") else [exe]


def parse_gpu_processes(xml_text: str) -> list[dict[str, Any]]:
    """[{gpu, pid, type, name}] from `nvidia-smi -q -x`; gpu is the position in PCI bus order."""
    root = ET.fromstring(xml_text)
    out = []
    for index, gpu in enumerate(root.findall("gpu")):
        for info in gpu.findall("./processes/process_info"):
            try:
                pid = int((info.findtext("pid") or "").strip())
            except ValueError:
                continue
            out.append({"gpu": index, "pid": pid, "type": (info.findtext("type") or "").strip(),
                         "name": (info.findtext("process_name") or "").strip()})
    return out


def gpu_processes(timeout: float = 30.0) -> list[dict[str, Any]] | None:
    """The GPU processes nvidia-smi reports, or None when it cannot be asked."""
    argv = nvidia_smi_argv()
    if argv is None:
        return None
    try:
        proc = subprocess.run(argv + ["-q", "-x"], capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL,
                              **({"creationflags": CREATE_NO_WINDOW} if IS_WIN else {}))
        if proc.returncode != 0:
            return None
        return parse_gpu_processes(proc.stdout.decode("utf-8", "replace"))
    except (OSError, subprocess.TimeoutExpired, ET.ParseError):
        return None


def own_pids() -> set[int]:
    import psutil

    me = psutil.Process()
    pids = {me.pid}
    try:
        pids.update(p.pid for p in me.children(recursive=True))
    except psutil.Error:
        pass
    return pids


def status_pids(entry: Mapping[str, Any]) -> list[int]:
    """The supervisor's and the job process's pids that rx keeps in an active job's status.json."""
    st = read_json(Path(entry["jd"]) / "status.json") if entry.get("jd") else None
    out = []
    for k in ("supervisor_pid", "pid"):
        try:
            pid = int((st or {}).get(k) or 0)
        except (TypeError, ValueError):
            continue
        if pid > 0:
            out.append(pid)
    return out


def process_lineage(pid: int) -> list[int]:
    """pid and its ancestors, nearest first. psutil drops a parent younger than its child (a reused pid)."""
    try:
        import psutil

        return [pid] + [p.pid for p in psutil.Process(pid).parents()]
    except Exception:
        return [pid]


def gpu_process_jobs(procs: Sequence[Mapping[str, Any]], active: Sequence[Mapping[str, Any]],
                     lineage: Callable[[int], list[int]] | None = None) -> dict[int, Mapping[str, Any]]:
    """{pid: active rx job} for the GPU processes that run under an rx job, found through their ancestors."""
    lineage = lineage or process_lineage
    owner: dict[int, Mapping[str, Any]] = {}
    for a in active:
        for pid in status_pids(a):
            owner.setdefault(pid, a)
    out: dict[int, Mapping[str, Any]] = {}
    for p in procs if owner else ():
        for pid in lineage(p["pid"]):
            if pid in owner:
                out[p["pid"]] = owner[pid]
                break
    return out


def shared_gpu_pids(procs: Sequence[Mapping[str, Any]], cap: Mapping[str, Any], active: Sequence[Mapping[str, Any]],
                    *, project: str | None, yield_to: Sequence[str] | None,
                    lineage: Callable[[int], list[int]] | None = None) -> set[int]:
    """GPU processes of rx jobs that share a GPU with this job as rx assigns it, so none is foreign: those
    of this project's jobs and, under yield_to, of a project neither named there nor ranked above this one."""
    keep = set()
    for pid, job in gpu_process_jobs(procs, active, lineage).items():
        other = job.get("project")
        if other == project or (yield_to is not None and other not in yield_to
                                and project_priority(cap, other) <= project_priority(cap, project)):
            keep.add(pid)
    return keep


def foreign_gpu_processes(procs: Sequence[Mapping[str, Any]], *, gpus: Sequence[int], mine: set[int],
                          types: Sequence[str], ignore: Sequence[str]) -> list[dict[str, Any]]:
    """Compute processes on these GPUs outside `mine` (this job's tree and the GPU's other sharers)."""
    wanted = {t.strip().upper() for t in types}
    out = []
    for p in procs:
        if p["gpu"] not in gpus or p["pid"] in mine or p["type"].upper() not in wanted:
            continue
        base = re.split(r"[\\/]", p["name"])[-1].lower()
        if any(base == n.lower() or p["name"].lower() == n.lower() for n in ignore):
            continue
        out.append(dict(p))
    return out


# ───────────────────────────── the job's own identity ─────────────────────────────

class JobContext:
    """What rx tells a job about itself (RX_* variables and its job directory)."""

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        env = os.environ if environ is None else environ
        self.home = env.get("RX_HOME")
        self.job_dir = env.get("RX_JOB_DIR")
        self.project = env.get("RX_PROJECT")
        self.job_id = env.get("RX_JOB_ID")
        self.gpus = [int(x) for x in (env.get("RX_GPUS") or "").split(",") if x.strip().isdigit()]
        self.spec = read_json(Path(self.job_dir) / "spec.json", {}) if self.job_dir else {}
        self.status = read_json(Path(self.job_dir) / "status.json", {}) if self.job_dir else {}

    @property
    def under_rx(self) -> bool:
        return bool(self.home and self.job_dir and self.project and self.job_id and self.spec)

    @property
    def key(self) -> str:
        return f"{self.project}--{self.job_id}"

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self.spec.get("lowprio") or {})


def make_id(name: str) -> str:
    """rx.make_id: yymmdd-HHMMSS-<name>-<4 hex>."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name or "job").strip("-._")[:40] or "job"
    return "%s-%s-%s" % (time.strftime("%y%m%d-%H%M%S"), name, secrets.token_hex(2))


def job_template(spec: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    """The spec of the next hop of this job, without an id: same command, request and workspace."""
    target = {k: v for k, v in spec.items() if k not in LAUNCH_FIELDS and k not in ("id", "lowprio_target")}
    target["lowprio"] = dict(meta, role="job")
    target["rerun_of"] = meta["origin"]
    target["_fetch"] = False
    return target


def waiter_spec(spec: Mapping[str, Any], target: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    """A small job that relaunches `target` when the host has room for it."""
    name = f"{spec.get('name') or 'job'}-wait"
    waiter = {k: v for k, v in spec.items() if k not in LAUNCH_FIELDS and k not in ("lowprio_target",)}
    waiter.update({
        "id": make_id(name), "name": name, "shell": "exec", "command": None,
        "argv": ["python", "-m", "scripts.canonical.launch.lowprio", "wait"],
        "cpus": WAITER_REQ["cpus"], "mem_gb": WAITER_REQ["mem_gb"], "gpus": WAITER_REQ["gpus"],
        "mem_hard_gb": None, "timeout_s": None, "outputs": ["outputs/lowprio/**"], "outputs_exclude": [],
        "display": f"lowprio wait -> {target.get('display') or target.get('name')}",
        "rerun_of": meta["origin"], "lowprio": dict(meta, role="waiter"), "lowprio_target": dict(target),
        "_fetch": False,
    })
    return waiter


def launched_spec(target: Mapping[str, Any]) -> dict[str, Any]:
    spec = dict(target)
    spec["id"] = make_id(str(target.get("name") or "job"))
    return spec


# ───────────────────────────── the host agent ─────────────────────────────

def _is_supervisor(pid: int, job_dir: str) -> str | None:
    try:
        import psutil

        proc = psutil.Process(pid)
        cmd = proc.cmdline()
        if "supervise" in cmd and any(os.path.normcase(os.path.abspath(c)) == os.path.normcase(os.path.abspath(job_dir))
                                      for c in cmd if c):
            return proc.exe()
    except Exception:
        return None
    return None


def rx_python(ctx: JobContext) -> str:
    """The interpreter rx runs its agent with: this job's supervisor's, else the home's standalone one."""
    override = os.environ.get("JIGSAW_RX_PYTHON")
    if override:
        return override
    pid = (ctx.status or {}).get("supervisor_pid")
    if pid and ctx.job_dir:
        exe = _is_supervisor(int(pid), ctx.job_dir)
        if exe:
            return exe
    if ctx.home:
        pattern = os.path.join(ctx.home, "python", "*", "python.exe" if IS_WIN else os.path.join("bin", "python3"))
        found = sorted(glob.glob(pattern))
        if found:
            return found[-1]
    return sys.executable


def agent_path(ctx: JobContext) -> str:
    folder = Path(ctx.home or "") / "agent"
    sha = str(ctx.spec.get("agent") or "")
    exact = folder / f"rx_agent-{sha}.py"
    if sha and exact.is_file():
        return str(exact)
    found = sorted(folder.glob("rx_agent-*.py"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise LowprioError(f"no rx agent under {folder}")
    return str(found[-1])


def agent_call(ctx: JobContext, request: Mapping[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    """One request to the host's rx agent (`rx_agent.py rpc`), as rx.py sends it over ssh."""
    env = dict(os.environ)
    env["RX_HOME"] = str(ctx.home)
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "VIRTUAL_ENV", "RX_LAUNCH"):
        env.pop(k, None)
    proc = subprocess.run([rx_python(ctx), agent_path(ctx), "rpc"],
                          input=(json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8"),
                          capture_output=True, timeout=timeout, env=env,
                          **({"creationflags": CREATE_NO_WINDOW} if IS_WIN else {}))
    lines = proc.stdout.decode("utf-8", "replace").splitlines()
    try:
        res = json.loads(lines[0]) if lines else None
    except ValueError:
        res = None
    if not isinstance(res, dict):
        raise LowprioError(f"rx agent answered rc {proc.returncode}: "
                           f"{proc.stdout.decode('utf-8', 'replace')[-300:]} {proc.stderr.decode('utf-8', 'replace')[-300:]}")
    if not res.get("ok"):
        raise LowprioError(f"rx agent refused {request.get('op')}: {res.get('error')}")
    return res


def launch(ctx: JobContext, spec: Mapping[str, Any], tries: int = 3) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            return agent_call(ctx, {"op": "launch", "spec": dict(spec)})
        except (LowprioError, OSError, subprocess.TimeoutExpired) as exc:
            last = exc
            time.sleep(min(60.0, 5.0 * (attempt + 1)))
    raise LowprioError(f"launch of {spec.get('id')} failed: {last}")


def job_state(ctx: JobContext, project: str, jid: str) -> str | None:
    st = read_json(Path(ctx.home or "") / "projects" / project / "jobs" / jid / "status.json")
    return st.get("state") if isinstance(st, dict) else None


# ───────────────────────────── campaign state ─────────────────────────────

def state_dir(arg: str | None) -> Path:
    return Path(arg or os.environ.get("JIGSAW_LOWPRIO_ROOT") or os.path.join("outputs", "lowprio"))


def rx_toml_var(name: str) -> str | None:
    """A [run] env_vars value of the checkout's rx.toml (local only; rx pushes it with the code)."""
    import tomllib

    try:
        with open(RX_TOML, "rb") as handle:
            value = ((tomllib.load(handle).get("run") or {}).get("env_vars") or {}).get(name)
    except (OSError, ValueError):
        return None
    return str(value) if value else None


def campaign_state_dir(arg: str | None) -> Path:
    """The state directory of status and stop. `rx exec` passes none of rx.toml's job variables, so
    without --state-dir or $JIGSAW_LOWPRIO_ROOT this is the directory rx.toml gives rx jobs."""
    if arg or os.environ.get("JIGSAW_LOWPRIO_ROOT"):
        return state_dir(arg)
    value = rx_toml_var("JIGSAW_LOWPRIO_ROOT")
    return Path(value) if value else state_dir(None)


def campaign_paths(root: Path, campaign: str) -> dict[str, Path]:
    if not NAME_RE.match(campaign):
        raise LowprioError(f"campaign name {campaign!r}: letters, digits, '.', '_', '-' only")
    return {"log": root / f"{campaign}.jsonl", "stop": root / f"{campaign}.STOP", "code": root / f"{campaign}.code.json",
            "stop_all": root / "ALL.STOP"}


def log_event(root: Path, campaign: str, event: str, **fields: Any) -> None:
    rec = {"utc": manifest.utc_now(), "t": round(time.time(), 3), "campaign": campaign, "event": event, **fields}
    line = json.dumps(rec, sort_keys=True, default=str)
    print(f"[lowprio {time.strftime('%H:%M:%S')}] {event} " + json.dumps(fields, sort_keys=True, default=str),
          flush=True)
    try:
        root.mkdir(parents=True, exist_ok=True)
        with open(campaign_paths(root, campaign)["log"], "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError as exc:
        print(f"[lowprio] cannot write the campaign log: {exc}", flush=True)


def stop_requested(root: Path, campaign: str) -> bool:
    paths = campaign_paths(root, campaign)
    return paths["stop"].exists() or paths["stop_all"].exists()


def check_code(root: Path, campaign: str, workspace: Path, *, first: bool) -> tuple[bool, str]:
    """(same, detail): the workspace's code fingerprint against the campaign's first hop."""
    fp = manifest.code_fingerprint(workspace)
    path = campaign_paths(root, campaign)["code"]
    recorded = read_json(path)
    if recorded is not None and first and recorded.get("tree_sha256") != fp["tree_sha256"]:
        return False, (f"campaign {campaign} already ran code {recorded.get('tree_sha256', '')[:12]}; this workspace "
                       f"has {fp['tree_sha256'][:12]}: start a new --campaign")
    if recorded is None:
        if not first:
            return False, f"{path} is missing; a relaunch cannot prove its code equals the first hop's"
        root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(fp, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return True, f"recorded code {fp['tree_sha256'][:12]} ({len(fp['files'])} files)"
    if recorded.get("tree_sha256") == fp["tree_sha256"]:
        return True, f"code {fp['tree_sha256'][:12]} unchanged"
    changed = sorted(set(recorded.get("files", {}).items()) ^ set(fp["files"].items()))
    names = sorted({rel for rel, _ in changed})[:10]
    return False, f"code {fp['tree_sha256'][:12]} differs from the campaign's {recorded.get('tree_sha256', '')[:12]}: {names}"


# ───────────────────────────── priority and the child ─────────────────────────────

def lower_priority() -> str:
    try:
        import psutil

        me = psutil.Process()
        if IS_WIN:
            me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            try:
                me.ionice(psutil.IOPRIO_LOW)
            except (psutil.Error, AttributeError, ValueError):
                pass
        else:
            os.nice(10)
            try:
                me.ionice(psutil.IOPRIO_CLASS_BE, value=7)
            except (psutil.Error, AttributeError, ValueError):
                pass
        return "below normal" if IS_WIN else "nice 10"
    except Exception as exc:  # the job still runs; the log says why the priority did not change
        return f"unchanged ({type(exc).__name__}: {exc})"


def start_child(cmd: Sequence[str]) -> subprocess.Popen:
    """Start cmd below normal priority; a bare python runs this interpreter, as rx maps it."""
    cmd = list(cmd)
    if cmd and cmd[0] in ("python", "python3", "py"):
        cmd[0] = sys.executable
    kw: dict[str, Any] = {"stdin": subprocess.DEVNULL}
    if IS_WIN:
        kw["creationflags"] = BELOW_NORMAL_PRIORITY_CLASS
    return subprocess.Popen(cmd, **kw)


def stop_tree(proc: subprocess.Popen, timeout: float = 60.0) -> int:
    """Stop the child and its descendants; returns how many processes were stopped."""
    import psutil

    try:
        parent = psutil.Process(proc.pid)
        procs = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return 0
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=timeout)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    return len(procs)


def latest_progress(patterns: Sequence[str]) -> float:
    best = 0.0
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            try:
                best = max(best, os.path.getmtime(path))
            except OSError:
                continue
    return best


# ───────────────────────────── contention ─────────────────────────────

class Watch:
    """Contention as this job sees it; `sample` returns the first reason or None."""

    def __init__(self, ctx: JobContext, args: argparse.Namespace,
                 gpu_query: Callable[[], list[dict[str, Any]] | None] = gpu_processes) -> None:
        self.ctx, self.args, self.gpu_query = ctx, args, gpu_query
        self.policy = {"yield_to": getattr(args, "yield_to", None), "peer_wait_s": getattr(args, "peer_wait_s", None)}
        self.gpu_note_shown = False
        self.view: HostView | None = None  # the snapshot of the last rx() sample

    def rx(self) -> dict[str, Any] | None:
        if not self.ctx.under_rx:
            return None
        view = self.view = HostView(self.ctx.home)
        if not view.usable:
            return None
        return rx_contention(view.cap, view.active, view.queue, project=self.ctx.project, key=self.ctx.key,
                             now=view.now, **self.policy)

    def gpu(self, gpus: Sequence[int], view: HostView | None = None) -> dict[str, Any] | None:
        if not gpus or self.args.gpu_policy == "ignore":
            return None
        procs = self.gpu_query()
        if procs is None:
            if not self.gpu_note_shown:
                print("[lowprio] nvidia-smi gave no process list; foreign GPU processes are not watched", flush=True)
                self.gpu_note_shown = True
            return None
        view = view or self.view
        mine = own_pids()
        if view is not None and view.usable:
            mine |= shared_gpu_pids(procs, view.cap, view.active, project=self.ctx.project,
                                    yield_to=self.policy["yield_to"])
        foreign = foreign_gpu_processes(procs, gpus=gpus, mine=mine, types=self.args.gpu_types.split(","),
                                        ignore=self.args.ignore_gpu_process)
        if foreign:
            return {"kind": "gpu", "processes": [{"pid": p["pid"], "name": p["name"], "type": p["type"]}
                                                 for p in foreign[:5]]}
        return None

    def sample(self) -> dict[str, Any] | None:
        return self.rx() or self.gpu(self.ctx.gpus)


class LoadLog:
    """What shared the host with a running job: the other rx jobs and the CPU time outside its tree.

    due() returns the fields of a `host` event when the set of other active jobs changed or every_s
    passed: others (key, project, cpus, gpus of every other active job, this project's included),
    cpu_host_pct (the whole machine over the interval), cpu_ours_pct (this job's process tree) and
    cpu_other_pct (the rest: other jobs, other users, the system). A wall time is a quiet-host
    measurement only when every event of its interval has no others and a small cpu_other_pct.
    """

    def __init__(self, ctx: JobContext, every_s: float) -> None:
        import psutil

        self.ctx, self.every_s = ctx, float(every_s)
        self.ncpu = psutil.cpu_count() or 1
        self.t = time.monotonic()
        self.logged_at = 0.0
        self.jobs: list[str] | None = None
        self.cpu: dict[int, float] = {}
        psutil.cpu_percent(None)  # starts the machine-wide counter
        self._tree_seconds()

    def _tree_seconds(self) -> float:
        """CPU seconds this job's process tree spent since the previous call."""
        import psutil

        spent, now = 0.0, {}
        for pid in own_pids():
            try:
                times = psutil.Process(pid).cpu_times()
            except psutil.Error:
                continue
            now[pid] = times.user + times.system
            spent += now[pid] - self.cpu.get(pid, 0.0)
        self.cpu = now
        return spent

    def due(self, view: HostView | None, now: float, *, force: bool = False) -> dict[str, Any] | None:
        import psutil

        active = [] if view is None else [a for a in view.active if a.get("key") != self.ctx.key]
        jobs = sorted(str(a.get("key")) for a in active)
        if not force and jobs == self.jobs and now - self.logged_at < self.every_s:
            return None
        elapsed = max(now - self.t, 1e-6)
        host = float(psutil.cpu_percent(None))
        ours = min(100.0, 100.0 * self._tree_seconds() / (elapsed * self.ncpu))
        self.t, self.logged_at, self.jobs = now, now, jobs
        others = [{"key": a.get("key"), "project": a.get("project"), "cpus": entry_assign(a).get("cpus"),
                   "gpus": list(entry_assign(a).get("gpus") or [])} for a in active]
        return {"others": sorted(others, key=lambda o: str(o["key"])), "interval_s": round(elapsed, 1),
                "cpu_host_pct": round(host, 1), "cpu_ours_pct": round(ours, 1),
                "cpu_other_pct": round(max(0.0, host - ours), 1)}


# ───────────────────────────── run ─────────────────────────────

def _meta_for(ctx: JobContext, args: argparse.Namespace) -> dict[str, Any]:
    meta = ctx.meta
    if meta:
        return meta
    campaign = args.campaign or (ctx.spec.get("name") if ctx.under_rx else None) or "local"
    return {"campaign": campaign, "gen": 0, "origin": ctx.job_id or "local", "state_dir": str(state_dir(args.state_dir)),
            "requeue": bool(args.requeue), "max_requeues": int(args.max_requeues), "poll_s": float(args.poll_s),
            "settle_s": float(args.settle_s), "gpu_quiet_s": float(args.gpu_quiet_s),
            "max_queue_s": float(args.max_queue_s), "gpu_policy": args.gpu_policy, "gpu_types": args.gpu_types,
            "ignore_gpu_process": list(args.ignore_gpu_process), "yield_to": args.yield_to,
            "peer_wait_s": args.peer_wait_s}


def requeue(ctx: JobContext, meta: Mapping[str, Any], root: Path, reason: Mapping[str, Any] | None) -> str | None:
    campaign = meta["campaign"]
    if not meta.get("requeue"):
        log_event(root, campaign, "no_requeue", job=ctx.job_id, why="started without --requeue")
        return None
    if not ctx.under_rx:
        log_event(root, campaign, "no_requeue", job=ctx.job_id, why="not an rx job")
        return None
    if stop_requested(root, campaign):
        log_event(root, campaign, "no_requeue", job=ctx.job_id, why="stop requested")
        return None
    gen = int(meta.get("gen", 0)) + 1
    if gen > int(meta.get("max_requeues", 0)):
        log_event(root, campaign, "no_requeue", job=ctx.job_id, why=f"requeue {gen} exceeds --max-requeues")
        return None
    # the waiter starts its GPU quiet period at a yield to a foreign GPU process
    next_meta = dict(meta, gen=gen, last_yield={"kind": (reason or {}).get("kind"), "t": round(time.time(), 3)})
    target = job_template(ctx.spec, next_meta)
    waiter = waiter_spec(ctx.spec, target, next_meta)
    try:
        res = launch(ctx, waiter)
    except LowprioError as exc:
        log_event(root, campaign, "requeue_failed", job=ctx.job_id, error=str(exc))
        return None
    log_event(root, campaign, "requeued", job=ctx.job_id, waiter=waiter["id"], gen=gen,
              waiter_state=(res.get("job") or {}).get("state"), reason=reason)
    return waiter["id"]


def parse_request(text: str) -> dict[str, float]:
    """CPUS,MEM_GB[,GPUS] as an rx request."""
    parts = [p.strip() for p in str(text).split(",")]
    try:
        values = [float(p) for p in parts]
    except ValueError:
        values = []
    if len(values) == 2:
        values.append(0.0)
    if len(values) != 3 or values[0] <= 0 or values[1] <= 0 or values[2] < 0:
        raise argparse.ArgumentTypeError(f"{text!r}: expected CPUS,MEM_GB[,GPUS] with positive CPUS and MEM_GB")
    return {"cpus": values[0], "mem_gb": values[1], "gpus": values[2]}


def parse_projects(text: str) -> list[str]:
    """PROJECT[,PROJECT...] as a list of rx project names."""
    names = [p.strip() for p in str(text).split(",") if p.strip()]
    if not names or not all(NAME_RE.match(n) for n in names):
        raise argparse.ArgumentTypeError(f"{text!r}: expected PROJECT[,PROJECT...]")
    return names


def defer(ctx: JobContext, meta: Mapping[str, Any], root: Path, request: Mapping[str, float],
          gpu_query: Callable[[], list[dict[str, Any]] | None] = gpu_processes) -> int:
    """The first hop of `run --defer`: a waiter-sized job that launches the full request once the host has room.
    A Jigsaw job then never waits in rx's queue at full size, where hold_after_s would let it block younger jobs."""
    campaign = meta["campaign"]
    if not ctx.under_rx:
        log_event(root, campaign, "no_defer", job=ctx.job_id, why="not an rx job")
        return 2
    next_meta = dict(meta, gen=1)
    target = job_template(ctx.spec, next_meta)
    target.update(cpus=float(request["cpus"]), mem_gb=float(request["mem_gb"]), gpus=float(request["gpus"]))
    log_event(root, campaign, "deferred", job=ctx.job_id, target_req=dict(request))
    return wait_and_launch(ctx, next_meta, target, root, lambda: waiter_spec(ctx.spec, target, next_meta),
                           gpu_query=gpu_query)


def cmd_run(args: argparse.Namespace, ctx: JobContext | None = None,
            gpu_query: Callable[[], list[dict[str, Any]] | None] = gpu_processes) -> int:
    ctx = ctx or JobContext()
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("lowprio run: nothing to run (lowprio run [options] -- COMMAND ...)")
    meta = _meta_for(ctx, args)
    root = state_dir(meta.get("state_dir"))
    campaign = meta["campaign"]
    for key in ("poll_s", "settle_s", "gpu_quiet_s", "max_queue_s", "yield_to", "peer_wait_s"):
        meta.setdefault(key, getattr(args, key))
    priority = lower_priority()
    log_event(root, campaign, "start", job=ctx.job_id, gen=meta.get("gen", 0), priority=priority, under_rx=ctx.under_rx,
              gpus=ctx.gpus, cmd=cmd)
    if stop_requested(root, campaign):
        log_event(root, campaign, "stopped", job=ctx.job_id, why="stop requested before the start")
        return 0
    same, detail = check_code(root, campaign, Path.cwd(), first=int(meta.get("gen", 0)) == 0)
    log_event(root, campaign, "code", job=ctx.job_id, same=same, detail=detail)
    if not same:
        return EXIT_CODE_CHANGED
    if args.defer and int(meta.get("gen", 0)) == 0:
        return defer(ctx, meta, root, args.defer, gpu_query)
    watch = Watch(ctx, args, gpu_query)
    reason = watch.sample()
    if reason:
        log_event(root, campaign, "yield", job=ctx.job_id, phase="before start", reason=reason)
        requeue(ctx, meta, root, reason)
        return EXIT_YIELD
    child = start_child(cmd)
    log_event(root, campaign, "child", job=ctx.job_id, pid=child.pid)
    load = LoadLog(ctx, args.load_log_s) if args.load_log_s > 0 else None

    def log_load(now: float, force: bool = False) -> None:
        fields = load.due(watch.view, now, force=force) if load is not None else None
        if fields:
            log_event(root, campaign, "host", job=ctx.job_id, **fields)

    contended_since: float | None = None
    yielding_since: float | None = None
    quiet_since: float | None = None
    mark = 0.0
    last_reason: dict[str, Any] | None = None
    while True:
        rc = child.poll()
        if rc is not None:
            log_load(time.monotonic(), force=True)
            log_event(root, campaign, "done" if rc == 0 else "child_failed", job=ctx.job_id, rc=rc,
                      during_yield=yielding_since is not None)
            return rc
        now = time.monotonic()  # durations only: a step of the wall clock must not delay or hasten a yield
        reason = watch.sample()
        log_load(now)
        if yielding_since is None:
            if reason:
                contended_since = contended_since or now
                last_reason = reason
                if now - contended_since >= args.confirm_s:
                    yielding_since, mark, quiet_since = now, latest_progress(args.progress), None
                    log_event(root, campaign, "yielding", job=ctx.job_id, reason=reason, grace_s=args.grace_s,
                              progress=args.progress)
            else:
                contended_since = None
        else:
            if reason:
                quiet_since, last_reason = None, reason
            else:
                quiet_since = quiet_since or now
                if now - quiet_since >= args.confirm_s:
                    log_event(root, campaign, "yield_cancelled", job=ctx.job_id, why="contention ended")
                    yielding_since = contended_since = quiet_since = None
                    continue
            progressed = args.progress and latest_progress(args.progress) > mark
            if progressed or now - yielding_since >= args.grace_s:
                if progressed:
                    time.sleep(args.after_progress_s)
                if child.poll() is None:
                    log_load(time.monotonic(), force=True)
                    stopped = stop_tree(child)
                    log_event(root, campaign, "yielded", job=ctx.job_id, stopped=stopped,
                              waited_s=round(time.monotonic() - yielding_since, 1),
                              on="progress" if progressed else "grace", reason=last_reason)
                    requeue(ctx, meta, root, last_reason)
                    return EXIT_YIELD
                continue
        time.sleep(args.poll_s)


# ───────────────────────────── wait (the waiter job) ─────────────────────────────

def cmd_wait(args: argparse.Namespace, ctx: JobContext | None = None,
             gpu_query: Callable[[], list[dict[str, Any]] | None] = gpu_processes) -> int:
    ctx = ctx or JobContext()
    if not ctx.under_rx or "lowprio_target" not in ctx.spec:
        raise SystemExit("lowprio wait runs only as the waiter job that `lowprio run --requeue` launches")
    meta = ctx.meta
    lower_priority()
    return wait_and_launch(ctx, meta, dict(ctx.spec["lowprio_target"]), state_dir(meta.get("state_dir")),
                           lambda: dict(ctx.spec, id=make_id(str(ctx.spec.get("name") or "wait"))),
                           poll_s=args.poll_s, give_up_s=args.give_up_s, gpu_query=gpu_query)


def wait_and_launch(ctx: JobContext, meta: Mapping[str, Any], target: Mapping[str, Any], root: Path,
                    step_aside: Callable[[], dict[str, Any]], *, poll_s: float | None = None,
                    give_up_s: float | None = None,
                    gpu_query: Callable[[], list[dict[str, Any]] | None] = gpu_processes) -> int:
    """Wait until the host has room for `target`, then launch it; step_aside() is the waiter to launch
    instead when even this job's share is in someone's way (a share above rx's fit tolerance)."""
    campaign = meta["campaign"]
    poll_s = float(poll_s if poll_s is not None else meta.get("poll_s", 30.0))
    settle_s = float(meta.get("settle_s", 120.0))
    gpu_quiet_s = float(meta.get("gpu_quiet_s", 600.0))
    max_queue_s = float(meta.get("max_queue_s", 600.0))
    policy = {"yield_to": meta.get("yield_to"), "peer_wait_s": meta.get("peer_wait_s")}
    wargs = argparse.Namespace(gpu_policy=meta.get("gpu_policy", "yield"), gpu_types=meta.get("gpu_types", "C"),
                               ignore_gpu_process=list(meta.get("ignore_gpu_process") or []), **policy)
    watch = Watch(ctx, wargs, gpu_query)
    req = job_request(target)
    target_gpus = [int(g["index"]) for g in (read_json(Path(ctx.home) / "host.json") or {}).get("gpus") or []] \
        if req["gpus"] > 0 else []
    log_event(root, campaign, "waiting", job=ctx.job_id, gen=meta.get("gen"), target_req=req)
    started = time.monotonic()  # durations run on the monotonic clock
    ready_since: float | None = None
    last_yield = meta.get("last_yield") or {}
    gpu_busy_at: float | None = None
    if last_yield.get("kind") == "gpu" and last_yield.get("t"):
        # the yielding process recorded wall time; a step of the clock back only lengthens the quiet period
        gpu_busy_at = started - max(0.0, time.time() - float(last_yield["t"]))
    last_why = None
    while True:
        now = time.monotonic()
        if give_up_s is not None and now - started > give_up_s:
            log_event(root, campaign, "gave_up", job=ctx.job_id, waited_s=round(now - started, 1))
            return EXIT_GAVE_UP
        if stop_requested(root, campaign):
            log_event(root, campaign, "stopped", job=ctx.job_id, why="stop requested while waiting")
            return 0
        view = HostView(ctx.home)
        mine = rx_contention(view.cap, view.active, view.queue, project=ctx.project, key=ctx.key, now=view.now,
                             **policy) if view.usable else None
        if mine:  # even the waiter's share is in someone's way: step aside behind them
            log_event(root, campaign, "waiter_yield", job=ctx.job_id, reason=mine)
            nxt = step_aside()
            try:
                launch(ctx, {k: v for k, v in nxt.items() if k not in LAUNCH_FIELDS})
            except LowprioError as exc:
                log_event(root, campaign, "requeue_failed", job=ctx.job_id, error=str(exc))
                return 1
            return 0
        blocker = launch_blocker(view.cap, view.active, view.queue, req, project=ctx.project, now=view.now,
                                 **policy) if view.usable else None
        gpu_reason = watch.gpu(target_gpus, view) if target_gpus else None
        if gpu_reason:
            gpu_busy_at = now
        why = None
        if not view.usable:
            why = "host.json unreadable"
        elif blocker:
            why = blocker
        elif gpu_reason:
            why = f"foreign GPU processes {gpu_reason['processes']}"
        elif gpu_busy_at is not None and now - gpu_busy_at < gpu_quiet_s:
            why = f"GPU quiet for {now - gpu_busy_at:.0f} of {gpu_quiet_s:.0f} s"
        if why:
            ready_since = None
            if why != last_why:
                print(f"[lowprio {time.strftime('%H:%M:%S')}] waiting: {why}", flush=True)
                last_why = why
            time.sleep(poll_s)
            continue
        ready_since = ready_since or now
        if now - ready_since < settle_s:
            time.sleep(poll_s)
            continue
        same, detail = check_code(root, campaign, Path.cwd(), first=False)
        if not same:
            log_event(root, campaign, "code", job=ctx.job_id, same=False, detail=detail)
            return EXIT_CODE_CHANGED
        spec = launched_spec(target)
        try:
            launch(ctx, spec)
        except LowprioError as exc:
            log_event(root, campaign, "requeue_failed", job=ctx.job_id, error=str(exc))
            return 1
        log_event(root, campaign, "relaunched", job=ctx.job_id, relaunch=spec["id"], gen=meta.get("gen"))
        t0 = time.monotonic()
        while True:
            state = job_state(ctx, ctx.project, spec["id"])
            if state not in (None, "launching", "queued"):
                log_event(root, campaign, "handed_over", job=ctx.job_id, relaunch=spec["id"], state=state)
                return 0
            if time.monotonic() - t0 > max_queue_s:
                try:
                    agent_call(ctx, {"op": "cancel", "project": ctx.project, "id": spec["id"]})
                except (LowprioError, OSError, subprocess.TimeoutExpired) as exc:
                    log_event(root, campaign, "cancel_failed", job=ctx.job_id, relaunch=spec["id"], error=str(exc))
                    return 1
                log_event(root, campaign, "relaunch_cancelled", job=ctx.job_id, relaunch=spec["id"],
                          why=f"still {state} after {max_queue_s:.0f} s")
                ready_since = None
                break
            time.sleep(min(poll_s, 5.0))


# ───────────────────────────── status, stop, probe ─────────────────────────────

def _missing_campaign(root: Path, campaign: str) -> bool:
    if campaign_paths(root, campaign)["log"].exists():
        return False
    print(f"no campaign {campaign} under {root}; give --state-dir (a job's $JIGSAW_LOWPRIO_ROOT)", file=sys.stderr)
    return True


def cmd_status(args: argparse.Namespace) -> int:
    root = campaign_state_dir(args.state_dir)
    if _missing_campaign(root, args.campaign):
        return 2
    paths = campaign_paths(root, args.campaign)
    lines = paths["log"].read_text(encoding="utf-8").splitlines()
    for line in lines[-args.last:]:
        print(line)
    host = []
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("event") == "host":
            host.append(rec)
    print(json.dumps({"campaign": args.campaign, "events": len(lines), "stop_requested": stop_requested(root, args.campaign),
                      "code": (read_json(paths["code"]) or {}).get("tree_sha256"), "host_events": len(host),
                      "host_events_shared": sum(1 for r in host if r.get("others")),
                      "max_cpu_other_pct": max((float(r.get("cpu_other_pct") or 0) for r in host), default=None)},
                     sort_keys=True))
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    root = campaign_state_dir(args.state_dir)
    path = campaign_paths(root, args.campaign)["stop"]
    if args.clear:
        if path.exists():
            path.unlink()
        print(f"cleared {path}")
        return 0
    if not args.force and _missing_campaign(root, args.campaign):  # a STOP file where the campaign never looks
        return 2                                                   # stops nothing
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.utc_now() + "\n", encoding="utf-8")
    print(f"created {path}: the current job finishes or yields and nothing new is launched")
    return 0


def cmd_probe(args: argparse.Namespace, gpu_query: Callable[[], list[dict[str, Any]] | None] = gpu_processes) -> int:
    ctx = JobContext()
    home = ctx.home or args.home
    out: dict[str, Any] = {"job": ctx.job_id, "project": ctx.project, "gpus": ctx.gpus}
    if home:
        view = HostView(home)
        out["capacity"] = view.cap
        out["active"] = [{k: a.get(k) for k in ("project", "id", "req", "assign", "admitted")} for a in view.active]
        out["queue"] = [{k: q.get(k) for k in ("project", "id", "req", "enq")} for q in view.queue]
        out["free"] = free_pool(view.cap, view.active)[:2] if view.usable else None
        if ctx.under_rx and view.usable:
            out["rx_contention"] = rx_contention(view.cap, view.active, view.queue, project=ctx.project,
                                                 key=ctx.key, now=view.now, yield_to=args.yield_to,
                                                 peer_wait_s=args.peer_wait_s)
    procs = gpu_query()
    out["gpu_processes"] = procs
    if procs is not None:
        mine = own_pids()
        if home and view.usable:
            jobs = gpu_process_jobs(procs, view.active)
            out["gpu_process_jobs"] = {str(pid): job.get("key") for pid, job in jobs.items()}
            mine |= shared_gpu_pids(procs, view.cap, view.active, project=ctx.project or args.project,
                                    yield_to=args.yield_to)
        out["foreign_gpu_processes"] = foreign_gpu_processes(
            procs, gpus=sorted({p["gpu"] for p in procs}), mine=mine, types=args.gpu_types.split(","),
            ignore=args.ignore_gpu_process)
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


# ───────────────────────────── command line ─────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def gpu_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--gpu-types", default="C",
                       help="nvidia-smi process types that count as GPU compute (comma list; C+G adds desktop apps)")
        p.add_argument("--ignore-gpu-process", action="append", default=[],
                       help="process name (or full path) never counted as a foreign GPU process (repeatable)")

    def policy_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--yield-to", type=parse_projects, default=None, metavar="PROJECT[,PROJECT...]",
                       help="projects whose queued jobs and GPU processes make this job yield at once, beside any "
                            "project rx's priority map ranks above this one (default: every other project)")
        p.add_argument("--peer-wait-s", type=float, default=None,
                       help="with --yield-to: how long another project's queued job waits before this job yields "
                            "to it (default: rx's hold_after_s)")

    r = sub.add_parser("run", help="run a command at lower priority, yielding under contention")
    r.add_argument("--campaign", default=None, help="name of the chain of hops (default: the rx job name)")
    r.add_argument("--state-dir", default=None, help="campaign logs and STOP files (default $JIGSAW_LOWPRIO_ROOT)")
    r.add_argument("--requeue", action="store_true", help="after a yield, relaunch through a waiter job")
    r.add_argument("--defer", type=parse_request, default=None, metavar="CPUS,MEM_GB[,GPUS]",
                   help="this rx job is waiter-sized (rx run --cpus 1e-11 --mem 1e-11) and launches the job "
                        "with this request once the host has room")
    r.add_argument("--max-requeues", type=int, default=500)
    r.add_argument("--progress", action="append", default=[],
                   help="glob (** allowed) of files whose new mtime marks a safe point to stop (repeatable)")
    r.add_argument("--grace-s", type=float, default=600.0, help="longest wait for progress once yielding")
    r.add_argument("--confirm-s", type=float, default=60.0, help="how long contention must persist")
    r.add_argument("--after-progress-s", type=float, default=15.0, help="pause after progress before stopping")
    r.add_argument("--poll-s", type=float, default=30.0)
    r.add_argument("--load-log-s", type=float, default=300.0,
                   help="log what shares the host (a `host` event) at least this often; 0 turns it off")
    r.add_argument("--settle-s", type=float, default=120.0, help="waiter: how long the host must have room")
    r.add_argument("--gpu-quiet-s", type=float, default=600.0,
                   help="waiter: how long no foreign GPU process may have run before a GPU relaunch")
    r.add_argument("--max-queue-s", type=float, default=600.0,
                   help="waiter: cancel a relaunch still queued after this long (below rx's hold_after_s)")
    r.add_argument("--gpu-policy", choices=["yield", "ignore"], default="yield",
                   help="what a foreign GPU compute process on this job's GPU means")
    gpu_options(r)
    policy_options(r)
    r.add_argument("cmd", nargs=argparse.REMAINDER)

    w = sub.add_parser("wait", help="the waiter job (launched by run --requeue)")
    w.add_argument("--poll-s", type=float, default=None)
    w.add_argument("--give-up-s", type=float, default=None, help="exit 4 after this long (tests)")

    s = sub.add_parser("status", help="print a campaign's log")
    s.add_argument("--campaign", required=True)
    s.add_argument("--state-dir", default=None)
    s.add_argument("--last", type=int, default=30)

    t = sub.add_parser("stop", help="stop a campaign after its current job")
    t.add_argument("--campaign", required=True)
    t.add_argument("--state-dir", default=None)
    t.add_argument("--clear", action="store_true", help="remove the STOP file")
    t.add_argument("--force", action="store_true",
                   help="create the STOP file although the campaign has no log there (one not started yet)")

    p = sub.add_parser("probe", help="print the host as a job sees it")
    p.add_argument("--home", default=None, help="rx home when not run as an rx job")
    p.add_argument("--project", default="jigsaw", help="the project whose GPU sharers count as its own outside rx")
    gpu_options(p)
    policy_options(p)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "wait":
        return cmd_wait(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "stop":
        return cmd_stop(args)
    return cmd_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
