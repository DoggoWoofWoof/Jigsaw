"""build_queries: README section 6 order, exclusions, and resuming past complete masters."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.canonical import manifest, protocol
from scripts.canonical.launch import build_queries as bq


def test_plan_order_and_exclusions(tmp_path):
    steps = bq.plan(tmp_path, ["corafull", "arxiv"])
    masters = [*protocol.SEEDS["dev"], *protocol.SEEDS["validation"]]
    assert len(steps) == 2 * len(masters)
    first = [s for s in steps if s["dataset"] == "corafull"]
    assert [s["master"] for s in first] == [str(m) for m in masters]
    for i, step in enumerate(first):
        assert step["exclude"] == [s["out"] for s in first[:i]]  # every directory built before it
        assert Path(step["out"]) == tmp_path / "corafull" / step["split"] / step["master"]
    assert [s for s in steps if s["dataset"] == "arxiv"][0]["exclude"] == []  # datasets are independent
    cmd = bq.command(first[3], "S", python="py")
    assert cmd[:4] == ["py", "-m", "scripts.canonical.queries", "build"]
    assert cmd[cmd.index("--split") + 1] == "validation" and cmd[cmd.index("--master-index") + 1] == "0"
    assert cmd[cmd.index("--exclude") + 1:] == first[3]["exclude"]


def test_driver_keeps_complete_masters_and_stops_on_failure(tmp_path, monkeypatch):
    steps = bq.plan(tmp_path, ["corafull"])
    done = Path(steps[0]["out"])
    done.mkdir(parents=True)
    (done / manifest.MANIFEST_NAME).write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    half = Path(steps[1]["out"])
    half.mkdir(parents=True)
    (half / manifest.MANIFEST_NAME).write_text(json.dumps({"status": "interrupted"}), encoding="utf-8")
    calls = []

    def fake_run(cmd, check=False):
        calls.append(cmd)
        return SimpleNamespace(returncode=0 if len(calls) < 3 else 7)

    monkeypatch.setattr(bq.subprocess, "run", fake_run)
    assert bq.main(["--store-root", "S", "--queries-root", str(tmp_path), "--datasets", "corafull"]) == 1
    outs = [c[c.index("--out") + 1] for c in calls]
    assert outs == [steps[1]["out"], steps[2]["out"], steps[3]["out"]]  # the complete master is kept
