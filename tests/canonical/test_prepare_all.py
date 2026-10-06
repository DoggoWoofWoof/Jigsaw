"""prepare_all: the dataset list, the prepare_data flags, memprobe files, verification and the first failure."""

from __future__ import annotations

import json

import pytest

from scripts.canonical.launch import prepare_all as pa


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setenv("JIGSAW_STORE_ROOT", str(tmp_path / "stores"))
    monkeypatch.setenv("JIGSAW_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("JIGSAW_RUNS_ROOT", str(tmp_path / "runs"))
    return tmp_path


def fake(monkeypatch, codes, problems=()):
    calls, verified = [], []

    def run_measured(cmd, *, cwd, stdout_path, stderr_path):
        calls.append(cmd)
        with open(stderr_path, "w", encoding="utf-8") as fh:
            fh.write("stage one\nstage two\n")
        return {"exit_code": codes[len(calls) - 1], "timed_out": False, "wall_seconds": 1.5, "peak_rss_bytes": 2**30}

    def verify(run_dir):
        verified.append(run_dir.parent.name)
        return list(problems)

    monkeypatch.setattr(pa.memprobe, "run_measured", run_measured)
    monkeypatch.setattr(pa.manifest, "verify_manifest", verify)
    return calls, verified


def test_default_datasets_and_flags(roots, monkeypatch, capsys):
    calls, verified = fake(monkeypatch, [0, 0, 0, 0])
    assert pa.main(["--alternatives", "--metis-tool", "pymetis", "--with-products"]) == 0
    assert [c[c.index("--dataset") + 1] for c in calls] == ["corafull", "arxiv", "mag", "products"]
    assert calls[0][1:3] == ["-m", "scripts.canonical.prepare_data"]
    assert calls[0][-3:] == ["--alternatives", "--metis-tool", "pymetis"]
    assert calls[0][calls[0].index("--data-root") + 1] == str(roots / "data")
    assert verified == ["corafull", "arxiv", "mag", "products"]
    probe = json.loads((roots / "runs" / "prepare" / "mag.memprobe.json").read_text(encoding="utf-8"))
    assert probe["exit_code"] == 0 and "stores ready" in capsys.readouterr().out


def test_named_datasets_replace_the_list_and_the_first_failure_stops(roots, monkeypatch, capsys):
    calls, verified = fake(monkeypatch, [0, 9])
    assert pa.main(["mag", "arxiv", "corafull"]) == 1
    assert [c[c.index("--dataset") + 1] for c in calls] == ["mag", "arxiv"]
    assert verified == ["mag"]
    assert "stage two" in capsys.readouterr().err


def test_a_store_that_fails_verification_stops(roots, monkeypatch, capsys):
    calls, _ = fake(monkeypatch, [0, 0], problems=["changed output store/meta.json"])
    assert pa.main(["corafull", "arxiv"]) == 1
    assert len(calls) == 1 and "changed output" in capsys.readouterr().err


def test_store_root_is_required(monkeypatch):
    monkeypatch.delenv("JIGSAW_STORE_ROOT", raising=False)
    with pytest.raises(SystemExit):
        pa.main([])
