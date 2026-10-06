"""closure_check: a run is compared with a commit on the files its entry module can reach."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from scripts.canonical import closure_check as cc, manifest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

FILES = {
    "scripts/__init__.py": "",
    "scripts/canonical/__init__.py": "",
    "scripts/canonical/entry.py": (
        "from . import direct\n"
        "import numpy as np\n"
        "\n"
        "\n"
        "def later():\n"
        "    from .lazy import value\n"
        "    return value\n"
        "\n"
        "\n"
        "CHILD = ['python', '-m', 'scripts.canonical.child']\n"
    ),
    "scripts/canonical/direct.py": "from .deep.inner import X\n",
    "scripts/canonical/deep/__init__.py": "",
    "scripts/canonical/deep/inner.py": "X = 1\n",
    "scripts/canonical/lazy.py": "value = 2\n",
    "scripts/canonical/child.py": "from .. import canonical\n",
    "scripts/canonical/unused.py": "Y = 3\n",
    "tests/test_entry.py": "def test():\n    pass\n",
}
CLOSURE = ["scripts/__init__.py", "scripts/canonical/__init__.py", "scripts/canonical/child.py",
           "scripts/canonical/deep/__init__.py", "scripts/canonical/deep/inner.py", "scripts/canonical/direct.py",
           "scripts/canonical/entry.py", "scripts/canonical/lazy.py"]
ENTRY_ARGV = ["C:\\Users\\host\\rx\\projects\\jigsaw\\ws\\scripts\\canonical\\entry.py", "--flag"]


def git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                           "-c", "commit.gpgsign=false", *args],
                          cwd=repo, capture_output=True, check=True, text=True).stdout.strip()


def commit(repo, files, message, remove=()):
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    for rel in remove:
        (repo / rel).unlink()
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    commit(root, FILES, "base")
    return root


def run_dir(tmp_path, repo, argv, name="run"):
    """A run made without git on a copy of the repository's current files."""
    out = tmp_path / name
    out.mkdir()
    doc = {"git": {"available": False, "code_fingerprint": manifest.code_fingerprint(repo, ("scripts", "tests"))},
           "argv": argv}
    (out / manifest.MANIFEST_NAME).write_text(json.dumps(doc), encoding="utf-8")
    return out


def test_closure_follows_imports_lazy_imports_and_module_strings(repo, tmp_path):
    res = cc.check(run_dir(tmp_path, repo, ENTRY_ARGV), "HEAD", repo_root=repo)
    assert res["entry"] == "scripts/canonical/entry.py" and res["closure"] == CLOSURE
    assert res["problems"] == [] and res["outside_closure"] == []


def test_changes_outside_the_closure_pass_and_are_listed(repo, tmp_path):
    run = run_dir(tmp_path, repo, ENTRY_ARGV)
    commit(repo, {"scripts/canonical/unused.py": "Y = 4\n", "tests/test_entry.py": "def test():\n    assert 1\n"},
           "outside")
    res = cc.check(run, "HEAD", repo_root=repo)
    assert res["problems"] == [] and res["outside_closure"] == ["scripts/canonical/unused.py", "tests/test_entry.py"]


def test_a_change_inside_the_closure_fails(repo, tmp_path):
    run = run_dir(tmp_path, repo, ENTRY_ARGV)
    commit(repo, {"scripts/canonical/lazy.py": "value = 5\n"}, "reached only through an import inside a function")
    assert cc.check(run, "HEAD", repo_root=repo)["problems"] == ["scripts/canonical/lazy.py: differs from HEAD"]
    assert cc.check(run, "HEAD~1", repo_root=repo)["problems"] == []


def test_files_missing_on_either_side(repo, tmp_path):
    run = run_dir(tmp_path, repo, ENTRY_ARGV)
    commit(repo, {}, "the subprocess module is gone", remove=["scripts/canonical/child.py"])
    assert cc.check(run, "HEAD", repo_root=repo)["problems"] == ["scripts/canonical/child.py: not in HEAD"]
    commit(repo, {"scripts/canonical/entry.py": FILES["scripts/canonical/entry.py"] + "from . import extra\n",
                  "scripts/canonical/extra.py": "Z = 1\n", "scripts/canonical/child.py": FILES["scripts/canonical/child.py"]},
           "a new import")
    assert cc.check(run, "HEAD", repo_root=repo)["problems"] == [
        "scripts/canonical/entry.py: differs from HEAD", "scripts/canonical/extra.py: not in the run's code"]


def test_entry_from_dash_m_or_given(repo, tmp_path):
    res = cc.check(run_dir(tmp_path, repo, ["-m", "scripts.canonical.direct"]), "HEAD", repo_root=repo)
    assert res["entry"] == "scripts/canonical/direct.py" and res["problems"] == []
    assert res["closure"] == ["scripts/__init__.py", "scripts/canonical/__init__.py", "scripts/canonical/deep/__init__.py",
                              "scripts/canonical/deep/inner.py", "scripts/canonical/direct.py"]
    inline = run_dir(tmp_path, repo, ["-c", "print(1)"], name="inline")
    assert cc.check(inline, "HEAD", repo_root=repo)["problems"] == ["no entry module in the run's argv; give --entry"]
    res = cc.check(inline, "HEAD", entry="scripts/canonical/lazy.py", repo_root=repo)
    assert res["problems"] == [] and "scripts/canonical/lazy.py" in res["closure"]
    assert cc.check(inline, "no-such-rev", repo_root=repo)["problems"] == ["unknown revision no-such-rev"]


def test_git_runs_are_checked_by_manifest(repo, tmp_path):
    out = tmp_path / "gitrun"
    out.mkdir()
    for dirty, expected in ((False, []), (True, ["run recorded dirty code at abc"])):
        doc = {"git": {"available": True, "head": "abc", "dirty_code": dirty}, "argv": ENTRY_ARGV}
        (out / manifest.MANIFEST_NAME).write_text(json.dumps(doc), encoding="utf-8")
        assert cc.check(out, "HEAD", repo_root=repo)["problems"] == expected


def test_cli(repo, tmp_path, capsys):
    run = run_dir(tmp_path, repo, ENTRY_ARGV)
    assert cc.main([str(run), "--repo-root", str(repo)]) == 0
    commit(repo, {"scripts/canonical/deep/inner.py": "X = 2\n"}, "inside")
    assert cc.main([str(run), "--repo-root", str(repo), "--json"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)[str(run)]["problems"] == ["scripts/canonical/deep/inner.py: differs from HEAD"]
    assert "1 problems" in captured.err


def test_the_trainer_closure():
    root = manifest.REPO_ROOT

    def exists(rel):
        return (root / rel).is_file()

    def read(rel):
        return (root / rel).read_text(encoding="utf-8") if exists(rel) else None

    files = set(cc.closure("scripts/canonical/train_encoder.py", read, exists))
    assert {"scripts/canonical/queries.py", "scripts/canonical/protocol.py", "scripts/canonical/manifest.py",
            "scripts/canonical/encoders.py", "scripts/canonical/store.py"} <= files
    assert not any(f.startswith(("tests/", "scripts/canonical/launch/")) for f in files)
    assert "scripts/canonical/closure_check.py" not in files and "scripts/canonical/reducer.py" not in files
