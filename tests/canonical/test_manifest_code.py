"""Code identity without git: the CRLF-folded fingerprint, the git_info fallback and code_check against a commit."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from scripts.canonical import manifest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "core.autocrlf=false",
                    *args], cwd=str(repo), check=True, capture_output=True)


def write_tree(root, newline=b"\n"):
    (root / "scripts" / "pkg" / "__pycache__").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "scripts" / "pkg" / "a.py").write_bytes(b"a = 1" + newline + b"b = 2" + newline)
    (root / "tests" / "t.py").write_bytes(b"def test(): pass" + newline)
    (root / "scripts" / "pkg" / "__pycache__" / "a.cpython-313.pyc").write_bytes(b"\x00junk")
    (root / "notes.txt").write_bytes(b"outside the code paths")


def test_fingerprint_folds_crlf_and_skips_caches(tmp_path):
    lf, crlf = tmp_path / "lf", tmp_path / "crlf"
    write_tree(lf)
    write_tree(crlf, newline=b"\r\n")
    a, b = manifest.code_fingerprint(lf), manifest.code_fingerprint(crlf)
    assert sorted(a["files"]) == ["scripts/pkg/a.py", "tests/t.py"]
    assert a == b and a["crlf_folded"] is True
    (crlf / "tests" / "t.py").write_bytes(b"def test(): assert 0\r\n")
    assert manifest.code_fingerprint(crlf)["tree_sha256"] != a["tree_sha256"]


def test_git_info_falls_back_to_the_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    write_tree(tmp_path / "copy")
    info = manifest.git_info(tmp_path / "copy")
    assert info["available"] is False
    assert info["code_fingerprint"] == manifest.code_fingerprint(tmp_path / "copy")


def test_code_check_against_a_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    repo, copy = tmp_path / "repo", tmp_path / "copy"
    write_tree(repo)
    git(repo, "init", "-q")
    git(repo, "add", "scripts/pkg/a.py", "tests/t.py")
    git(repo, "commit", "-q", "-m", "code")
    write_tree(copy, newline=b"\r\n")  # a Windows checkout of the same commit, copied to the host
    run = tmp_path / "run"
    run.mkdir()
    (run / manifest.MANIFEST_NAME).write_text(json.dumps({"git": manifest.git_info(copy)}), encoding="utf-8")
    assert manifest.code_check(run, "HEAD", repo_root=repo) == []
    (copy / "scripts" / "pkg" / "a.py").write_bytes(b"a = 3\r\n")
    (copy / "scripts" / "pkg" / "new.py").write_bytes(b"x = 1\n")
    (run / manifest.MANIFEST_NAME).write_text(json.dumps({"git": manifest.git_info(copy)}), encoding="utf-8")
    problems = manifest.code_check(run / manifest.MANIFEST_NAME, "HEAD", repo_root=repo)
    assert problems == ["scripts/pkg/a.py: differs from HEAD", "scripts/pkg/new.py: not in HEAD"]
    assert manifest.code_check(run, "no-such-rev", repo_root=repo) == ["unknown revision no-such-rev"]
    (run / manifest.MANIFEST_NAME).write_text(json.dumps({"git": {"available": False}}), encoding="utf-8")
    assert manifest.code_check(run, repo_root=repo) == ["the run recorded neither git state nor a code fingerprint"]
