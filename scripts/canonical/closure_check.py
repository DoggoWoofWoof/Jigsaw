"""Compare the code a run could execute with a commit: the import closure of its entry module.

    python -m scripts.canonical.closure_check RUN_DIR [RUN_DIR ...] [--rev REV] [--entry PATH]
        [--repo-root CHECKOUT] [--json]

A run made without git records the SHA-256 of every code file (manifest.code_fingerprint), and
`manifest code-check` requires all of them to equal a commit. This check requires only the files
the run's entry module can reach: the repository modules its import statements name at any depth
(imports inside functions too), the packages above them, and repository modules named by a string
constant (the `python -m scripts.canonical.X` subprocesses). The closure is read from REV. When every
file of it equals the run's fingerprint, the run's own closure is the same set of files, so the run
executed REV's code; files outside it (tests, launchers, reducers) may differ and are listed.

The entry is argv[0] of the run's manifest (a script path, or "-m MODULE"), or --entry. A run that
recorded git state is checked by manifest.code_check.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from . import manifest

MODULE_STRING = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")


def _git(args: list[str], repo_root: Path, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(repo_root), input=stdin, capture_output=True, timeout=300,
                          check=False)


class RevTree:
    """The files of one commit and their contents, read with one git cat-file process per batch."""

    def __init__(self, repo_root: Path, sha: str) -> None:
        self.repo_root, self.sha = repo_root, sha
        listing = _git(["ls-tree", "-r", "-z", "--name-only", sha], repo_root)
        self.files = {f for f in listing.stdout.decode("utf-8", "replace").split("\0") if f}
        self.blobs: dict[str, bytes] = {}

    def fetch(self, rels: Iterable[str]) -> None:
        wanted = sorted({r for r in rels if r in self.files and r not in self.blobs})
        if not wanted:
            return
        out = _git(["cat-file", "--batch"], self.repo_root,
                   stdin="".join(f"{self.sha}:{r}\n" for r in wanted).encode("utf-8")).stdout
        pos = 0
        for rel in wanted:  # "<sha> blob <size>\n<content>\n", or "<name> missing\n"
            end = out.index(b"\n", pos)
            header = out[pos:end].split()
            if len(header) != 3:
                pos = end + 1
                continue
            size = int(header[2])
            self.blobs[rel] = out[end + 1:end + 1 + size]
            pos = end + 1 + size + 1

    def blob(self, rel: str) -> bytes | None:
        self.fetch([rel])
        return self.blobs.get(rel)

    def digest(self, rel: str) -> str | None:
        data = self.blob(rel)
        return None if data is None else hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


_TREES: dict[tuple[str, str], RevTree] = {}


def rev_tree(repo_root: Path, rev: str) -> RevTree | None:
    resolved = _git(["rev-parse", "--verify", f"{rev}^{{commit}}"], repo_root)
    if resolved.returncode != 0:
        return None
    sha = resolved.stdout.decode().strip()
    key = (str(repo_root), sha)
    if key not in _TREES:
        _TREES[key] = RevTree(repo_root, sha)
    return _TREES[key]


def module_file(module: str, exists: Callable[[str], bool]) -> str | None:
    """The repository file of a dotted module name: the module file, or the package's __init__.py."""
    base = module.replace(".", "/")
    for rel in (f"{base}.py", f"{base}/__init__.py"):
        if exists(rel):
            return rel
    return None


def packages_above(rel: str, exists: Callable[[str], bool]) -> list[str]:
    parts = rel.split("/")[:-1]
    out = []
    for k in range(1, len(parts) + 1):
        init = "/".join(parts[:k]) + "/__init__.py"
        if init != rel and exists(init):
            out.append(init)
    return out


def referenced_modules(rel: str, source: str) -> set[str]:
    """Dotted names the file imports (resolved relative imports included) or names as module strings."""
    tree = ast.parse(source, filename=rel)
    parts = rel[:-3].split("/")
    package = parts[:-1]  # a package's __init__.py belongs to that package; a module to its directory
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                up = package[:len(package) - (node.level - 1)] if node.level > 1 else package
                base = ".".join(up + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            if base:
                out.add(base)
                out.update(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and MODULE_STRING.match(node.value):
            out.add(node.value)
    return out


def closure(entry: str, read: Callable[[str], str | None], exists: Callable[[str], bool]) -> list[str]:
    """Every repository file reachable from entry, entry included, sorted."""
    seen: set[str] = set()
    todo = [entry, *packages_above(entry, exists)]
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        source = read(rel)
        if source is None or not rel.endswith(".py"):
            continue
        for module in referenced_modules(rel, source):
            found = module_file(module, exists)
            if found is not None:
                todo.append(found)
                todo.extend(packages_above(found, exists))
    return sorted(seen)


def entry_of(argv: list[str], exists: Callable[[str], bool]) -> str | None:
    if not argv:
        return None
    if argv[0] == "-m" and len(argv) > 1:
        return module_file(argv[1], exists)
    path = argv[0].replace("\\", "/")
    hits = [rel for rel in _suffixes(path) if exists(rel)]
    return max(hits, key=len) if hits else None


def _suffixes(path: str) -> Iterable[str]:
    parts = [p for p in path.split("/") if p]
    for k in range(len(parts)):
        yield "/".join(parts[k:])


def check(run: str | Path, rev: str = "HEAD", *, entry: str | None = None,
          repo_root: Path = manifest.REPO_ROOT) -> dict[str, Any]:
    """{"entry", "rev", "closure", "problems", "outside_closure"} for one run directory or manifest."""
    run = Path(run)
    doc = json.loads((run / manifest.MANIFEST_NAME if run.is_dir() else run).read_text(encoding="utf-8"))
    git = doc.get("git") or {}
    if git.get("available"):
        return {"entry": None, "rev": git.get("head"), "closure": [], "outside_closure": [],
                "problems": manifest.code_check(run, rev, repo_root=repo_root)}
    fp = (git.get("code_fingerprint") or {}).get("files")
    if not fp:
        return {"entry": None, "rev": None, "closure": [], "outside_closure": [],
                "problems": ["the run recorded neither git state nor a code fingerprint"]}
    tree = rev_tree(repo_root, rev)
    if tree is None:
        return {"entry": None, "rev": None, "closure": [], "outside_closure": [], "problems": [f"unknown revision {rev}"]}

    def exists(rel: str) -> bool:
        return rel in tree.files or rel in fp

    def read(rel: str) -> str | None:
        data = tree.blob(rel)
        return None if data is None else data.decode("utf-8", "replace")

    entry = entry or entry_of(list(doc.get("argv") or []), exists)
    if entry is None:
        return {"entry": None, "rev": tree.sha, "closure": [], "outside_closure": [],
                "problems": ["no entry module in the run's argv; give --entry"]}
    files = closure(entry, read, exists)
    problems = []
    for rel in files:
        digest = tree.digest(rel)
        if rel not in fp:
            problems.append(f"{rel}: not in the run's code")
        elif digest is None:
            problems.append(f"{rel}: not in {rev}")
        elif digest != fp[rel]:
            problems.append(f"{rel}: differs from {rev}")
    inside = set(files)
    tree.fetch(fp)
    outside = [rel for rel, digest in sorted(fp.items()) if rel not in inside and tree.digest(rel) != digest]
    return {"entry": entry, "rev": tree.sha, "closure": files, "problems": problems, "outside_closure": outside}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="run directories or manifest.json files")
    parser.add_argument("--rev", default="HEAD")
    parser.add_argument("--entry", default=None, help="repository path of the entry script (default: from argv)")
    parser.add_argument("--repo-root", default=str(manifest.REPO_ROOT), help="the git checkout holding REV")
    parser.add_argument("--json", action="store_true", help="print the full results as JSON")
    args = parser.parse_args(argv)
    results = {str(run): check(run, args.rev, entry=args.entry, repo_root=Path(args.repo_root)) for run in args.runs}
    if args.json:
        print(json.dumps(results, indent=1, sort_keys=True))
    for run, res in results.items():
        state = "ok" if not res["problems"] else f"{len(res['problems'])} problems"
        print(f"{run}: {state} (entry {res['entry']}, {len(res['closure'])} files; "
              f"{len(res['outside_closure'])} outside the closure differ)", file=sys.stderr)
        for problem in res["problems"]:
            print(f"  {problem}", file=sys.stderr)
    return 1 if any(res["problems"] for res in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
