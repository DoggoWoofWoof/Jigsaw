#!/usr/bin/env bash
# Shared steps for external/build_<tool>.sh. Source this file; do not run it.
#
#   pin_field TOOL FIELD       one field of external/PINS.json (lists print one item per line)
#   fetch_pinned TOOL          clone into external/src/TOOL, check out the pinned commit, verify HEAD
#   apply_patches TOOL         git apply --check, then git apply, every pinned patch in order
#   install_file SRC DEST      copy a built file into external/bin and print its sha256
#   write_build_json TOOL BIN [FILE...] [key=value...]
#                              write external/bin/TOOL.build.json (commit, patch sha256s,
#                              compiler, -march=native target, binary sha256, extra files, checks)
#
# Environment: CXX (default g++), JOBS (default nproc), PYTHON (default python3).
set -euo pipefail

EXTERNAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXTERNAL_DIR}/.." && pwd)"
PINS="${EXTERNAL_DIR}/PINS.json"
SRC_ROOT="${EXTERNAL_DIR}/src"
BIN_ROOT="${EXTERNAL_DIR}/bin"
PYTHON="${PYTHON:-python3}"
CXX="${CXX:-g++}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

die() {
  echo "error: $*" >&2
  exit 1
}

require_cmd() {
  local cmd
  for cmd in "$@"; do
    command -v "$cmd" >/dev/null 2>&1 || die "'${cmd}' is not on PATH"
  done
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

pin_field() {
  "${PYTHON}" - "$PINS" "$1" "$2" <<'PY'
import json
import sys

pins = json.load(open(sys.argv[1], encoding="utf-8"))
if sys.argv[2] not in pins:
    sys.exit(f"no entry for {sys.argv[2]!r} in {sys.argv[1]}")
value = pins[sys.argv[2]].get(sys.argv[3])
if value is None:
    print("null")
elif isinstance(value, list):
    print("\n".join(str(v) for v in value))
else:
    print(value)
PY
}

require_pin() {
  local commit
  commit="$(pin_field "$1" commit)"
  if [[ -z "$commit" || "$commit" == "null" ]]; then
    die "external/PINS.json has no commit for $1; refusing to build an unpinned tool"
  fi
  [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || die "pinned commit for $1 is not a full 40-hex SHA: ${commit}"
  printf '%s\n' "$commit"
}

fetch_pinned() {
  local tool="$1" commit repo dest head
  commit="$(require_pin "$tool")"
  repo="$(pin_field "$tool" repo)"
  dest="${SRC_ROOT}/${tool}"
  if [[ ! -d "${dest}/.git" ]]; then
    mkdir -p "$SRC_ROOT"
    git clone --quiet "$repo" "$dest"
  fi
  if ! git -C "$dest" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "$dest" fetch --quiet origin "$commit"
  fi
  # A previous build leaves patched sources and build outputs; start from the pristine commit.
  git -C "$dest" -c advice.detachedHead=false checkout --quiet --force "$commit"
  git -C "$dest" clean -fdxq
  head="$(git -C "$dest" rev-parse HEAD)"
  [[ "$head" == "$commit" ]] || die "${tool} HEAD is ${head}, the pin is ${commit}"
  echo "${tool}: ${repo} at ${head}"
}

apply_patches() {
  local tool="$1" dest="${SRC_ROOT}/$1" patch list
  # Read the list first: a failure inside a process substitution would go unnoticed.
  list="$(pin_field "$tool" patches)"
  while IFS= read -r patch; do
    [[ -z "$patch" ]] && continue
    [[ -f "${REPO_ROOT}/${patch}" ]] || die "missing patch ${patch}"
    git -C "$dest" apply --check "${REPO_ROOT}/${patch}" || die "${patch} does not apply to the pinned ${tool}"
    git -C "$dest" apply "${REPO_ROOT}/${patch}"
    echo "applied ${patch} sha256=$(sha256_of "${REPO_ROOT}/${patch}")"
  done <<< "$list"
}

install_file() {
  mkdir -p "$(dirname "$2")"
  cp -f "$1" "$2"
  echo "$(sha256_of "$2")  ${2#"${REPO_ROOT}/"}"
}

write_build_json() {
  local tool="$1" binary="$2"
  shift 2
  "${PYTHON}" - "$PINS" "$tool" "$binary" "$REPO_ROOT" "$CXX" "${SRC_ROOT}/${tool}" "$@" <<'PY'
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

pins_path, tool, binary, root, cxx, src, *rest = sys.argv[1:]
pin = json.load(open(pins_path, encoding="utf-8"))[tool]


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def rel(path):
    return os.path.relpath(os.path.abspath(path), root).replace(os.sep, "/")


march = None
for line in run([cxx, "-march=native", "-Q", "--help=target"]).splitlines():
    parts = line.split()
    if len(parts) == 2 and parts[0] == "-march=":
        march = parts[1]
        break
extra_files = [p for p in rest if "=" not in p]
checks = dict(p.split("=", 1) for p in rest if "=" in p)
record = {
    "tool": tool,
    "repo": pin["repo"],
    "commit": pin["commit"],
    "source_head": run(["git", "-C", src, "rev-parse", "HEAD"]),
    "patches": [{"path": p, "sha256": sha(os.path.join(root, p))} for p in pin["patches"]],
    "compiler": (run([cxx, "--version"]).splitlines() or [""])[0],
    "compiler_path": run(["bash", "-c", f"command -v {cxx}"]),
    "march_native": march,
    "cmake": (run(["cmake", "--version"]).splitlines() or [""])[0],
    "binary": rel(binary),
    "binary_sha256": sha(binary),
    "extra_files": [{"path": rel(p), "sha256": sha(p)} for p in extra_files],
    "checks": checks,
    "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "platform": platform.platform(),
    "machine": platform.machine(),
}
out = os.path.join(root, "external", "bin", f"{tool}.build.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2)
    handle.write("\n")
print(f"wrote {rel(out)}")
PY
}
