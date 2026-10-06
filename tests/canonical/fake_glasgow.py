"""Stand-in for the Glasgow binary in backend tests. FAKE_GLASGOW_MODE picks the reply."""

import os
import sys
import time

mode = os.environ.get("FAKE_GLASGOW_MODE", "found")
args = sys.argv[1:]
pattern, target = args[-2], args[-1]
with open(pattern, encoding="ascii") as handle:
    n = int(handle.readline())
if mode == "found":
    print("status = true")
    print("runtime = 1")
    print("mapping = " + " ".join(f"({i} -> {n - 1 - i})" for i in range(n)))
elif mode == "all":
    print("status = true")
    print("mapping = " + " ".join(f"({i} -> {i})" for i in range(n)))
    print("mapping = " + " ".join(f"({i} -> {n - 1 - i})" for i in range(n)))
    print("solution_count = 2")
elif mode == "none":
    print("status = false")
elif mode == "aborted":
    print("status = aborted")
elif mode == "sleep":
    time.sleep(30)
    print("status = true")
elif mode == "echo_args":
    print("args = " + " ".join(args[:-2]))
else:
    print("segmentation fault", file=sys.stderr)
    sys.exit(139)
