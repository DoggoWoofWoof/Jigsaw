"""Stand-in for gpmetis and kaffpa in partitioner tests: contiguous chunks of the vertex order."""

import json
import os
import sys

args = sys.argv[1:]
record = os.environ.get("FAKE_PARTITIONER_RECORD")
if record:
    with open(record, "w", encoding="utf-8") as handle:
        json.dump(args, handle)
if os.environ.get("FAKE_PARTITIONER_MODE") == "crash":
    print("segmentation fault", file=sys.stderr)
    sys.exit(139)
kahip = any(a.startswith("--k=") for a in args)
if kahip:
    graph = next(a for a in args if not a.startswith("--"))
    k = int(next(a for a in args if a.startswith("--k=")).split("=", 1)[1])
    out = next(a for a in args if a.startswith("--output_filename=")).split("=", 1)[1]
else:
    positional = [a for a in args if not a.startswith("-")]
    graph, k = positional[0], int(positional[1])
    out = f"{graph}.part.{k}"
with open(graph, encoding="ascii") as handle:
    n = int(handle.readline().split()[0])
mode = os.environ.get("FAKE_PARTITIONER_MODE", "")
if mode == "one-sided":  # every node in part 0
    labels = [0] * n
elif mode == "empty-last" and k > 2:  # part k - 1 left empty
    labels = [min(i * k // n, k - 2) for i in range(n)]
else:
    labels = [i * k // n for i in range(n)]
with open(out, "w", encoding="ascii") as handle:
    handle.write("".join(f"{label}\n" for label in labels))
if not kahip:
    print("METIS 5.1.0 Copyright 1998-13, Regents of the University of Minnesota")
    print(" size of idx_t: 64bits, real_t: 32bits, idx_t *: 64bits")
    print(" - Edgecut: 17, communication volume: 20.")
