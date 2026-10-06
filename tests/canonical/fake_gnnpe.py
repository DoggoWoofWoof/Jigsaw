"""Stand-in for the patched GNN-PE binary (external/bin/gnnpe/main) in tests. FAKE_GNNPE_MODE picks the reply.

-m offline  enumerates 3-vertex paths like dfs (include/custom.h lines 66-92): from each vertex in
            membership.txt order, each path once, in the orientation first reached, in the
            partition of its first vertex; writes partition_paths.txt into every partition
            directory that exists (an ofstream into a missing one writes nothing) and all_paths.txt
-m online   prints the data graph meta (src/main.cpp line 73), writes index.dat into partitions
            that lack one (include/custom.h lines 218-257), prints the plan size (line 630), then
            enumerates embeddings (non-induced, injective, label equality, degree filter) up to -n,
            prints them with --print-embeddings, and prints the Answer line (src/main.cpp line 179).
            It does not model the R-tree or its orientation-dependent candidate filter.

Online modes: found (default), none, unpatched (counts one answer without printing it), invalid
(prints a non-injective embedding), crash (partial line, exit 139), error0 (a failed block read:
message, exit 0, no Answer line), sleep (30 s), echo_args (prints the arguments and
OMP_NUM_THREADS, exit 1). offline_error0 makes the offline stage print an error and exit 0
without writing a file. Pure Python, so each run starts fast.
"""

from __future__ import annotations

import os
import sys
import time

from fake_subgraphmatching import meta, read_text_graph

OPTIONS = {"-f": "../Test/", "-d": "../Test/data_graph.graph", "-q": "../Test/query_graph.graph", "-m": "offline",
           "-p": "5", "-l": "2", "-e": "2", "-n": "MAX"}  # src/main.cpp lines 40-54


def parse(args: list[str]) -> tuple[dict[str, str], bool]:
    opts = dict(OPTIONS)
    printing = False
    i = 0
    while i < len(args):
        if args[i] == "--print-embeddings":
            printing = True
            i += 1
        elif args[i] in opts and i + 1 < len(args):
            opts[args[i]] = args[i + 1]
            i += 2
        else:
            raise SystemExit(f"The following argument was not expected: {args[i]}")
    return opts, printing


def say(*lines: str) -> None:
    for line in lines:
        print(line)
    sys.stdout.flush()


def adjacency(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for row in adj:
        row.sort()  # libsrc/graph/graph.cpp line 232
    return adj


def paths_from(order: list[int], adj: list[list[int]]) -> list[tuple[int, int, int]]:
    """3-vertex paths from each start in order, each kept once in the orientation reached first."""
    stored: set[tuple[int, int, int]] = set()
    out = []
    for v in order:
        for a in adj[v]:
            for b in adj[a]:
                if b == v or (b, a, v) in stored:
                    continue
                stored.add((v, a, b))
                out.append((v, a, b))
    return out


def offline(opts: dict[str, str], mode: str) -> int:
    root = opts["-f"]
    labels, edges = read_text_graph(opts["-d"])
    say(*meta(labels, edges))
    if mode == "offline_error0":
        sys.stderr.write("RTNode::enter: called, but node is full")
        return 0
    with open(root + "gnn-pe/membership.txt", encoding="ascii") as handle:
        tokens = handle.read().split()
    order = [int(t) for t in tokens[0::2]]
    part = dict(zip(order, (int(t) for t in tokens[1::2])))
    paths = paths_from(order, adjacency(len(labels), edges))
    members: list[list[int]] = [[] for _ in range(int(opts["-p"]))]
    for pid, path in enumerate(paths):
        members[part[path[0]]].append(pid)
    for i, ids in enumerate(members):
        directory = f"{root}gnn-pe/partitions/partition-{i}/"
        if os.path.isdir(directory):
            with open(directory + "partition_paths.txt", "w", encoding="ascii", newline="\n") as handle:
                handle.write(f"{len(ids)}\n" + "".join(f"{p}\n" for p in ids))
    with open(root + "/gnn-pe/all_paths.txt", "w", encoding="ascii", newline="\n") as handle:
        handle.write(f"{len(paths)}\n" + "".join(f"{a} {b} {c} \n" for a, b, c in paths))
    return 0


def plan_size(adj: list[list[int]]) -> int:
    """gen_query_pde (include/custom.h lines 601-628): heaviest paths first while they cover a new vertex."""
    degree = [len(row) for row in adj]
    paths = sorted(paths_from(list(range(len(adj))), adj), key=lambda p: -sum(degree[x] for x in p))
    covered: set[int] = set()
    size = 0
    for path in paths:
        if not set(path) <= covered:
            covered.update(path)
            size += 1
        if len(covered) == len(adj):
            break
    return size


def embeddings(q_labels, q_adj, d_labels, d_adj):
    """Non-induced, injective, label-equal embeddings in a GraphQL-like order (start: fewest candidates)."""
    n = len(q_labels)
    q_deg = [len(row) for row in q_adj]
    d_sets = [set(row) for row in d_adj]
    fits = [[v for v in range(len(d_labels)) if d_labels[v] == q_labels[u] and len(d_adj[v]) >= q_deg[u]]
            for u in range(n)]
    order = [min(range(n), key=lambda u: (len(fits[u]), -q_deg[u], u))]
    while len(order) < n:
        frontier = [u for u in range(n) if u not in order and any(w in order for w in q_adj[u])]
        order.append(min(frontier, key=lambda u: (len(fits[u]), -q_deg[u], u)))
    earlier = [[w for w in order[:i] if w in q_adj[u]] for i, u in enumerate(order)]
    phi = [-1] * n
    used: set[int] = set()

    def extend(i):
        if i == n:
            yield list(phi)
            return
        u = order[i]
        pool = fits[u] if i == 0 else d_adj[phi[earlier[i][0]]]
        for v in pool:
            if (v in used or d_labels[v] != q_labels[u] or len(d_adj[v]) < q_deg[u]
                    or any(phi[w] not in d_sets[v] for w in earlier[i])):
                continue
            phi[u] = v
            used.add(v)
            yield from extend(i + 1)
            used.discard(v)
            phi[u] = -1

    yield from extend(0)


def online(opts: dict[str, str], printing: bool, mode: str) -> int:
    root = opts["-f"]
    labels, edges = read_text_graph(opts["-d"])
    say(*meta(labels, edges))
    if mode == "sleep":
        time.sleep(30)
        return 0
    for i in range(int(opts["-p"])):
        directory = f"{root}gnn-pe/partitions/partition-{i}/"
        if not os.path.isdir(directory):
            sys.stderr.write("BlockFile::new_file: Schreibfehler")
            return 0
        if not os.path.exists(directory + "index.dat"):
            with open(directory + "index.dat", "wb") as handle:
                handle.write(b"fake R-tree\n")
    if not os.path.isfile(opts["-q"]):
        say(f"Can not open the graph file {opts['-q']} .")
        return 255
    q_labels, q_edges = read_text_graph(opts["-q"])
    q_adj = adjacency(len(q_labels), q_edges)
    say(str(plan_size(q_adj)))
    if mode == "crash":
        sys.stdout.write("embedding: 0:")
        sys.stdout.flush()
        return 139
    if mode == "error0":
        sys.stdout.write("Requested block 7 is illegal.")  # blk_file.cpp line 115, then error("\n", true)
        sys.stdout.flush()
        sys.stderr.write("\n")
        return 0
    t0 = time.perf_counter()
    limit = 2**32 - 1 if opts["-n"] == "MAX" else int(opts["-n"])
    count = 0
    if mode == "unpatched":
        count = 1
    elif mode == "invalid":
        count = 1
        if printing:
            say("embedding:" + "".join(f" {u}:0" for u in range(len(q_labels))))
    elif mode != "none":
        for phi in embeddings(q_labels, q_adj, labels, adjacency(len(labels), edges)):
            count += 1
            if printing:
                say("embedding:" + "".join(f" {u}:{v}" for u, v in enumerate(phi)))
            if count >= limit:
                break
    say(f"Answer Number: {count} Query Time (ms): {(time.perf_counter() - t0) * 1000:.6g}")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    mode = os.environ.get("FAKE_GNNPE_MODE", "found")
    args = sys.argv[1:]
    opts, printing = parse(args)
    if opts["-m"] == "offline":
        return offline(opts, mode)
    if mode == "echo_args":
        say("args = " + " ".join(args), f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '')}")
        return 1
    return online(opts, printing, mode)


if __name__ == "__main__":
    sys.exit(main())
