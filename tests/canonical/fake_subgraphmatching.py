"""Stand-in for the patched SubgraphMatching binary in tests. FAKE_SM_MODE picks the reply.

found (default)  networkx VF2 subgraph monomorphism with label equality; prints up to -num
                 embeddings in the 0001-print-embeddings.patch format and the full summary
none             a finished search that found nothing (#Embeddings: 0)
sleep            the command-line echo, then 30 s of sleep
crash            partial output, then exit 139 without the summary
limit            the spectrum-mode timeout line with a summary and no embeddings
unpatched        a summary that counts one embedding without printing it (no 0001 patch)
invalid          prints a non-injective embedding
echo_args        prints the arguments and exits 1
Output strings follow matching/StudyPerformance.cpp at the pinned commit (lines 94-105,
110-148, 197-226, 284-291, 356-357, 399-414), matching/BuildTable.cpp printTableCardinality
and graph/graph.cpp line 169. Only the found mode imports networkx, and only -csr input
imports numpy, so the other modes start fast.
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter

SEP = "-" * 68


def option(args: list[str], key: str, default: str = "") -> str:
    """The token after the first occurrence of key (utility/commandparser.cpp)."""
    if key in args:
        i = args.index(key)
        if i + 1 < len(args):
            return args[i + 1]
    return default


def read_text_graph(path: str) -> tuple[list[int], list[tuple[int, int]]]:
    with open(path, encoding="ascii") as handle:
        tokens = handle.read().split()
    assert tokens[0] == "t", "the first token must be t"
    n = int(tokens[1])
    labels: list[int] = []
    degrees: list[int] = []
    edges: list[tuple[int, int]] = []
    i = 3
    while i < len(tokens):
        if tokens[i] == "v":
            assert int(tokens[i + 1]) == len(labels), "v lines must come in id order"
            labels.append(int(tokens[i + 2]))
            degrees.append(int(tokens[i + 3]))
            i += 4
        elif tokens[i] == "e":
            edges.append((int(tokens[i + 1]), int(tokens[i + 2])))
            i += 3
        else:
            raise ValueError(f"unexpected token {tokens[i]!r}")
    assert len(labels) == n and len(edges) == int(tokens[2]), "counts differ from the t line"
    seen = Counter(v for e in edges for v in e)
    assert all(seen[v] == d for v, d in enumerate(degrees)), "declared degrees differ from the edges"
    return labels, edges


def read_csr_graph(prefix: str) -> tuple[list[int], list[tuple[int, int]]]:
    import numpy as np

    head = np.fromfile(prefix + "_deg.bin", dtype="<u4")
    assert int(head[0]) == 4
    n, m = int(head[1]), int(head[2])
    degrees = head[3:3 + n].astype(np.int64)
    adjacency = np.fromfile(prefix + "_adj.bin", dtype="<u4").astype(np.int64)
    labels = np.fromfile(prefix + "_label.bin", dtype="<u4").astype(np.int64)
    assert adjacency.shape[0] == 2 * m == int(degrees.sum()) and labels.shape[0] == n
    rows = np.repeat(np.arange(n, dtype=np.int64), degrees)
    keep = rows < adjacency
    return labels.tolist(), list(zip(rows[keep].tolist(), adjacency[keep].tolist()))


def meta(labels: list[int], edges: list[tuple[int, int]]) -> list[str]:
    degree = Counter(v for e in edges for v in e)
    frequency = Counter(labels)
    sigma = max(len(frequency), max(labels) + 1) if labels else 0
    return [f"|V|: {len(labels)}, |E|: {len(edges)}, |\u03a3|: {sigma}",
            f"Max Degree: {max(degree.values(), default=0)}, Max Label Frequency: {max(frequency.values(), default=0)}"]


def matches(q_labels, q_edges, d_labels, d_edges, limit):
    import networkx as nx
    from networkx.algorithms import isomorphism

    data = nx.Graph()
    data.add_nodes_from((v, {"l": x}) for v, x in enumerate(d_labels))
    data.add_edges_from(d_edges)
    query = nx.Graph()
    query.add_nodes_from((v, {"l": x}) for v, x in enumerate(q_labels))
    query.add_edges_from(q_edges)
    matcher = isomorphism.GraphMatcher(data, query, node_match=lambda a, b: a["l"] == b["l"])
    for count, mapping in enumerate(matcher.subgraph_monomorphisms_iter()):
        if count >= limit:
            return
        phi = [0] * len(q_labels)
        for d, q in mapping.items():
            phi[q] = d
        yield phi


def summary(count: int) -> list[str]:
    return [SEP, "Release memories...", SEP,
            "Load graphs time (seconds): 0.0010", "Filter vertices time (seconds): 0.0002",
            "Build table time (seconds): 0.0001", "Generate query plan time (seconds): 0.0000",
            "Enumerate time (seconds): 0.0003", "Preprocessing time (seconds): 0.0003",
            "Total time (seconds): 0.0006", "Memory cost (MB): 0.0123", f"#Embeddings: {count}",
            f"Call Count: {count + 3}", "Per Call Count Time (nanoseconds): 12.5000", "End."]


def say(*lines: str) -> None:
    for line in lines:
        print(line)
    sys.stdout.flush()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    mode = os.environ.get("FAKE_SM_MODE", "found")
    args = sys.argv[1:]
    if mode == "echo_args":
        say("args = " + " ".join(args))
        return 1
    printing = "-print_embeddings" in args
    num = option(args, "-num", "MAX")
    say("Command Line:", f"\tData Graph CSR: {option(args, '-csr')}", f"\tData Graph: {option(args, '-d')}",
        f"\tQuery Graph: {option(args, '-q')}", f"\tFilter Type: {option(args, '-filter', 'CFL')}",
        f"\tOrder Type: {option(args, '-order', 'GQL')}", f"\tEngine Type: {option(args, '-engine', 'LFTJ')}",
        f"\tOutput Limit: {num}", f"\tTime Limit (seconds): {option(args, '-time_limit', '60')}",
        f"\tOrder Num: {option(args, '-order_num', '100')}",
        f"\tDistribution File Path: {option(args, '-dis_file', 'temp.distribution')}",
        f"\tPrint Embeddings: {'true' if printing else 'false'}", SEP, "Load graphs...")
    if mode == "sleep":
        time.sleep(30)
        return 0
    q_labels, q_edges = read_text_graph(option(args, "-q"))
    csr = option(args, "-csr")
    d_labels, d_edges = read_csr_graph(csr) if csr else read_text_graph(option(args, "-d"))
    say("-----", "Query Graph Meta Information", *meta(q_labels, q_edges), "-----", *meta(d_labels, d_edges),
        SEP, "Start queries...", "-----", "Filter candidates...", "-----", "Build indices...",
        "Index Info: CoreTable(0), TreeTable(0), LeafTable(0)", "Total Cardinality: 0.0", "-----",
        "Generate a matching order...", "Query Plan: " + "".join(f"{i} " for i in range(len(q_labels))),
        "-----", "Enumerate...")
    if mode == "crash":
        say("embedding: 0:")
        return 139
    if mode == "limit":
        say("Spectrum Order 1 status: Timeout", *summary(0))
        return 0
    if mode == "none":
        say(*summary(0))
        return 0
    if mode == "unpatched":
        say(*summary(1))
        return 0
    if mode == "invalid":
        say("embedding: " + " ".join(f"{u}:0" for u in range(len(q_labels))), *summary(1))
        return 0
    count = 0
    for phi in matches(q_labels, q_edges, d_labels, d_edges, 2**63 if num == "MAX" else int(num)):
        count += 1
        if printing:
            say("embedding: " + " ".join(f"{u}:{v}" for u, v in enumerate(phi)))
    say(*summary(count))
    return 0


if __name__ == "__main__":
    sys.exit(main())
