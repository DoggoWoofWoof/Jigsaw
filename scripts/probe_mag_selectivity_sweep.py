"""MAG label-selectivity crossover sweep (retrieval-only, no Glasgow).

Runs the sweep at MAG scale with strict information parity: BOTH the
classical FeatureIndex and the learned retriever are evaluated on the SAME
feature-derived label, coarsened identically by a bucket count K
(label = md5(feature) mod K). We sweep K from near-unique down to coarse and
measure, per query, the worst-true-partition rank (lower = better localization).

Crucial fairness property the sweep makes explicit:
  * The learned retriever ranks partitions by EMBEDDING similarity and never
    looks at the label -> its rank is IDENTICAL at every K (a flat line). The
    encoder keeps using full features; nothing about it is "coarsened".
  * FeatureIndex ranks purely by label coverage -> as K shrinks, its labels
    stop being discriminative and its rank degrades.
The K at which FI's rank crosses the (flat) neural line is the selectivity
crossover: the label granularity below which learned retrieval earns its keep.

Also emits the offline selectivity statistic (median coarse-partitions-per-label)
at each K, so the crossover can be plotted against a label-only quantity.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, statistics as st
from collections import defaultdict
import numpy as np
import torch

# Our own trusted checkpoints/hierarchies are full PyG objects.
_orig_torch_load = torch.load
def _torch_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)
torch.load = _torch_load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_glasgow as bench
import benchmark_overlap_glasgow_cascade as C
from src.model import get_graph_embedding


def max_true_rank(ranking, true_coarse):
    pos = {int(c): r for r, c in enumerate(ranking)}
    return max((pos.get(int(t), 10**9) for t in true_coarse), default=10**9)


def base_feature_hash_per_node(data, cache_dir, cache_key):
    """md5(feature) as a big int per node, computed ONCE; bucket labels are base % K."""
    path = os.path.join(cache_dir, f"{cache_key}_base_feature_hash.pt") if cache_dir else ""
    if path and os.path.exists(path):
        print(f"[sweep] loading cached base feature hashes: {path}", flush=True)
        return torch.load(path, map_location="cpu")
    print(f"[sweep] computing base feature hashes for {data.num_nodes} nodes...", flush=True)
    vals = [C._feature_bucket_hash_int(data.x[i]) for i in range(data.num_nodes)]
    # md5 ints exceed int64; keep the low 62 bits (bucket residues for K<=2^62 are
    # exact because 2^62 is a multiple of every K we use only if K | 2^62 -- so we
    # instead store residues per K directly below and keep the python big-ints here).
    if path:
        torch.save(vals, path)
        print(f"[sweep] saved base feature hashes: {path}", flush=True)
    return vals


def query_bucket_labels(query, bucket_count):
    return [C._feature_bucket_hash_int(query.x[i]) % bucket_count for i in range(query.num_nodes)]


def median_parts_per_label(node_labels_sets, ids):
    """Offline selectivity: for each distinct label, how many coarse partitions contain it."""
    counts = defaultdict(int)
    for pid in ids:
        nl = node_labels_sets.get(int(pid))
        if nl is None:
            continue
        for lab in nl.tolist():
            counts[int(lab)] += 1
    if not counts:
        return -1.0, 0
    vals = list(counts.values())
    return float(st.median(vals)), len(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mag")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--hierarchy-path", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-dir", default="cache/mag_selectivity_sweep")
    ap.add_argument("--seeds", default="20260607")
    ap.add_argument("--sizes", default="20,50,100")
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--query-types", default="positive")
    ap.add_argument("--buckets", default="1000000,100000,10000,1000,100,10",
                    help="comma list of K (bucket counts); near-unique on the left")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    device = torch.device("cpu")
    os.makedirs(args.cache_dir, exist_ok=True)
    data = C.load_named_data(args.dataset, args.data_root)
    cache_key = C.safe_cache_key(
        args.dataset, C.path_fingerprint(args.hierarchy_path),
        data.num_nodes, data.edge_index.size(1),
    )
    hierarchy = C.load_or_prepare_hierarchy(
        data, args.hierarchy_path, args.cache_dir, cache_key, args.dataset
    )
    if "coarse_part_node_sets" not in hierarchy:
        hierarchy["coarse_part_node_sets"] = {
            int(pid): set(int(n) for n in torch.as_tensor(nodes).view(-1).tolist())
            for pid, nodes in hierarchy["coarse_part_nodes_map"].items()
        }
    ncoarse = len(hierarchy["coarse_part_node_sets"])
    print(f"[sweep] {args.dataset}: {data.num_nodes} nodes, {ncoarse} coarse partitions", flush=True)

    buckets = [int(b) for b in args.buckets.split(",") if b.strip()]

    # base md5 per node (once) -> bucket labels are base % K
    base = base_feature_hash_per_node(data, args.cache_dir, cache_key)
    base_t = None  # built per K to avoid a giant int64 overflow; use python ints

    # neural partition embeddings (label-independent, computed once)
    from probe_multivector_ranking import partition_embed_matrix
    encoder, _ = bench.load_model(args.model, data.x.size(1), device)
    encoder.eval()
    P, coarse_ids = partition_embed_matrix(
        data, hierarchy, encoder, device, args.model, args.cache_dir, args.dataset
    )
    Pn = torch.nn.functional.normalize(P.float(), dim=1)
    print(f"[sweep] neural: {P.size(0)} partition embeddings", flush=True)

    # Build a FeatureIndex per K (distinct cache_key so tags don't collide),
    # and record its offline selectivity statistic.
    fi_by_k = {}
    sel_by_k = {}
    for K in buckets:
        node_labels_override = torch.tensor([int(b) % K for b in base], dtype=torch.long)
        fi = C.build_coarse_feature_index(
            data, hierarchy, args.cache_dir, cache_key + f"_bkt{K}",
            node_labels_override=node_labels_override,
        )
        fi_by_k[K] = fi
        # node_labels sets are fi[0]
        med_ppl, n_labels = median_parts_per_label(fi[0], fi[2])
        sel_by_k[K] = {"median_parts_per_label": med_ppl, "distinct_labels": n_labels}
        print(f"[sweep] K={K}: distinct_labels={n_labels} median_parts_per_label={med_ppl}", flush=True)

    method_cols = [f"fi_bucket_{K}" for K in buckets] + ["neural"]
    rows = []
    fams = ["single", "multi_fine", "multi_coarse", "degree_k_hop", "k_hop", "random_walk"]
    for seed in [int(s) for s in args.seeds.split(",")]:
        for size in [int(s) for s in args.sizes.split(",")]:
            items = C.generate_cascade_queries(data, hierarchy, args.queries, size, seed, args.query_types)
            for it in items:
                if it["is_negative"]:
                    continue
                q = it["query"]; tc = [int(t) for t in it["true_coarse"]]
                row = {"seed": seed, "query_type": it["query_type"],
                       "size": it["target_query_size"], "true_coarse_count": len(tc)}
                # neural (flat across K)
                zq = torch.nn.functional.normalize(
                    get_graph_embedding(q, encoder, device).detach().cpu().float().view(1, -1), dim=1)
                order = torch.argsort((zq @ Pn.t()).view(-1), descending=True).tolist()
                row["max_true_rank_neural"] = max_true_rank([coarse_ids[i] for i in order], tc)
                # FI at each K
                for K in buckets:
                    qlab = query_bucket_labels(q, K)
                    r = C.rank_by_feature_index(q, fi_by_k[K], query_labels=qlab)
                    row[f"max_true_rank_fi_bucket_{K}"] = max_true_rank(r, tc)
                rows.append(row)
            print(f"[sweep] seed={seed} size={size} done ({len(rows)} rows)", flush=True)

    # budgets for FullCov, matching the paper convention (FullCov@1000 on MAG)
    budgets = sorted({max(1, ncoarse // 10), max(1, ncoarse // 4), max(1, ncoarse // 2), ncoarse})
    midb = budgets[len(budgets) // 2]

    def agg(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return {"median_rank": -1, "fullcov": -1.0, "n": 0}
        return {"median_rank": int(st.median(vals)),
                "fullcov": 100.0 * sum(1 for v in vals if v < midb) / len(vals),
                "n": len(vals)}

    byfam = defaultdict(list)
    for r in rows:
        byfam[r["query_type"]].append(r)

    summary = {"dataset": args.dataset, "ncoarse": ncoarse, "fullcov_budget": midb,
               "buckets": buckets, "selectivity": sel_by_k,
               "n_queries": len(rows), "by_family": {}, "overall": {}}
    print(f"\n=== MAG selectivity sweep: median worst-true-partition rank (of {ncoarse}); [FullCov@{midb} %] ===")
    hdr = f"{'K / method':16s}" + "".join(f"{m:>16s}" for m in ["neural"] + buckets_as_str(buckets))
    print(hdr)
    for fam in fams + ["OVERALL"]:
        rs = rows if fam == "OVERALL" else byfam.get(fam, [])
        if not rs:
            continue
        cells = []
        fam_entry = {}
        for m in ["neural"] + [f"fi_bucket_{K}" for K in buckets]:
            a = agg([r.get(f"max_true_rank_{m}") for r in rs])
            fam_entry[m] = a
            cells.append(f"{a['median_rank']}[{a['fullcov']:.0f}]")
        print(f"{fam:16s}" + "".join(f"{c:>16s}" for c in cells))
        if fam == "OVERALL":
            summary["overall"] = fam_entry
        else:
            summary["by_family"][fam] = fam_entry

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    cols = ["seed", "query_type", "size", "true_coarse_count",
            "max_true_rank_neural"] + [f"max_true_rank_fi_bucket_{K}" for K in buckets]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[sweep] wrote {args.output_csv} and {args.output_json}", flush=True)


def buckets_as_str(buckets):
    return [f"fi_bucket_{K}" for K in buckets]


if __name__ == "__main__":
    main()
