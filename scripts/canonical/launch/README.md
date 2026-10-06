# Canonical runs on a Linux GPU machine

The order below takes a clean clone to prepared stores, query sets, trained
encoders, the protocol freeze, the sealed test runs and the numbers the paper
prints. Every step writes a `manifest.json` (git state, inputs with SHA-256,
machine, library versions, allowed environment variables) next to its outputs;
`python -m scripts.canonical.manifest verify <run dir>` re-hashes a run.

Nothing in this pipeline downloads data. Solver builds fetch pinned sources.

## 1. Machine and Python environment

- Linux x86-64. SubgraphMatching needs a CPU with AVX2.
- An NVIDIA GPU and driver for training and serving.
- RAM: corafull, arxiv and mag prepare within 8 GB. The measured peaks are
  1.5 GiB, 0.4 GiB and about 4.6 GiB of commit for mag; the mag METIS worker
  alone takes 4.1 GiB (details in the `prepare_data.py` docstring). products
  needs about 10 GB. papers100M runs out of core, but METIS on its 3.2 billion
  adjacency entries needs roughly 100-150 GB. Partition it on a large-memory
  host and pass `--import-partition`.
- Disk: papers100M needs about 150 GB free during preparation. Work files peak
  near 70 GB, and the store keeps about 80 GB. The other stores are small:
  1.3 GB for mag, about 0.2 GB for CoraFull (its dense uint8 word-presence
  attributes; 0.7 GB when they were float32) and 0.1 GB for arxiv.

The canonical package and its tests were developed with Python 3.13.5 and these
library versions:

```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements-canonical.txt   # numpy 2.3.2, scipy 1.16.1, pymetis 2025.2.1, networkx 3.5, ...
# torch 2.8.0: install the CUDA wheel that matches the driver, for example
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pytest tests/canonical -q -p no:cacheprovider
```

`requirements-canonical.txt` pins every package above except torch and
includes `requirements-repro.txt` (the figure scripts). Do not use
`requirements_lightning_rgcn.txt` for these runs. It is the legacy stack
(numpy<2, pymetis 2022.1, torch 2.2.1), and the canonical package was not
tested with it.

## 2. Environment variables

```bash
cp scripts/canonical/launch/env.example .env.canonical   # fill in the values, keep it out of git
set -a; . ./.env.canonical; set +a
```

`JIGSAW_STORE_ROOT` is required. `JIGSAW_RUNS_ROOT` collects logs, query sets
and runs. Manifests record `JIGSAW_DATA_ROOT`, `JIGSAW_STORE_ROOT`, the solver
paths and `CUDA_VISIBLE_DEVICES`.

## 3. External solvers

```bash
bash external/build_glasgow.sh             # external/bin/glasgow_subgraph_solver
bash external/build_subgraphmatching.sh    # external/bin/subgraphmatching/SubgraphMatching.out
bash external/build_gnnpe.sh               # external/bin/gnnpe/main (E12 only)
export GLASGOW_SOLVER_BIN="$PWD/external/bin/glasgow_subgraph_solver"
export SUBGRAPHMATCHING_BIN="$PWD/external/bin/subgraphmatching/SubgraphMatching.out"
export GNNPE_BIN="$PWD/external/bin/gnnpe/main"
```

Commits and patches are pinned in `external/PINS.json`. Each build writes
`external/bin/<tool>.build.json` with the binary SHA-256. GNN-PE has no license
file at its pinned commit, so its source, patch and binary stay on the run
machine. `external/README.md` lists the semantic differences of every tool and
the hazards each build probes.

Partitioners are optional. prepare_data.py uses pymetis unless gpmetis is on
PATH or `GPMETIS_BIN` is set (`--metis-tool auto`). The two tools give different
partitions, so pick one for every dataset of a paper run and pass
`--metis-tool pymetis` or `--metis-tool gpmetis` explicitly. The tool, version
and options land in `part/primary/partition.json`. METIS k-way can leave a part
empty: with pymetis 2025.2.1 and seed 1, ogbn-mag at P=2000 gets 1999 non-empty
parts. Each empty part then takes one side of a 2-way METIS split (same seed
and ufactor) of a largest part, so every METIS partition has exactly the
requested number of blocks. `partition.json` records each fill
(`metis_empty_parts`, `empty_parts_filled`). KaHIP `kaffpa`
(`KAHIP_BIN`) is only used for the `kahip_fast` sensitivity partition; without
it that partition is skipped and the skip is recorded.

## 4. Datasets (offline copies)

Put the raw files under `$JIGSAW_DATA_ROOT` (default `<repo>/data`). A missing
file stops preparation with its path in the message.

| dataset | files |
|---|---|
| corafull | `CoraFull/cora/raw/cora.npz` (PyG CitationFull) |
| arxiv | `ogbn_arxiv/ogbn_arxiv/raw/{num-node-list,num-edge-list,edge,node-feat,node-label}.csv.gz` |
| mag | `ogbn_mag/ogbn_mag/raw/`: `num-node-dict.csv.gz`, `triplet-type-list.csv.gz`, `relations/*/{edge,num-edge-list}.csv.gz`, `node-feat/paper/node-feat.csv.gz`, `node-label/paper/node-label.csv.gz` |
| products | `ogbn_products/ogbn_products/raw/` (same files as arxiv) |
| papers100m | `ogbn_papers100M/ogbn_papers100M/raw/{data.npz,node-label.npz}` |

## 5. Stores

```bash
bash scripts/canonical/launch/prepare_all.sh --metis-tool pymetis --alternatives
# products and papers100M on request:
bash scripts/canonical/launch/prepare_all.sh --metis-tool pymetis --with-products --with-papers100m
```

This runs `python -m scripts.canonical.prepare_data` for each dataset under
memprobe. Logs and peak RSS go to `$JIGSAW_RUNS_ROOT/prepare/`. Each store gets:

- `meta.json` (schema, types, attribute dtype, relation pairs, source SHA-256s,
  normalisation counts);
- `prepare_report.json` (sizes, degree quantiles, block and fine sizes, cut
  edges, sigma code counts, per-stage wall time and peak RSS);
- `_prepare_run/manifest.json`.

Reruns skip finished stages; `--force` redoes a store.

CoraFull note: `cora.npz` holds L2-normalised term weights. The store keeps
word presence (`x > 0 -> 1`, uint8), which is how PyG's CitationFull and the
legacy pipeline read them. `meta.json` `attribute_notes` records the raw value
statistics, and loading fails on negative values.

## 6. Query sets

Dev and validation masters are public (`protocol.SEEDS`). Build every master,
and exclude the node sets of every directory built before it, so node sets never
repeat across masters or splits:

```bash
Q="$JIGSAW_RUNS_ROOT/queries"
for d in corafull arxiv mag; do
  done_dirs=()
  for split in dev validation; do
    masters=($(python -c "from scripts.canonical import protocol; print(*protocol.SEEDS['$split'])"))
    for i in "${!masters[@]}"; do
      out="$Q/$d/$split/${masters[$i]}"
      python -m scripts.canonical.queries build --store-root "$JIGSAW_STORE_ROOT" --dataset "$d" \
        --split "$split" --master-index "$i" --out "$out" ${done_dirs[@]+--exclude "${done_dirs[@]}"}
      done_dirs+=("$out")
    done
  done
done
```

`launch/build_queries.py` runs the same loop as a resumable driver. It keeps
every master whose manifest is complete and stops at the first failure, so a
later master never excludes an unfinished one:

```bash
python -m scripts.canonical.launch.build_queries --store-root "$JIGSAW_STORE_ROOT" --queries-root "$Q"
```

Within a master, a draw is redrawn when its node set is taken or when its query
is isomorphic to an earlier query in a regime whose set holds both. A node
bijection that keeps every edge, type and label gives the two queries the same
matches, so the set would count one query twice (on mag, stars of featureless
nodes do this). A negative whose modification is isomorphic to an earlier query
of its regime moves on to its next modification attempt. Check the saved sets
before any run reads them:

```bash
python -m scripts.canonical.queries duplicates "$Q"    # exit 1 on any isomorphic pair
```

The test split (and `reserve_a`, `reserve_b`) is sealed. Its masters are
committed in `protocol.SEAL_COMMITMENT`, and the seeds file lives outside git
(`internal/sealed/SEALED_SEEDS.json`, restored from its backup only for the
sealed run). Section 10 builds it after the protocol freeze.

### Negative certification (E14)

A generated negative is a candidate until complete search decides it: a
changed label or an added edge can still leave an embedding somewhere in the
data graph. `certify_negatives.py` decides every negative of a set in two
stages, FilterAll on the label-filtered whole graph (no component cap, 3600 s;
its NONE is a proof) and then classical GQL on the whole graph (3600 s; its
NONE counts only when the search finished). Certify every dev and validation
set:

```bash
for d in corafull arxiv mag; do
  for qs in "$Q/$d"/dev/* "$Q/$d"/validation/*; do
    python -m scripts.canonical.certify_negatives run --store-root "$JIGSAW_STORE_ROOT" --dataset "$d" \
      --queries "$qs" --solver-bin "$GLASGOW_SOLVER_BIN" --classical-binary "$SUBGRAPHMATCHING_BIN"
  done
done
```

Each set gets `<set>/certification/<regime>.json` (`jigsaw-certification-v1`),
and the query files never change. A negative with a checked witness is
certified `found` and counts as a positive with that witness. One that both
stages leave open is `undecided`: the reducer drops it from the negatives and
reports the count. `--shard i/n` splits a set over processes, each in
`certification/shard<i>of<n>/`; the last one to finish writes the merged file,
and `certify_negatives finalize --queries <set>` redoes the merge. Rerunning a
command resumes it.

## 7. Encoder training

```bash
python -m scripts.canonical.launch.train_matrix --store-root "$JIGSAW_STORE_ROOT" --queries-root "$Q" \
  --run-root "$JIGSAW_RUNS_ROOT" --models-root "$JIGSAW_RUNS_ROOT/models" --out matrix.json
python -m scripts.canonical.launch.train_matrix ... --execute    # runs the rows here, resumable
```

Validation files are read from `$Q/<dataset>/validation/<master>/<regime>.npz`.
Every row selects its checkpoint on the checkpoint-selection master and holds
out the node sets of all three validation masters. `est_gpu_hours` in
`matrix.json` is a rough estimate; replace it with the measured time of the
first rows before planning the rest. `--execute` skips a row only when both
exports exist and its run manifest is complete. An interrupted row resumes
from its last checkpoint.

One row keeps the GPU about a quarter busy, because the trainer builds its batch
inputs on the CPU. `--jobs N` therefore runs up to N ablation rows at once on
the one GPU, each with `OMP_NUM_THREADS // N` threads and its output in
`<run_dir>.log`. A main row starts only when no other row runs, and nothing
starts beside it, because E17 reports its training time. After a failed row no
new row starts unless `--keep-going` is given. Measure the GPU memory of one
ablation row of a dataset before choosing N for it: every line of a row's
`train_log.jsonl` records the GPU peaks of its epoch.

Each epoch line also records the load of the whole machine next to the row. E17
reports the summed epoch seconds and, from the epochs during which other work
kept fewer than one CPU busy, the quiet training time (the median quiet epoch
times the epochs, plus the validations; `offline_build.quiet_training`).

## 8. Serving inputs

The cascade reads a label index per regime, ranker inputs per partition and,
for the learned rankers, block embeddings per model, regime and partition. The
numpy encoder builds the embeddings, and `serving check` compares them with the
trainer's torch export. `--run-dir` records a build's wall time, peak memory and
the files it wrote in a manifest (`offline_build.py`), which E17 reads:

```bash
M="$JIGSAW_RUNS_ROOT/models"; O="$JIGSAW_RUNS_ROOT/offline"
parts=(primary $(python -c "from scripts.canonical import run_experiment as rx; print(*rx.sensitivity_partitions())"))
for d in corafull arxiv mag; do
  python -m scripts.canonical.cascade build-label-index --root "$JIGSAW_STORE_ROOT" --dataset "$d" \
    --run-dir "$O/$d/label_index"
  for part in "${parts[@]}"; do
    [ -f "$JIGSAW_STORE_ROOT/$d/part/$part/partition.json" ] || continue
    python -m scripts.canonical.rankers build --root "$JIGSAW_STORE_ROOT" --dataset "$d" --partition "$part" \
      --run-dir "$O/$d/rankers-$part"
  done
  # the main-arm encoder of every training seed and trained regime
  for id in $(python -c "from scripts.canonical import protocol as p, serving as s
print(*sorted({s.default_model_id('$d', r, int(x)) for r in p.regimes_for('$d') for x in p.SEEDS['training']}))"); do
    python -m scripts.canonical.serving embed-blocks --root "$JIGSAW_STORE_ROOT" --dataset "$d" --model-dir "$M/$id" \
      --run-dir "$O/$d/embed-$id-primary"
    python -m scripts.canonical.serving check --root "$JIGSAW_STORE_ROOT" --dataset "$d" --model-dir "$M/$id"
  done
  # E16 runs the first seed's R0 encoder on every sensitivity partition
  first=$(python -c "from scripts.canonical import protocol as p, serving as s
print(s.default_model_id('$d', 'R0', int(p.SEEDS['training'][0])))")
  for part in "${parts[@]:1}"; do
    [ -f "$JIGSAW_STORE_ROOT/$d/part/$part/partition.json" ] || continue
    python -m scripts.canonical.serving embed-blocks --root "$JIGSAW_STORE_ROOT" --dataset "$d" \
      --partition "$part" --regimes R0 --model-dir "$M/$first" --run-dir "$O/$d/embed-$first-$part"
  done
done
```

The ablation encoders of E5 and E6 get their embeddings from
`retrieval_eval.py --build-embeddings` (section 11).

## 9. Router threshold, dev runs and the freeze

The router sends a query to the index ranker when its selectivity is at most
tau. Tau is fit per dataset on the validation master whose role is
`router_tau_and_rank_fusion`, from the retrieval costs E4 reports:

```bash
for d in corafull arxiv mag; do
  python -m scripts.canonical.fit_router fit --store-root "$JIGSAW_STORE_ROOT" --dataset "$d" --queries-root "$Q" \
    --models-root "$M" --out "$JIGSAW_RUNS_ROOT/router_fit/$d/router_fit.json"
done
python -m scripts.canonical.fit_router merge --inputs "$JIGSAW_RUNS_ROOT"/router_fit/*/router_fit.json \
  --out "$JIGSAW_RUNS_ROOT/router_fit/router_tau.json"
```

`merge` prints the `RANKERS["router_tau"]` line for `protocol.py`. Every other
choice is made on dev and validation sets as well: run any command of section
11 on a dev master, without `--unseal` and with a `--out` of its own, and settle
what it shows in `protocol.py`. Then freeze:

1. Set `protocol.FROZEN_PROTOCOL_HASH` to `protocol_hash()`.
2. Commit the code, so the tracked code is clean.
3. Run `python -m scripts.canonical.seal verify`.

From here on any edit of `protocol.py` changes the hash, and every sealed step
refuses to run until the protocol is frozen again. A store records the hash it
was prepared under in `meta.json`; that record is provenance. An edit of what a
store holds (normalisation, partitioning, sigma regimes) needs
`prepare_all.sh --force` and everything after it again.

## 10. Sealed test sets

Build the test split, excluding every dev and validation directory, and certify
its negatives. Both steps unseal, and every unseal is appended to
`internal/sealed/UNSEAL_LOG.jsonl`:

```bash
for d in corafull arxiv mag; do
  python -m scripts.canonical.queries build --store-root "$JIGSAW_STORE_ROOT" --dataset "$d" \
    --split test --out "$Q/$d/test" --exclude "$Q/$d"/dev/* "$Q/$d"/validation/* \
    --unseal --reason "sealed run"
  python -m scripts.canonical.certify_negatives run --store-root "$JIGSAW_STORE_ROOT" --dataset "$d" \
    --queries "$Q/$d/test" --solver-bin "$GLASGOW_SOLVER_BIN" --classical-binary "$SUBGRAPHMATCHING_BIN" \
    --unseal --reason "E14 $d"
done
```

A sealed certification runs with the protocol's stages, time limits and
verifier only.

## 11. Experiments

Every command below runs on `$Q/$d/test` with `--unseal --reason "..."`. A
sealed run is a pre-registered experiment: the runner refuses arms, time
limits, regimes, caches or query selections outside `experiment_plan()`, a
protocol that is not frozen, and tracked code that differs from HEAD. Every run
writes `<out>/manifest.json` and one JSON record per query and arm under
`<out>/records/<regime>/`, and rerunning the same command resumes it. A process
needs its own `--out`; `--arms` runs a subset of the plan's arms and `--shard
i/n` a slice of its queries.

```bash
R="$JIGSAW_RUNS_ROOT/runs"; d=mag; T="$Q/$d/test"
common=(--store-root "$JIGSAW_STORE_ROOT" --dataset "$d" --queries "$T")
```

E1, solve rate and exactness at T_max = 120 s. One run gives solved@t for every
t up to 120 s (E3). Spread the queries over cores with one process per shard;
each process runs every arm in a rotated order, so machine drift falls on all
arms alike:

```bash
n=8
for i in $(seq 0 $((n - 1))); do
  taskset -c "$i" python -m scripts.canonical.run_experiment --experiment E1 "${common[@]}" --models-root "$M" \
    --out "$R/E1/$d/shard$i" --shard "$i/$n" --unseal --reason "E1 $d $i/$n" &
done
wait
```

E3x, the 600 s extension, on every query E1 left open for its arms:

```bash
python -m scripts.canonical.extension_uids --e1 "$R/E1/$d"/shard* --dataset "$d" --out "$R/E3x/$d.uids"
python -m scripts.canonical.run_experiment --experiment E3x --uids "$R/E3x/$d.uids" "${common[@]}" \
  --models-root "$M" --out "$R/E3x/$d" --unseal --reason "E3x $d"
```

`$R/E3x/$d.uids.json` records the E1 runs it was read from.

E7 (selectivity sweep, both sides on the same queries), E8 (direct
whole-graph control), E9 (operator ablation) and E16 (sensitivity, one run per
partition):

```bash
python -m scripts.canonical.run_experiment --experiment E7 "${common[@]}" --models-root "$M" \
  --out "$R/E7/$d/jigsaw" --unseal --reason "E7 $d"
for m in classical_gql classical_cfl classical_dpiso; do
  python -m scripts.canonical.run_external --experiment E7 --method "$m" "${common[@]}" \
    --out "$R/E7/$d/$m" --unseal --reason "E7 $d $m"
done
for e in E8 E9; do
  python -m scripts.canonical.run_experiment --experiment "$e" "${common[@]}" --models-root "$M" \
    --out "$R/$e/$d" --unseal --reason "$e $d"
done
for part in "${parts[@]}"; do
  [ -f "$JIGSAW_STORE_ROOT/$d/part/$part/partition.json" ] || continue
  python -m scripts.canonical.run_experiment --experiment E16 --partition "$part" "${common[@]}" \
    --models-root "$M" --out "$R/E16/$d/$part" --unseal --reason "E16 $d $part"
done
```

E12, GNN-PE on class labels (R1). `--prepare-only` builds its offline index
once, outside any query's time; the shards then share it:

```bash
python -m scripts.canonical.run_external --experiment E12 --method gnnpe "${common[@]}" \
  --out "$R/E12/$d/prepare" --prepare-only
python -m scripts.canonical.run_external --experiment E12 --method gnnpe "${common[@]}" \
  --out "$R/E12/$d/run" --unseal --reason "E12 $d"
```

A GNN-PE NONE is not a proof, so its records never count as decided negatives.
The run's manifest records the offline build it used (`extra.prepare`: build
seconds, build time, whether it was reused), and E17 reports that build time
beside Jigsaw's offline stages.

E2, memory, I/O and latency, is Linux only. Each (system, mode, cache, cap,
repeat) runs as its own measured process: summed RSS of the process tree every
20 ms, a cgroup v2 `memory.max` cap with swap off (systemd-run or a delegated
cgroup), and the page cache evicted with `posix_fadvise` for cold cells. A
system killed by its cap is a recorded result:

```bash
python -m scripts.canonical.streaming --experiment E2 "${common[@]}" --models-root "$M" --e2-root "$R/E2" \
  --external --unseal --reason "E2 $d"
```

`--dry-run` lists the cells; `--systems`, `--caches` and `--caps` split the
sweep over machines, and `--retry-failed` measures a failed cell again while
keeping the failed attempt.

E15, enumeration recall of the ladder:

```bash
python -m scripts.canonical.enumeration "${common[@]}" --models-root "$M" --out "$R/E15/$d" \
  --unseal --reason "E15 $d"
```

E4, E5 and E6, retrieval quality per ranker, policy, objective and encoder, and
the layerwise analysis of the E6 encoders:

```bash
python -m scripts.canonical.retrieval_eval run --experiment E4 "${common[@]}" --models-root "$M" \
  --out "$R/E4/$d" --unseal --reason "E4 $d"
for e in E5 E6; do
  python -m scripts.canonical.retrieval_eval run --experiment "$e" --matrix matrix.json --build-embeddings \
    "${common[@]}" --models-root "$M" --out "$R/$e/$d" --unseal --reason "$e $d"
done
ids=$(python -c "import json; print(*[r['model_id'] for r in json.load(open('matrix.json'))
  if r['kind'] == 'architecture' and r['dataset'] == '$d'])")
python -m scripts.canonical.oversmoothing run --store-root "$JIGSAW_STORE_ROOT" --dataset "$d" --regime R0 \
  --models-root "$M" --models $ids --out "$R/E6/layerwise/$d-R0"
python -m scripts.canonical.oversmoothing summarize --inputs "$R"/E6/layerwise/* --out "$R/E6/layerwise/summary.csv"
```

E20, the pattern-overlap audit. Once every main encoder of the dataset is
trained, the audit regenerates each encoder's training queries from its seed
and marks each test query seen (isomorphic, labels included, to a training
query of some main encoder of its regime) or novel. It solves nothing and runs
on the CPU:

```bash
python -m scripts.canonical.pattern_audit --store-root "$JIGSAW_STORE_ROOT" --queries "$T" \
  --run-root "$JIGSAW_RUNS_ROOT" --out "$JIGSAW_RUNS_ROOT/audit/pattern/$d-test" --workers 4 \
  --unseal --reason "E20 $d"
```

Each worker regenerates one run and opens the store, so size `--workers` by
the store's memory. The audit refuses a run that is incomplete or a smoke run,
whose sampler settings differ from the protocol's, or whose held-out or
training node sets do not hash to its manifest's, and it then writes no
`pattern_audit.json`. `--held-out-root "$Q"` rebuilds the held-out node sets
from the canonical layout when the run's manifest records paths on another
machine.

Repeat the block for each dataset (`d=corafull`, `d=arxiv`, `d=mag`). E11 runs
E1, E2 and E4 again with `d=products`. E10 is the scale-out run on papers100M
under a 32 GB cap on a host with at least 256 GB (`protocol.SERVING["scale_out"]`),
cold, against the classical matchers loading the whole graph:

```bash
python -m scripts.canonical.streaming --experiment E10 --store-root "$JIGSAW_STORE_ROOT" --dataset papers100m \
  --queries "$Q/papers100m/test" --models-root "$M" --e2-root "$R/E10" --external --unseal --reason "E10"
```

E13 (the `sm_standard` family) runs inside every set; E17 (offline cost), E18
(failure localization) and E19 (latency by stage) come from the manifests and
records in section 12.

## 12. Reduction and the paper

Three reducers write `numbers.json` files (`numbers_json.py`), and every
number the paper prints comes from them:

- `reducer.py` reads the run records, the certification files, the E2/E10
  measurements and the offline manifests: E1, E3, E3x, E7 with E12, E8, E9,
  E14, E16, E17, E18, E19, PE1 to PE4, and E10 under the prefix `E10.`;
- `reduce_retrieval.py` reads E4, E5, E6 and the layerwise runs: PE5, PE6 and
  the retrieval tables;
- `reduce_enumeration.py` reads E15: recall per ladder step and every lost
  embedding.

The first two also read the pattern audits (E20). The reducer's plan names
`RUNS/audit/pattern/<d>-test` when it exists, and the reducer splits the E1
and E7 solve rates into seen and novel queries; `reduce_retrieval.py
--pattern-audit` splits the E4 coverage of the planted positives the same way.
The split is descriptive and leaves every pre-registered number unchanged.

Two more files hold what the paper states about its setup:
`protocol_numbers.py` writes a key for every numeric protocol setting
(`protocol.TIME.primary_T_seconds`, `protocol.QUERY_SIZES.2`, the seeds, the
memory caps; `--list` prints them all), and `dataset_numbers.py` reads each
store's prepare report into the dataset keys (`data.mag.nodes`,
`data.corafull.sigma.R0.singleton_share`, the partition quality) and the
datasets table.

Each reducer refuses a run of another split, protocol hash or query set, an
incomplete run and an output that no longer matches its manifest.
`launch/reduce_plan.py` writes the reducer's plan from the layout of sections 5
to 11 and prints every directory of that layout it did not find; read that list
before reducing:

```bash
N="$R/numbers"
python -m scripts.canonical.launch.reduce_plan --runs-root "$JIGSAW_RUNS_ROOT" --store-root "$JIGSAW_STORE_ROOT" \
  --queries-root "$Q" --out "$N/plan.json"
python -m scripts.canonical.reducer --plan "$N/plan.json" --out "$N/main"
python -m scripts.canonical.reduce_retrieval --e4 "$R"/E4/* --e5 "$R"/E5/* --e6 "$R"/E6/{corafull,arxiv,mag} \
  --layerwise "$R"/E6/layerwise/*-R0 --pattern-audit "$JIGSAW_RUNS_ROOT"/audit/pattern/*-test \
  --out "$N/retrieval"
python -m scripts.canonical.reduce_enumeration --runs "$R"/E15/* --out "$N/E15"
python -m scripts.canonical.protocol_numbers --out "$N/protocol"
python -m scripts.canonical.dataset_numbers --store-root "$JIGSAW_STORE_ROOT" --out "$N/datasets"
```

The reducer exits 3 when PE1 fails (a false positive, a false negative, a
contradiction or a crash), and `reduce_enumeration` exits 3 on any violation it
lists in `violations.json` (a rejected witness, a lost embedding, an
inconsistent solve, a solver error or a crash). Both write every output first:
a failure is a result of the paper. `build_paper.py` turns the files into `numbers.tex`, tables and figures,
and `validate_paper_numbers.py` checks that the paper prints no hand-typed
result:

```bash
python -m scripts.canonical.build_paper --numbers "$N"/*/numbers.json --out paper/generated
python -m scripts.canonical.validate_paper_numbers --tex paper/jigsaw_vldb2027.tex --generated paper/generated \
  --allowlist scripts/canonical/paper_allowlist.json
```

`build_paper` refuses inputs from a split other than test or from a protocol
that is not frozen; `--allow-dev` builds a dev draft that stamps every page,
table and figure. The paper takes `\input{generated/numbers}` in the preamble
and `\jnum{KEY}`, `\jci{KEY}` and `\jn{KEY}` in the text.

## 13. Running on a shared host

On a machine that other projects schedule through rx (`tools/rx` of the
message-passing-retrieval checkout), every long job runs under
`launch/lowprio.py`. rx admits jobs by project priority (host.json's `priority`
map; the lab host ranks mpr first), then first in, first out with backfill, and
never preempts. The wrapper adds the preemption, with Jigsaw's jobs the ones
that give way:

- the job and its children run at below-normal CPU priority and low I/O
  priority;
- the job yields when a queued job of another project is refused by rx and would
  be admitted if Jigsaw jobs returned their resources, and this job is among the
  fewest that must (taken youngest first; a job holding nothing the blocked job
  lacks keeps running). Without `--yield-to` every other project's job counts at
  once. With `--yield-to mpr` a job of mpr, or of any project rx ranks above
  Jigsaw, counts at once, and another project's job counts once it has waited
  `--peer-wait-s` (default rx's `hold_after_s`, 30 min);
- the job also yields while a foreign compute process runs on its GPU: one
  outside rx (another user) or one of another project's rx job. Under
  `--yield-to` only the named projects' jobs count, so Jigsaw shares a GPU with
  peer projects' jobs as rx assigns it. Jigsaw's own jobs never count.
  Contention must last `--confirm-s` (60 s);
- a yield waits up to `--grace-s` (600 s) for new progress (a file matching
  `--progress`, such as the next checkpoint), stops the child tree and exits 75.
  The trainer resumes from its last checkpoint;
- with `--requeue` the job leaves a waiter. It asks rx for less CPU and memory
  than rx's fit tolerance, so it never keeps another job out. The waiter
  relaunches the job once rx would admit it (lowprio mirrors rx's `decide`:
  priority, older queued jobs that fit, held-back requests and the hold rule),
  the relaunched job would not have to yield at once and, for a GPU job, no
  foreign compute process has run for `--gpu-quiet-s` (600 s). A queued job
  blocked on something the relaunch does not take (another project's GPU, say)
  therefore leaves the free CPUs to Jigsaw. A relaunch still queued after
  `--max-queue-s` is cancelled and retried, so a waiting Jigsaw job never
  reaches rx's `hold_after_s` rule;
- with `--defer CPUS,MEM_GB[,GPUS]` the first job is such a waiter too. Launched
  with `rx run --cpus 1e-11 --mem 1e-11`, it launches the job with the given
  request once the host has room, so no Jigsaw job waits in rx's queue at full
  size.

rx passes a command's arguments to the host after the local shell has expanded
them, so host paths in an rx command are written out. Below, `H_RUNS` and
`H_STORES` hold the host's `JIGSAW_RUNS_ROOT` and `JIGSAW_STORE_ROOT` (the
`[run] env_vars` of rx.toml):

```bash
rx run --cpus 1e-11 --mem 1e-11 --ws train1 --name cal-corafull -- python -m scripts.canonical.launch.lowprio run \
  --campaign cal-corafull --defer 8,32,1 --requeue --yield-to mpr --progress "$H_RUNS/train/**/ckpt_epoch*.pt" -- \
  python -m scripts.canonical.launch.train_matrix --store-root "$H_STORES" --queries-root "$H_RUNS/queries" \
  --run-root "$H_RUNS" --models-root "$H_RUNS/models" --filter 'corafull/*' --kind main --execute
rx exec --env jigsaw-cu128 -- python -m scripts.canonical.launch.lowprio status --campaign cal-corafull
rx exec --env jigsaw-cu128 -- python -m scripts.canonical.launch.lowprio stop --campaign cal-corafull
rx exec --env jigsaw-cu128 -- python -m scripts.canonical.launch.lowprio probe   # the host as a job sees it
```

`stop` lets the current job finish or yield and launches nothing after it.
`rx cancel` on the running job or its waiter stops the campaign at once, and the
next campaign that runs the row resumes it from its last checkpoint. `rx exec`
runs in the default workspace (`ws`) and passes none of rx.toml's job variables,
so status and stop read the state directory from the rx.toml there (push the
current code to `ws` with `rx push`). Both exit 2 when the campaign has no log
in that directory.

A campaign pins its code. Every hop compares the workspace with the first hop
(`manifest.code_fingerprint`), and a changed workspace ends the campaign with
exit 3. `rx run --ws X` pushes to X, so every campaign gets a workspace of its
own and nothing is launched into a workspace whose campaign still runs. A host
workspace has no git, so manifests record the fingerprint (per-file SHA-256
with CRLF folded to LF) in place of the git state. Two checks compare a run with
a commit of this repository:

```bash
python -m scripts.canonical.manifest code-check <run dir> --rev <commit>   # every fingerprinted file
python -m scripts.canonical.closure_check <run dir> --rev <commit>        # the files the run could import
```

`closure_check` requires only the import closure of the run's entry module, read
at the commit: its imports at any depth (imports inside functions too), the
packages above them and the `scripts.canonical.X` modules it starts as
subprocesses. It lists the files outside the closure that differ. A run
therefore stays checkable after later edits to tests, launchers or reducers,
while an edit inside the closure (for the trainer: queries, protocol, manifest,
encoders, store and what they import) fails every earlier run against the new
commit. Freeze those files before the canonical training starts.

Wall times and memory measured next to other jobs are not canonical. The prepare
stage times that E17 reads, solved@T, the E2 and E10 measurements and E19 need a
host with no other job. Request the whole host for them, so rx admits the run
only when nothing else runs and lowprio still yields it to any queued job; rerun
whatever a yield interrupted. Training needs such a window only for a row with
fewer than five quiet epochs, because E17 times training from its quiet epochs
(section 7).
