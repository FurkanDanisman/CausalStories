# Causal Stories pipeline

The agreed method is `working_latex/method_Oct.tex`; the reference run is `out_api_world100/` with its
results page `results/api_world100_matrix/`. No imputation anywhere: "cannot tell" stays missing.

## Stages (`run.py --mode ...`)

| Mode | Model | Step | Output |
|---|---|---|---|
| `chain-nodes` | generator (e.g. Gemma 4 31B) | S1 Step 1: the K most important events + participants | `chain_nodes.json` |
| `chain-kev` | decision model (Jev / Kev) | S1 Steps 2-3: chain walk + separate influence walk | `<id>__<tag>.chain.json` |
| `merge-kev` | decision model | S2 Step 1: groups, one story at a time (same / opposite / different) | `abstraction.json` |
| `name-groups` | generator | S2 Step 1: high-level group names (and opposite names) | `abstraction.json` |
| `node-happened` | decision model | S2 Step 3: did each named event node happen? | `<id>__happened.json` |
| `implied` | decision model | S2 Step 3: did each group a story does not name happen? (1 / 0 / cannot tell) | `implied.json` |
| `connect` | decision model | S2 Step 4: inferred groups connected on top of each story graph (walk rules) | `<id>__connect.json` |

S2 Steps 5-6 (usage = node has an arrow in the connected graph; matrix of averaged enables / blocks /
none; DerSimonian-Laird interval from `combine.meta`) are computed from these files.

## Run

- Hosted (OpenRouter, one key in `.env`): `./run_api_or.sh texts.json out_folder`
- Killarney (GPU stages only): `bench_kev27b.sbatch`, `bench_baseline.sbatch`, `bench_judge.sbatch` (helpers in `bench_lib.sh`)
- Benchmark (worlds, stories, scorer, competing approaches): `benchmark/`
