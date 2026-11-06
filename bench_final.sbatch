#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=bench-final
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=bench-final-%j.out
#
# Last job, after every bench_run job has finished:
#   1. A4 (LLM merge, aggregation comparison) with each of the 4 generators, on OUR story graphs and groups of the
#      Gemma + decision-model run of each world (DEC, Kev-9B by default) (the same input as A1-A3 and A5, which run on the laptop);
#   2. A1-A3, A5 (PC / GES) and C5's clustering (no model; quick CPU steps run here so the judge can see them);
#   3. the judge (Gemma 4 31B) matches the nodes of every finished run and stores its answers in each run's judge cache.
# Scores are computed on the laptop from those caches. Resumable: finished A4 runs and judged runs are skipped, and
# runs whose bench_run job is unfinished are left for the next submission, so it can run before every job is done.
#   sbatch bench_final.sbatch      (GENS="gemma qwen" to run a subset)
set -uo pipefail
OUT=grid_cluster/_logs/final
mkdir -p "$OUT"
source bench_lib.sh

echo "########## A4 with each generator"
for G in ${GENS:-gemma qwen llama gptoss}; do
  start_llm "$G" 65536
  for R in grid_cluster/*__gemma__${DEC}; do
    case $(basename "$R") in agg__*) continue;; esac          # our runs only, not aggregation outputs
    [ -f "$R/abstraction.json" ] || continue
    A=grid_cluster/agg__A4-${G}__$(basename "$R")
    [ -f "$A/network.json" ] || { t0=$(date +%s); python benchmark/alt_aggregate.py A4 "$R" "$A" && printf "A4\t%s\n" $(( $(date +%s) - t0 )) >> "$A/timing.tsv"; }
  done
  stop_llm
done

echo "########## aggregations without a model (A1-A3, A5 PC/GES, C5 clustering): quick CPU steps, run here so the judge sees them"
for R in grid_cluster/*__gemma__${DEC}; do
  case $(basename "$R") in agg__*) continue;; esac
  [ -f "$R/abstraction.json" ] || continue
  for M in A1 A2 A3 pc ges; do
    A=grid_cluster/agg__${M}__$(basename "$R")
    [ -f "$A/network.json" ] || { t0=$(date +%s); python benchmark/alt_aggregate.py $M "$R" "$A" && printf "%s\t%s\n" $M $(( $(date +%s) - t0 )) >> "$A/timing.tsv"; }
  done
done
for R in grid_cluster/*__*__C5; do
  case $(basename "$R") in agg__*) continue;; esac
  [ -d "$R" ] || continue
  A=grid_cluster/agg__C5agg__$(basename "$R")
  [ -f "$A/network.json" ] || { t0=$(date +%s); python benchmark/alt_aggregate.py C5agg "$R" "$A" && printf "C5agg\t%s\n" $(( $(date +%s) - t0 )) >> "$A/timing.tsv"; }
done

echo "########## judge (Gemma 4 31B): node matching only"
start_llm gemma 32768
export JUDGE_URL=$LLM_URL JUDGE=llm JUDGE_SLUG=gemma-4-31b-it JUDGE_ONLY=1
for R in grid_cluster/*; do                                   # every run and aggregation output (not _logs, _smoke)
  case $(basename "$R") in _*) continue;; esac
  [ -d "$R" ] && [ -f "$R/abstraction.json" ] || continue
  [ -f "$R/.judged" ] && continue
  if [ -f "$R/.done_groups" ] && [ ! -f "$R/.done_connect" ]; then echo "--- skip $R (its bench_run job has not finished)"; continue; fi
  name=$(basename "$R" | sed 's/^agg__[^_]*__//')
  world=$(echo "$name" | sed 's/__.*//')
  echo "--- $R ($world)"
  python benchmark/evaluate_run.py "$world" "$R" 5,10,15,20,25,30 && touch "$R/.judged"
done
stop_llm
