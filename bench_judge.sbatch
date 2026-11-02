#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=bench-judge
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=bench-judge-%j.out
#
# GPU part of judging (benchmark/workplan.md §4): Gemma 4 31B served on the GPUs matches the nodes of every
# finished run (stories and groups at all usage cuts) and stores the answers in each run's judge cache.
# The scores themselves are computed on the laptop from that cache (Killarney is GPU-only).
# Also the LLM-merge aggregation A4 (it needs the same LLM).
#   sbatch bench_judge.sbatch
set -uo pipefail
OUT=grid_cluster/_judge
mkdir -p "$OUT"
source bench_lib.sh
[ -d grid ] || tar -xzf grid_seed.tar.gz          # the OpenRouter runs needed here (A4 input, one unjudged run)
start_llm gemma 32768
export JUDGE_URL=$LLM_URL JUDGE=llm JUDGE_SLUG=gemma-4-31b-it JUDGE_ONLY=1

echo "### A4: LLM merge of our story graphs"
for R in grid/*__gemma__jev grid_cluster/*__gemma__kev27b; do
  [ -f "$R/abstraction.json" ] || continue
  A=grid_cluster/agg__A4__$(basename "$R")
  [ -f "$A/network.json" ] || python benchmark/alt_aggregate.py A4 "$R" "$A"
done

echo "### judging"
for R in grid/*__*__* grid_cluster/*__*__* grid_cluster/agg__*; do
  [ -d "$R" ] && [ -f "$R/abstraction.json" ] || continue
  [ -f "$R/evaluation_gemma-4-31b-it_cut30.json" ] && continue        # already scored with this judge
  name=$(basename "$R" | sed 's/^agg__[^_]*__//')
  world=$(echo "$name" | sed 's/__.*//')
  echo "--- $R ($world)"
  python benchmark/evaluate_run.py "$world" "$R" 5,10,15,20,25,30
done
stop_llm
