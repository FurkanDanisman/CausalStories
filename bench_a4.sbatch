#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=bench-a4
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=bench-a4-%j.out
#
# Aggregation comparison A4 (LLM merge, benchmark/workplan.md §5) with each of the 4 generators, on OUR story
# graphs of every finished Gemma + Kev-27B run (one per world). A1-A3 and A5 need no GPU and run on the laptop.
#   sbatch bench_a4.sbatch            (GENS="gemma qwen" to run a subset)
set -uo pipefail
OUT=grid_cluster/_a4
mkdir -p "$OUT"
source bench_lib.sh
for G in ${GENS:-gemma qwen llama gptoss}; do
  start_llm "$G" 65536
  for R in grid_cluster/*__gemma__kev27b; do
    [ -f "$R/abstraction.json" ] || continue
    A=grid_cluster/agg__A4-${G}__$(basename "$R")
    [ -f "$A/network.json" ] || { t0=$(date +%s); python benchmark/alt_aggregate.py A4 "$R" "$A"; echo "$G $(basename "$R") $(( $(date +%s) - t0 )) s"; }
  done
  stop_llm
done
