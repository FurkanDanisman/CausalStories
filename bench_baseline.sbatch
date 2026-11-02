#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=bench-baseline
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=bench-baseline-%x-%j.out
#
# Competing approaches C1-C6 (benchmark/workplan.md §2) with Gemma 4 31B as their LLM.
#   C1-C4 (one world per job): their story graphs, then OUR aggregation (Kev-27B + Gemma), like our method.
#     sbatch --export=ALL,METHOD=C1,WORLD=homelessness bench_baseline.sbatch
#     C2-C4 reuse our node lists: grid_cluster/<world>__gemma__kev27b/chain_nodes.json (from bench_kev27b),
#     or extract them here if that run is not there yet.
#   C5 and C6 (all three worlds in one job; their own aggregation is done on the laptop for C5):
#     sbatch --export=ALL,METHOD=C5 bench_baseline.sbatch
#   SMOKE=1: first 2 stories only.
set -uo pipefail
GEN=gemma
WORLDS=${WORLD:-"homelessness career mental_health"}
for WORLD in $WORLDS; do
  OUT=grid_cluster/${WORLD}__baseline__${METHOD}${SMOKE:+_smoke}
  mkdir -p "$OUT"
  source bench_lib.sh
  TEXTS=${WORLD}_v3.json
  if [ -n "${SMOKE:-}" ]; then python -c "import json;json.dump(json.load(open('$TEXTS'))[:2],open('$OUT/texts.json','w'))"; TEXTS=$OUT/texts.json; fi
  t0=$(date +%s)
  if [ "$METHOD" = C6 ]; then start_llm gemma 65536; else start_llm gemma 32768; fi   # C6 reads all 200 stories
  NODES=grid_cluster/${WORLD}__gemma__kev27b/chain_nodes.json
  if [[ "$METHOD" =~ ^C[234]$ ]] && { [ ! -f "$NODES" ] || [ -n "${SMOKE:-}" ]; }; then
    python run.py --mode chain-nodes --backend openai --base-url "http://127.0.0.1:$LLM_PORT/v1" --model llm \
      --texts-file "$TEXTS" --max-nodes 8 --outdir "$OUT" --workers 16
    NODES=$OUT/chain_nodes.json
  fi
  python benchmark/baselines.py "$METHOD" "$TEXTS" "$OUT" --nodes "$NODES" --workers 16
  stop_llm
  if [[ "$METHOD" =~ ^C[1234]$ ]]; then
    our_aggregation gemma
  fi
  run_info "${MODELS[gemma]}" "baseline $METHOD" $(( $(date +%s) - t0 ))
  cat "$OUT/run_info.json"
done
