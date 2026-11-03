#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=bench-kev27b
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=bench-kev27b-%x-%j.out
#
# Our method with a local Kev decision model (DEC), one world x one generator (benchmark/workplan.md §1).
# Generator (nodes, names) on vLLM; Kev ($DEC) for arrows, groups, did-it-happen and connect; one model on the GPUs at a time.
#   sbatch --export=ALL,GEN=gemma,WORLD=homelessness[,DEC=kev9b] bench_kev27b.sbatch
#   DEC: kev27b (default) | kev9b
#   GEN: gemma | qwen | llama | gptoss     WORLD: homelessness | career | mental_health
#   SMOKE=1 runs the first 2 stories only (to check the job and measure speed).
set -uo pipefail
OUT=grid_cluster/${WORLD}__${GEN}__${DEC:-kev27b}${SMOKE:+_smoke}
mkdir -p "$OUT"
source bench_lib.sh
TEXTS=${WORLD}_v3.json
if [ -n "${SMOKE:-}" ]; then python -c "import json;json.dump(json.load(open('$TEXTS'))[:2],open('$OUT/texts.json','w'))"; TEXTS=$OUT/texts.json; fi
t0=$(date +%s)

echo "### $GEN: events + participants"
start_llm "$GEN"
python run.py --mode chain-nodes --backend openai --base-url "http://127.0.0.1:$LLM_PORT/v1" --model llm \
  --texts-file "$TEXTS" --max-nodes 8 --outdir "$OUT" --workers 16
stop_llm

echo "### Kev ($DEC): chain walk"
start_kev
python run.py --mode chain-kev --kev-url "http://127.0.0.1:$KEV_PORT" --nodes-file "$OUT/chain_nodes.json" \
  --tag gemma-kev --outdir "$OUT"
stop_kev

echo "### our aggregation, GPU part (Kev $DEC groups, $GEN names, Kev $DEC did-it-happen + connect; Steps 5-6 run locally)"
our_aggregation "$GEN"

run_info "${MODELS[$GEN]}" "${KEVS[$DEC]} (local)" $(( $(date +%s) - t0 ))
cat "$OUT/run_info.json"
