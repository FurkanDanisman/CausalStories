#!/bin/bash
# Compare several local models on ONE held-out example, judged by a fixed model.
# One process per model => clean GPU teardown between models (vLLM frees on exit).
#
# Usage:  bash compare_models.sh
# Requires: vllm installed; GPUs visible. Set VLLM_TP for tensor-parallel (e.g. 2).
set -euo pipefail

EX=docid_CNN19980227.2130.0067_sentid_10   # held-out; disjoint from the prompt's example
SPLIT=train
OUT=out_compare
JUDGE=$HOME/projects/aip-rgrosse/furkanbd/model-weights/gemma-4-31B-it

# model tag  ->  weights path  (models UNDER TEST)
TAGS=(qwen3.5-27b            llama3.1-8b                            gemma3-12b)
PATHS=(/model-weights/Qwen3.5-27B  /model-weights/Meta-Llama-3.1-8B-Instruct  /model-weights/gemma-3-12b-it)

mkdir -p "$OUT"

# ---- Pass 1: extraction (separate process per model) ----
for i in "${!TAGS[@]}"; do
  echo "############ EXTRACT ${TAGS[$i]} ############"
  python run.py --mode extract --backend vllm --model "${PATHS[$i]}" \
    --tag "${TAGS[$i]}" --torque-id "$EX" --split "$SPLIT" --outdir "$OUT"
done

# ---- Pass 2: judge once over all extracted graphs ----
echo "############ JUDGE (gemma-4-31b) ############"
python run.py --mode judge --backend vllm --model "$JUDGE" \
  --torque-id "$EX" --split "$SPLIT" --outdir "$OUT"

echo "Done. See $OUT/summary.json and $OUT/*.comparison.html"
