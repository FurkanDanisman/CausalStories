#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=causal-heldout
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=heldout-%j.out
#
# Score the 3 models on the held-out set (heldout.txt), judged by Gemma-4-31B.
# Each model is loaded once and extracts all examples; then the judge scores every
# saved graph and reports MEAN P/R/F1 per model. Submit: sbatch run_heldout.sbatch
set -uo pipefail

module load gcc opencv
source .venv/bin/activate
export VLLM_MAX_LEN=8192 VLLM_TP=2

OUT=out_heldout
EXF=heldout.txt
JUDGE=$HOME/projects/aip-rgrosse/furkanbd/model-weights/gemma-4-31B-it
TAGS=(qwen3.5-27b            llama3.1-8b                            gemma3-12b)
PATHS=(/model-weights/Qwen3.5-27B  /model-weights/Meta-Llama-3.1-8B-Instruct  /model-weights/gemma-3-12b-it)

mkdir -p "$OUT"; rm -f "$OUT"/*.graph.json "$OUT"/summary*.json

for i in "${!TAGS[@]}"; do
  echo "############ EXTRACT ${TAGS[$i]} over $(grep -cve '^\s*$' "$EXF") examples ############"
  python run.py --mode extract --backend vllm --model "${PATHS[$i]}" \
    --tag "${TAGS[$i]}" --examples-file "$EXF" --outdir "$OUT"
done

echo "############ JUDGE (gemma-4-31b) ############"
python run.py --mode judge --backend vllm --model "$JUDGE" --outdir "$OUT"

echo "############ RESULTS (means over held-out set) ############"
cat "$OUT/summary.json"
