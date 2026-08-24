#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=causal-sweep
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=sweep-%j.out
#
# A/B the paper-only prompt variant on the held-out example:
#   baseline  = current prompts
#   paper_v1  = explicit participant head/tail direction + salience/connected/SVO
#               node pruning, all grounded strictly in the paper  [experiments/paper_v1]
# Each arm runs all 3 models + the fixed judge into out_<arm>/.
set -uo pipefail

module load gcc opencv
source .venv/bin/activate
export VLLM_MAX_LEN=8192 VLLM_TP=2

EX=docid_CNN19980227.2130.0067_sentid_10
SPLIT=train
JUDGE=$HOME/projects/aip-rgrosse/furkanbd/model-weights/gemma-4-31B-it
TAGS=(qwen3.5-27b            llama3.1-8b                            gemma3-12b)
PATHS=(/model-weights/Qwen3.5-27B  /model-weights/Meta-Llama-3.1-8B-Instruct  /model-weights/gemma-3-12b-it)

for V in baseline paper_v1; do
  echo "################## VARIANT: $V ##################"
  OUT=out_$V
  mkdir -p "$OUT"; rm -f "$OUT"/*.graph.json "$OUT"/summary.json
  if [ "$V" = baseline ]; then unset PROMPT_TEMPLATE_DIR; else export PROMPT_TEMPLATE_DIR="experiments/$V"; fi

  for i in "${!TAGS[@]}"; do
    python run.py --mode extract --backend vllm --model "${PATHS[$i]}" \
      --tag "${TAGS[$i]}" --torque-id "$EX" --split "$SPLIT" --outdir "$OUT"
  done
  python run.py --mode judge --backend vllm --model "$JUDGE" \
    --torque-id "$EX" --split "$SPLIT" --outdir "$OUT"
done
unset PROMPT_TEMPLATE_DIR

echo "################## SWEEP RESULTS ##################"
for V in baseline paper_v1; do
  echo "== $V =="; cat "out_$V/summary.json"; echo
done
