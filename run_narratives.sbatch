#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=causal-narratives
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=narratives-%j.out
#
# Extract a causal graph from each narrative in narratives.json with each of the 3
# models (no gold, no judge). Saves out_raw/<model>##<id>.graph.json. Render PNGs
# locally afterwards with render_png.py (Graphviz needed only for the PNG step).
set -uo pipefail

module load gcc opencv
source .venv/bin/activate
export VLLM_MAX_LEN=8192 VLLM_TP=2

OUT=out_raw
TAGS=(qwen3.5-27b            llama3.1-8b                            gemma3-12b)
PATHS=(/model-weights/Qwen3.5-27B  /model-weights/Meta-Llama-3.1-8B-Instruct  /model-weights/gemma-3-12b-it)

mkdir -p "$OUT"; rm -f "$OUT"/*.graph.json

for i in "${!TAGS[@]}"; do
  echo "############ ${TAGS[$i]} ############"
  python run.py --mode raw-extract --backend vllm --model "${PATHS[$i]}" \
    --tag "${TAGS[$i]}" --texts-file narratives.json --outdir "$OUT"
done

echo "Done. Pull out_raw/*.graph.json to your Mac and run:  python render_png.py"
