#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=causal-homeless
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=homeless-%j.out
#
# Aggregation on the synthetic homelessness story-set (6 perspectives, no gold):
# extract a causal graph from each perspective with each model, then canonicalize +
# aggregate into ONE probability-weighted common graph per model. Render PNGs locally
# afterwards with render_png.py. Submit: sbatch run_homeless.sbatch
set -uo pipefail

module load gcc opencv
source .venv/bin/activate
export VLLM_MAX_LEN=8192 VLLM_TP=2

SYNTH=out_synth_h
AGG=out_agg_h
JUDGE=$HOME/projects/aip-rgrosse/furkanbd/model-weights/gemma-4-31B-it
TAGS=(qwen3.5-27b            llama3.1-8b                            gemma3-12b)
PATHS=(/model-weights/Qwen3.5-27B  /model-weights/Meta-Llama-3.1-8B-Instruct  /model-weights/gemma-3-12b-it)

mkdir -p "$SYNTH"; cp homelessness.variants.json homelessness_individuals.variants.json "$SYNTH"/
mkdir -p "$AGG"; rm -f "$AGG"/*.varsgraphs.json "$AGG"/*.aggregated.graph.json

for i in "${!TAGS[@]}"; do
  echo "############ AGG-EXTRACT ${TAGS[$i]} ############"
  python run.py --mode agg-extract --backend vllm --model "${PATHS[$i]}" \
    --tag "${TAGS[$i]}" --synth-dir "$SYNTH" --outdir "$AGG"
done

echo "############ AGG-COMBINE (Gemma-4-31B canonicalizer) ############"
python run.py --mode agg-combine --backend vllm --model "$JUDGE" \
  --outdir "$AGG" --agg-min-count 2

echo "Done. Pull $AGG/*.aggregated.graph.json to your Mac and run:  python render_png.py out_agg_h out_png_h"
