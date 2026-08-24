#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=causal-aggregate
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=aggregate-%j.out
#
# FULL aggregation experiment, end to end:
#   1. generate 6 synthetic retellings per base (Gemma-4-31B; skipped if out_synth exists)
#   2. for each model: extract a causal graph from each of the 6 variants
#   3. canonicalize event names across variants, aggregate into ONE probability-
#      weighted graph (edge prob = fraction of variants that drew it, keep >=2/6),
#      score vs the base's gold graph, and render TRUE (gold) vs ESTIMATED (aggregated)
# Output: out_agg/<base>__<model>.comparison.html (one picture per base per model) +
#         out_agg/summary_agg.json.  Submit: sbatch run_aggregate.sbatch
set -uo pipefail

module load gcc opencv
source .venv/bin/activate
export VLLM_MAX_LEN=8192 VLLM_TP=2

GEN=$HOME/projects/aip-rgrosse/furkanbd/model-weights/gemma-4-31B-it
JUDGE=$GEN
SYNTH=out_synth
AGG=out_agg
TAGS=(qwen3.5-27b            llama3.1-8b                            gemma3-12b)
PATHS=(/model-weights/Qwen3.5-27B  /model-weights/Meta-Llama-3.1-8B-Instruct  /model-weights/gemma-3-12b-it)

# 1) generate variants (skip if already present so a re-run reuses the same stories)
if ls "$SYNTH"/*.variants.json >/dev/null 2>&1; then
  echo "### GENERATE skipped ($SYNTH already has variants) ###"
else
  echo "### GENERATE (Gemma-4-31B) ###"
  python run.py --mode generate --backend vllm --model "$GEN" \
    --examples-file bases.txt --n-variants 6 --outdir "$SYNTH"
fi

mkdir -p "$AGG"; rm -f "$AGG"/*.varsgraphs.json "$AGG"/*.aggregated.json "$AGG"/summary_agg*.json

# 2) extract a graph from each variant, one model per process
for i in "${!TAGS[@]}"; do
  echo "### AGG-EXTRACT ${TAGS[$i]} ###"
  python run.py --mode agg-extract --backend vllm --model "${PATHS[$i]}" \
    --tag "${TAGS[$i]}" --synth-dir "$SYNTH" --outdir "$AGG"
done

# 3) canonicalize + aggregate + score + render, judged by the strong model
echo "### AGG-COMBINE (Gemma-4-31B judge) ###"
python run.py --mode agg-combine --backend vllm --model "$JUDGE" \
  --outdir "$AGG" --agg-min-count 2

echo "### RESULTS (mean over bases, per model) ###"
cat "$AGG/summary_agg.json"
echo
echo "True-vs-estimated pictures: $AGG/*.comparison.html"
