#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=causal-generate
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=generate-%j.out
#
# Generate 6 synthetic retellings for each base in bases.txt (simulating different
# narrators of the same events), for the aggregation experiment. Submit:
#   sbatch run_generate.sbatch
# We generate with a STRONG model that is NOT one of the extraction test models,
# to avoid generator/extractor shared blind spots (uses the Gemma-4-31B judge model
# as the generator here).
set -uo pipefail

module load gcc opencv
source .venv/bin/activate
export VLLM_MAX_LEN=8192 VLLM_TP=2

GEN=$HOME/projects/aip-rgrosse/furkanbd/model-weights/gemma-4-31B-it

python run.py --mode generate --backend vllm --model "$GEN" \
  --examples-file bases.txt --n-variants 6 --outdir out_synth

echo "############ story-sets ############"
for f in out_synth/*.variants.json; do echo "== $f =="; python -c "import json,sys; d=json.load(open('$f')); print('BASE:', d['base_text']); [print(f\"  v{i}: {v['text']}\") for i,v in enumerate(d['variants'])]"; echo; done
