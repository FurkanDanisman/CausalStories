#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=causal-world
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=world-%j.out
#
# Qwen-only world experiment: extract a graph per narrative, canonicalize names
# across them, and build the event x person table with a refutation pass. No judge.
# MICE + the 4 plots run locally afterwards (python world_plots.py). Submit:
#   sbatch run_world.sbatch
set -uo pipefail

module load gcc opencv
source .venv/bin/activate
export VLLM_MAX_LEN=8192 VLLM_TP=2

python run.py --mode world-extract --backend vllm --model /model-weights/Qwen3.5-27B \
  --world-file world_homelessness.variants.json --outdir out_w

echo "Done. Pull out_w/ to your Mac and run:  python world_plots.py out_w"
