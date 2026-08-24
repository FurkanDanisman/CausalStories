#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=causal-compare
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=compare-%j.out
#
# Submit from the repo root:   sbatch run_compare.sbatch
# Watch:                       squeue --me   ;   tail -f compare-<jobid>.out
#
# Runs entirely on the allocated GPU node (2x L40S). No interactive shell, no
# risk of anything executing on the login node.
set -euo pipefail

module load gcc opencv
source .venv/bin/activate

export VLLM_MAX_LEN=8192
export VLLM_TP=2                 # shard each model across the 2 GPUs

bash compare_models.sh
