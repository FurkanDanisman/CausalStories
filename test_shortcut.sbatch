#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=test-shortcut
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=test-shortcut-%j.out
# 10 shortcut questions (A -> C where the story has A -> B -> C; Llama-70B answered enables in v8), each with the
# current direct-only sentence and the stronger one, asked to our vLLM server as in the benchmark.
#   sbatch --exclude=kn133,kn036,kn101 [--export=ALL,GEN=gemma] test_shortcut.sbatch
set -uo pipefail
source .venv/bin/activate
OUT=_test_shortcut; mkdir -p $OUT
source bench_lib.sh
GEN=${GEN:-llama70}
start_llm $GEN 32768
python benchmark/test_shortcut_wording.py
stop_llm
