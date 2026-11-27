#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=test-arrow
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=test-arrow-%j.out
# One arrow question to our vLLM Gemma 4 31B (as in the benchmark): ho002, eviction -> being homeless,
# 5 times with the current definition and 5 times with Regan et al.'s two "Without effect" rows.
#   sbatch --exclude=kn133,kn036,kn101 [--export=ALL,GEN=llama70] test_ho002_arrow.sbatch
set -uo pipefail
source .venv/bin/activate
OUT=_test_arrow; mkdir -p $OUT
source bench_lib.sh
GEN=${GEN:-gemma}                          # GEN=llama70: the same test with Llama-3.3-70B
start_llm $GEN 32768
VARIANTS=${VARIANTS:-current,headers} TEST_FREE=${TEST_FREE:-30} python benchmark/test_ho002_arrow.py
stop_llm
