#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=bench-rejudge
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=bench-rejudge-%j.out
# Judge every finished run of several grids again, at every cut, with one Gemma 4 31B judge server (cached answers
# are reused; only new questions are asked, e.g. the world-graph polarity question), then pack each grid.
#   sbatch --exclude=kn133,kn036,kn101 --export=ALL,GRIDS="grid_v8 grid_v8c grid_v9 grid_v8_or" bench_rejudge.sbatch
set -uo pipefail
source .venv/bin/activate
GRIDS=${GRIDS:-grid_v8}; CUTS=0,5,10,15,20,25,30
OUT=_rejudge; mkdir -p $OUT
source bench_lib.sh
start_llm gemma 32768
export JUDGE_URL=$LLM_URL JUDGE=llm JUDGE_SLUG=gemma-4-31b-it JUDGE_ONLY=1
for GRID in $GRIDS; do
  for R in $GRID/*; do                                 # as in bench_final.sbatch: every finished run and aggregation
    case $(basename "$R") in _*) continue;; esac
    [ -d "$R" ] || continue
    [ -f "$R/abstraction.json" ] || ls "$R"/*__gemma-kev.chain.json >/dev/null 2>&1 || continue
    if [ -f "$R/.done_groups" ] && [ ! -f "$R/.done_connect" ]; then echo "--- skip $R (unfinished)"; continue; fi
    if [ ! -f "$R/abstraction.json" ] && [ ! -f "$R/.done_story_graphs" ]; then echo "--- skip $R (story graphs not finished)"; continue; fi
    name=$(basename "$R" | sed 's/^agg__[^_]*__//'); world=$(echo "$name" | sed 's/__.*//')
    echo "--- $R ($world)"
    python benchmark/evaluate_run.py "$world" "$R" $CUTS && touch "$R/.judged"
  done
  GRID=$GRID bash bench_pack.sh | tail -1
done
stop_llm
echo "########## rejudged: $GRIDS"
