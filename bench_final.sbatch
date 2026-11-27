#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=bench-final
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=bench-final-%j.out
#SBATCH --signal=B:USR1@600
#
# Last job, after every bench_run job has finished:
#   1. A4 (LLM merge, aggregation comparison) with each of the 4 generators, on OUR story graphs and groups of the
#      Gemma + decision-model run of each world (DEC, Kev-9B by default) (the same input as A1-A3 and A5, which run on the laptop);
#   2. A1-A3, A5 (PC / GES) and C5's clustering (no model; quick CPU steps run here so the judge can see them);
#   3. the judge (Gemma 4 31B) matches the nodes of every finished run and stores its answers in each run's judge cache.
# Scores are computed on the laptop from those caches. Resumable: finished A4 runs and judged runs are skipped, and
# runs whose bench_run job is unfinished are left for the next submission, so it can run before every job is done.
#   sbatch bench_final.sbatch      (GENS="gemma qwen" to run a subset)
set -uo pipefail
GRID=${GRID:-grid_v7}; SKIP_ABLATIONS=${SKIP_ABLATIONS:-1}
CUTS=${CUTS:-0,5,10,15,20,25,30}; REJUDGE=${REJUDGE:-}   # REJUDGE=1: judge finished runs again (new cuts; cached answers are reused)   # SKIP_ABLATIONS=1: no A1-A5 (not part of the comparison)
OUT=$GRID/_logs/final
mkdir -p "$OUT"
source bench_lib.sh

# Judge only after this grid's run jobs: a bench_run job resubmits itself under a new id at its time limit, so a
# dependency on the first ids can be satisfied too early. If a run job of this GRID is still queued or running, this
# job submits itself again after those jobs and stops. At its own time limit it submits itself again (judged runs are
# skipped). At the end it packs the results into results_cluster/$GRID.
ROUND=${ROUND:-1}
self_again() { (cd "$SLURM_SUBMIT_DIR" && env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /') sbatch --parsable \
  --exclude=${EXCLUDE:-kn133,kn036} "$@" \
  --export=ALL,GRID=$GRID,SKIP_ABLATIONS=$SKIP_ABLATIONS,CUTS=$CUTS,REJUDGE=$REJUDGE,ROUND=$(( ROUND + 1 )) bench_final.sbatch); }
WAIT=""; for j in $(squeue --me -h -n bench-run -o %i); do
  s=$(sacct -j $j -n -X -o SubmitLine%2000 2>/dev/null)
  { [ -z "$s" ] || [[ $s == *"GRID=$GRID,"* ]] || [[ $s == *"GRID=$GRID "* ]]; } && WAIT="$WAIT:$j"; done
if [ -n "$WAIT" ]; then
  [ "$ROUND" -ge 20 ] && { echo "########## round $ROUND: not resubmitting"; exit 1; }
  echo "########## run jobs of $GRID still queued or running (${WAIT#:}): judging after them as job $(self_again --dependency=afterany$WAIT)"
  exit 0
fi
trap 'trap - USR1; echo "########## time limit: judging the rest as job $(self_again)"; stop_llm; exit 0' USR1

[ "$SKIP_ABLATIONS" = 1 ] || { echo "########## A4 with each generator"
for G in ${GENS:-gemma qwen llama gptoss}; do
  start_llm "$G" 65536
  for R in $GRID/*__gemma__${DEC}; do
    case $(basename "$R") in agg__*) continue;; esac          # our runs only, not aggregation outputs
    [ -f "$R/abstraction.json" ] || continue
    A=$GRID/agg__A4-${G}__$(basename "$R")
    [ -f "$A/network.json" ] || { t0=$(date +%s); mkdir -p "$A"; INFER_LOG="$PWD/$A/inference.jsonl" INFER_STEP=A4 python benchmark/alt_aggregate.py A4 "$R" "$A" && printf "A4\t%s\n" $(( $(date +%s) - t0 )) >> "$A/timing.tsv"; }
  done
  stop_llm
done; }

echo "########## aggregations without a model (A1-A3, A5 PC/GES, C5 clustering): quick CPU steps, run here so the judge sees them"
for R in $GRID/*__gemma__${DEC}; do
  case $(basename "$R") in agg__*) continue;; esac
  [ -f "$R/abstraction.json" ] || continue
  [ "$SKIP_ABLATIONS" = 1 ] && break
  for M in A1 A2 A3 pc ges; do
    A=$GRID/agg__${M}__$(basename "$R")
    [ -f "$A/network.json" ] || { t0=$(date +%s); python benchmark/alt_aggregate.py $M "$R" "$A" && printf "%s\t%s\n" $M $(( $(date +%s) - t0 )) >> "$A/timing.tsv"; }
  done
done
for R in $GRID/*__*__C5 $GRID/*__*__C5-*; do
  case $(basename "$R") in agg__*) continue;; esac
  [ -d "$R" ] || continue
  A=$GRID/agg__C5agg__$(basename "$R")
  [ -f "$A/network.json" ] || { t0=$(date +%s); python benchmark/alt_aggregate.py C5agg "$R" "$A" && printf "C5agg\t%s\n" $(( $(date +%s) - t0 )) >> "$A/timing.tsv"; }
done

echo "########## judge (Gemma 4 31B): node matching only"
start_llm gemma 32768
export JUDGE_URL=$LLM_URL JUDGE=llm JUDGE_SLUG=gemma-4-31b-it JUDGE_ONLY=1
for R in $GRID/*; do                                   # every run and aggregation output (not _logs, _smoke)
  case $(basename "$R") in _*) continue;; esac
  [ -d "$R" ] || continue
  [ -f "$R/abstraction.json" ] || ls "$R"/*__gemma-kev.chain.json >/dev/null 2>&1 || continue   # C1-C5 story graphs too
  [ -f "$R/.judged" ] && [ -z "$REJUDGE" ] && continue
  if [ -f "$R/.done_groups" ] && [ ! -f "$R/.done_connect" ]; then echo "--- skip $R (its bench_run job has not finished)"; continue; fi
  if [ ! -f "$R/abstraction.json" ] && [ ! -f "$R/.done_story_graphs" ]; then echo "--- skip $R (story graphs not finished)"; continue; fi
  name=$(basename "$R" | sed 's/^agg__[^_]*__//')
  world=$(echo "$name" | sed 's/__.*//')
  echo "--- $R ($world)"
  python benchmark/evaluate_run.py "$world" "$R" $CUTS & wait $! && touch "$R/.judged"   # background + wait: the time-limit signal is handled at once
done
stop_llm
GRID=$GRID bash bench_pack.sh | tail -1                # results_cluster/$GRID (commit + push from the login node)
