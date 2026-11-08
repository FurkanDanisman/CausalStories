#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=bench-run
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=bench-run-%x-%j.out
#SBATCH --signal=B:USR1@600
#
# Every GPU stage for ONE generator (GEN) and ONE decision model (DEC), over all worlds and methods, with each model
# loaded once per phase (method_Oct.tex; benchmark/workplan.md):
#   phase 1  GEN   our node extraction; C1-C6 story graphs (C2-C4 on our nodes)
#   phase 2  DEC   our chain walk + influence walk; groups (merge-kev) for ours and C1-C4
#   phase 3  GEN   group names for ours and C1-C4
#   phase 4  DEC   did it happen (named nodes, missing groups) and connect, for ours and C1-C4
# C5's clustering, Steps 5-6 (run.py --mode matrix) and scoring run on the laptop.
# Resumable: a finished step leaves a .done_<step> file and is skipped when the job is submitted again.
# Times: every step of every run -> <run>/timing.tsv; every model load and the whole job -> grid_cluster/_logs/<GEN>__<DEC>__<worlds>/model_loads.tsv.
#   sbatch --export=ALL,GEN=gemma,KEV_WORKERS=32 bench_run.sbatch       (DEC=kev9b by default)
#   GEN: gemma | qwen | llama | gptoss    DEC: kev9b (default) | kev27b
#   METHODS: default "ours C1 C2 C3 C4 C5 C6" (the competing approaches use the same decision model as ours)
#   WORLDS:  default "homelessness career mental_health"     SMOKE=1: first 2 stories; N=20: first 20 (timing test)
# Requeue: 10 min before the time limit the job submits itself again (same settings) and stops; the next job skips
#   finished steps. It stops requeueing after MAX_ROUNDS (default 6) rounds, or when a round finished no step.
#   DEC=llm: the generator LLM also answers the decision questions (no decision model); GROUPING=llm: the LLM groups
#   (batches of 10 stories, merged in a tournament). Run folders: <world>__<gen>__<dec><-llmgroup>.
#   KEV_WORKERS: stories sent to the Kev servers at once (default 32) for the chain walk, did it happen and connect
set -uo pipefail
GEN=${GEN:?GEN}; DEC=${DEC:-kev9b}
GROUPING=${GROUPING:-decision}                # Step 1 grouping: decision (the decision model) | llm (pipeline/llm_group.py)
VARIANT=""; [ "$GROUPING" = llm ] && VARIANT="-llmgroup"      # run folders of the LLM-grouping setting
WORLDS=${WORLDS:-"homelessness career mental_health"}
WORLDS=${WORLDS//+/ }
METHODS=${METHODS:-"ours C1 C2 C3 C4 C5 C6"}
METHODS=${METHODS//+/ }                    # METHODS=ours+C1 on the sbatch line (no spaces inside --export)
N=${N:-}; [ -n "${SMOKE:-}" ] && N=2
TAG=${N:+_n$N}; KW=${KEV_WORKERS:-32}
OUT=grid_cluster/_logs/${GEN}__${DEC}${VARIANT}__${WORLDS// /+}${TAG}     # one log folder per job (jobs for different worlds run side by side)
mkdir -p "$OUT"
source bench_lib.sh
K=${K:-8}; export MAX_EVENTS=$K              # event budget K: our Prompt 1a and C1, C5, C6 (same for every method)
T0=$(date +%s); LOGS=$OUT                  # LOGS keeps the job folder (OUT is reused per run below)
ROUND=${ROUND:-1}; MAX_ROUNDS=${MAX_ROUNDS:-6}
n_done() { ls -a grid_cluster/*__${GEN}__* 2>/dev/null | grep -c '^\.done_\|__connect\.json$'; }   # finished steps + connected stories
DONE0=$(n_done)
FAILS=0
resubmit() {                                # resubmit <reason>: the same job again (next round skips finished steps)
  local now=$(n_done)
  if [ "$ROUND" -ge "$MAX_ROUNDS" ]; then echo "########## $1: round $ROUND of $MAX_ROUNDS, not requeueing"
  elif [ "$now" -le "$DONE0" ]; then echo "########## $1: no step finished this round, not requeueing (a step that always fails or takes longer than the limit?)"
  else
    echo "########## $1: requeueing as round $(( ROUND + 1 )) ($(( now - DONE0 )) steps finished this round)"
    (cd "$SLURM_SUBMIT_DIR" && env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /') sbatch \
      --export=ALL,GEN=$GEN,DEC=$DEC,GROUPING=$GROUPING,BATCH_SIZE=${BATCH_SIZE:-10},WORLDS=${WORLDS// /+},METHODS=${METHODS// /+},ROUND=$(( ROUND + 1 )),MAX_ROUNDS=$MAX_ROUNDS,KEV_WORKERS=$KW,K=$K${N:+,N=$N} \
      bench_run.sbatch)
  fi
}
requeue() { trap - USR1; resubmit "time limit"; kill $LLM_PID $KEV_PIDS 2>/dev/null; exit 0; }   # USR1, 10 min before the limit
trap requeue USR1

run_dir() {                                 # competing approaches keep their v4 folders unless the setting differs
  local v=""; { [ "$DEC" != kev9b ] || [ "$GROUPING" != decision ]; } && v="-${DEC}${VARIANT}"
  [ "$2" = ours ] && echo "grid_cluster/${1}__${GEN}__${DEC}${VARIANT}${TAG}" || echo "grid_cluster/${1}__${GEN}__${2}${v}${TAG}"; }
texts() {                                   # texts file of a world (first 2 stories with SMOKE=1)
  local f=${1}_v3.json
  if [ -n "$N" ]; then f=$OUT/${1}_texts.json
    python -c "import json;json.dump(json.load(open('${1}_v3.json'))[:$N],open('$f','w'))"; fi
  echo "$f"
}
step() {                                    # step <run dir> <name> <command...>: skip if done; time it
  local d=$1 s=$2; shift 2
  [ -f "$d/.done_$s" ] && { echo "--- skip $d $s (done)"; return 0; }
  echo "--- $d $s"; local t=$(date +%s)
  INFER_LOG="$PWD/$d/inference.jsonl" INFER_STEP="$s" "$@" & wait $! && { touch "$d/.done_$s"; printf "%s\t%s\n" "$s" $(( $(date +%s) - t )) >> "$d/timing.tsv"; } \
       || { echo "!!! FAILED $d $s"; FAILS=$(( FAILS + 1 )); }
}
graphs() { echo "$1/*__gemma-kev.chain.json"; }
agg_runs() { for W in $WORLDS; do [[ " $METHODS " == *" ours "* ]] && run_dir $W ours; done; }   # C1-C4: individual graphs only

echo "########## phase 1: $GEN (nodes, competing approaches' story graphs)"
start_llm "$GEN" 65536
for W in $WORLDS; do
  TX=$(texts $W); R=$(run_dir $W ours); mkdir -p "$R"
  step "$R" nodes python run.py --mode chain-nodes --backend openai --base-url "http://127.0.0.1:$LLM_PORT/v1" --model llm \
    --texts-file "$TX" --max-nodes $K --outdir "$R" --workers 16
  for M in $METHODS; do
    [ "$M" = ours ] && continue
    C=$(run_dir $W $M); mkdir -p "$C"
    step "$C" story_graphs python benchmark/baselines.py $M "$TX" "$C" --nodes "$R/chain_nodes.json" --workers 16
  done
done
stop_llm

echo "########## phase 2: $DEC (chain walk, groups)"
start_kev
for W in $WORLDS; do
  R=$(run_dir $W ours)
  [[ " $METHODS " == *" ours "* ]] && step "$R" chain_walk python run.py --mode chain-kev --kev-url "$KEV_URL" \
    --nodes-file "$R/chain_nodes.json" --tag gemma-kev --outdir "$R" --workers $KW
done
[ "$GROUPING" = decision ] && for R in $(agg_runs); do
  step "$R" groups python run.py --mode merge-kev --kev-url "$KEV_URL" --graphs-glob "$(graphs $R)" --outdir "$R"
done
stop_kev

echo "########## phase 3: $GEN (group names; groups too with GROUPING=llm)"
start_llm "$GEN" 65536
for R in $(agg_runs); do
  [ "$GROUPING" = llm ] && step "$R" groups python run.py --mode merge-llm --batch-size ${BATCH_SIZE:-10} \
    --graphs-glob "$(graphs $R)" --outdir "$R"
  step "$R" names python run.py --mode name-groups --backend openai --base-url "http://127.0.0.1:$LLM_PORT/v1" --model llm --outdir "$R"
done
stop_llm

echo "########## phase 4: $DEC (did it happen, connect)"
start_kev
for R in $(agg_runs); do
  for M in node-happened implied connect; do
    step "$R" $M python run.py --mode $M --kev-url "$KEV_URL" --graphs-glob "$(graphs $R)" --outdir "$R" --workers $KW
  done
done
stop_kev

for W in $WORLDS; do for M in $METHODS; do R=$(run_dir $W $M); [ -d "$R" ] || continue
  WORLD=$W; OUT=$R; run_info "${MODELS[$GEN]}" "$( [ $M = ours ] && echo "${KEVS[$DEC]}" || echo "$M + ${KEVS[$DEC]}" )" \
    "$(awk -F'\t' '{s+=$2} END {print s+0}' "$R/timing.tsv" 2>/dev/null)"
done; done
printf "job\t%s\t%s\n" "${GEN}__${DEC}" $(( $(date +%s) - T0 )) >> "$LOGS/model_loads.tsv"
echo "########## done in $(( $(date +%s) - T0 )) s ($FAILS failed steps: !!! FAILED above)"
[ "$FAILS" -gt 0 ] && resubmit "$FAILS failed steps"
echo "times: each run's timing.tsv (its own steps, start to end); model loads + whole job: $LOGS/model_loads.tsv"
