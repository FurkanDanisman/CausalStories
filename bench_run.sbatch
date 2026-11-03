#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=bench-run
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=bench-run-%x-%j.out
#
# Every GPU stage for ONE generator (GEN) and ONE decision model (DEC), over all worlds and methods, with each model
# loaded once per phase (method_Oct.tex; benchmark/workplan.md):
#   phase 1  GEN   our node extraction; C1-C6 story graphs (C2-C4 on our nodes)
#   phase 2  DEC   our chain walk + influence walk; groups (merge-kev) for ours and C1-C4
#   phase 3  GEN   group names for ours and C1-C4
#   phase 4  DEC   did it happen (named nodes, missing groups) and connect, for ours and C1-C4
# C5's clustering, Steps 5-6 (run.py --mode matrix) and scoring run on the laptop.
# Resumable: a finished step leaves a .done_<step> file and is skipped when the job is submitted again.
# Times: every step of every run -> <run>/timing.tsv; every model load and the whole job -> grid_cluster/_logs/<GEN>__<DEC>/model_loads.tsv.
#   sbatch --export=ALL,GEN=gemma,DEC=kev27b bench_run.sbatch
#   GEN: gemma | qwen | llama | gptoss    DEC: kev27b | kev9b
#   METHODS: default "ours C1 C2 C3 C4 C5 C6" with kev27b and "ours" with kev9b (the competing approaches use Kev-27B)
#   WORLDS:  default "homelessness career mental_health"     SMOKE=1: first 2 stories of each world
set -uo pipefail
GEN=${GEN:?GEN}; DEC=${DEC:-kev27b}
WORLDS=${WORLDS:-"homelessness career mental_health"}
[ "$DEC" = kev27b ] && METHODS=${METHODS:-"ours C1 C2 C3 C4 C5 C6"} || METHODS=${METHODS:-"ours"}
OUT=grid_cluster/_logs/${GEN}__${DEC}${SMOKE:+_smoke}
mkdir -p "$OUT"
source bench_lib.sh
T0=$(date +%s); LOGS=$OUT                  # LOGS keeps the job folder (OUT is reused per run below)

run_dir() { [ "$2" = ours ] && echo "grid_cluster/${1}__${GEN}__${DEC}${SMOKE:+_smoke}" || echo "grid_cluster/${1}__${GEN}__${2}${SMOKE:+_smoke}"; }
texts() {                                   # texts file of a world (first 2 stories with SMOKE=1)
  local f=${1}_v3.json
  if [ -n "${SMOKE:-}" ]; then f=$OUT/${1}_texts.json
    python -c "import json;json.dump(json.load(open('${1}_v3.json'))[:2],open('$f','w'))"; fi
  echo "$f"
}
step() {                                    # step <run dir> <name> <command...>: skip if done; time it
  local d=$1 s=$2; shift 2
  [ -f "$d/.done_$s" ] && { echo "--- skip $d $s (done)"; return 0; }
  echo "--- $d $s"; local t=$(date +%s)
  "$@" && { touch "$d/.done_$s"; printf "%s\t%s\n" "$s" $(( $(date +%s) - t )) >> "$d/timing.tsv"; } \
       || echo "!!! FAILED $d $s"
}
graphs() { echo "$1/*__gemma-kev.chain.json"; }
agg_runs() { for W in $WORLDS; do for M in $METHODS; do [[ "$M" =~ ^(ours|C1|C2|C3|C4)$ ]] && run_dir $W $M; done; done; }

echo "########## phase 1: $GEN (nodes, competing approaches' story graphs)"
start_llm "$GEN" 65536
for W in $WORLDS; do
  TX=$(texts $W); R=$(run_dir $W ours); mkdir -p "$R"
  step "$R" nodes python run.py --mode chain-nodes --backend openai --base-url "http://127.0.0.1:$LLM_PORT/v1" --model llm \
    --texts-file "$TX" --max-nodes 8 --outdir "$R" --workers 16
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
  [[ " $METHODS " == *" ours "* ]] && step "$R" chain_walk python run.py --mode chain-kev --kev-url "http://127.0.0.1:$KEV_PORT" \
    --nodes-file "$R/chain_nodes.json" --tag gemma-kev --outdir "$R"
done
for R in $(agg_runs); do
  step "$R" groups python run.py --mode merge-kev --kev-url "http://127.0.0.1:$KEV_PORT" --graphs-glob "$(graphs $R)" --outdir "$R"
done
stop_kev

echo "########## phase 3: $GEN (group names)"
start_llm "$GEN"
for R in $(agg_runs); do
  step "$R" names python run.py --mode name-groups --backend openai --base-url "http://127.0.0.1:$LLM_PORT/v1" --model llm --outdir "$R"
done
stop_llm

echo "########## phase 4: $DEC (did it happen, connect)"
start_kev
for R in $(agg_runs); do
  for M in node-happened implied connect; do
    step "$R" $M python run.py --mode $M --kev-url "http://127.0.0.1:$KEV_PORT" --graphs-glob "$(graphs $R)" --outdir "$R"
  done
done
stop_kev

for W in $WORLDS; do for M in $METHODS; do R=$(run_dir $W $M); [ -d "$R" ] || continue
  WORLD=$W; OUT=$R; run_info "${MODELS[$GEN]}" "$( [ $M = ours ] && echo "${KEVS[$DEC]}" || echo "$M + ${KEVS[$DEC]}" )" \
    "$(awk -F'\t' '{s+=$2} END {print s+0}' "$R/timing.tsv" 2>/dev/null)"
done; done
printf "job\t%s\t%s\n" "${GEN}__${DEC}" $(( $(date +%s) - T0 )) >> "$LOGS/model_loads.tsv"
echo "########## done in $(( $(date +%s) - T0 )) s (failed steps: !!! FAILED above)"
echo "times: each run's timing.tsv (its own steps, start to end); model loads + whole job: $LOGS/model_loads.tsv"
