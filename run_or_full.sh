#!/bin/bash
# One generator through OpenRouter, every method, one world (laptop): Causal Stories (the model also answers the
# decision questions, as Gemma/Qwen on the cluster: DEC=llm, grouping procedure with the tie question), then
# Per-story prompting (C1), Causal Map (C5, + its clustering) and Whole-corpus prompting (C6) on the same nodes.
# One provider for every call (OR_PROVIDER). Judge: on the cluster (bench_rejudge.sbatch), then score on the laptop.
# Usage: MODEL=deepseek/deepseek-v4-flash TAG=deepseek OR_PROVIDER=Wafer ./run_or_full.sh homelessness [grid_or]
set -euo pipefail
cd "$(dirname "$0")"
set -a; [ -f .env ] && . ./.env; set +a
: "${OPENROUTER_API_KEY:?missing in .env}" "${MODEL:?}" "${TAG:?}" "${OR_PROVIDER:?}"
W=$1; GRID=${2:-grid_or}; PY=.venv/bin/python; K=${K:-8}; WK=${WORKERS:-8}
export OR_PROVIDER MAX_EVENTS=$K KEV_MODEL="llm:$MODEL"; unset LLM_URL
mkdir -p $GRID; R=$GRID/${W}__${TAG}__llm; mkdir -p "$R"
TX=${W}_v8.json
if [ -n "${N:-}" ]; then                                   # N=2: a quick test on the first N stories
  $PY -c "import json; json.dump(json.load(open('$TX'))[:$N], open('$GRID/${W}_texts_$N.json','w'))"; TX=$GRID/${W}_texts_$N.json; fi
step() { local d=$1 s=$2; shift 2; [ -f "$d/.done_$s" ] && { echo "--- skip $d $s"; return; }; echo "### $d $s"; local t=$(date +%s)
         INFER_LOG="$PWD/$d/inference.jsonl" INFER_STEP=$s "$@" && touch "$d/.done_$s" && printf "%s\t%s\n" $s $(( $(date +%s) - t )) >> "$d/timing.tsv"; }
G="$R/*__gemma-kev.chain.json"; KEV=http://127.0.0.1:1      # unused: llm:<model> answers through OpenRouter
step $R nodes         $PY run.py --mode chain-nodes --backend openrouter --model "$MODEL" --texts-file $TX --max-nodes $K --outdir $R --workers $WK
step $R chain_walk    $PY run.py --mode chain-kev --kev-url $KEV --nodes-file $R/chain_nodes.json --tag gemma-kev --outdir $R --workers $WK
step $R groups        $PY run.py --mode merge-kev --kev-url $KEV --graphs-glob "$G" --outdir $R
step $R names         $PY run.py --mode name-groups --backend openrouter --model "$MODEL" --outdir $R
step $R node-happened $PY run.py --mode node-happened --kev-url $KEV --graphs-glob "$G" --outdir $R --workers $WK
step $R implied       $PY run.py --mode implied --kev-url $KEV --graphs-glob "$G" --outdir $R --workers $WK
step $R connect       $PY run.py --mode connect --kev-url $KEV --graphs-glob "$G" --outdir $R --workers $WK
step $R matrix        $PY run.py --mode matrix --graphs-glob "$G" --outdir $R
export LLM_URL=https://openrouter.ai/api/v1/chat/completions LLM_MODEL=$MODEL
for M in C1 C5 C6; do
  C=$GRID/${W}__${TAG}__${M}-llm; mkdir -p $C
  step $C story_graphs $PY benchmark/baselines.py $M $TX $C --nodes $R/chain_nodes.json --workers $WK
done
A=$GRID/agg__C5agg__${W}__${TAG}__C5-llm
[ -f $A/network.json ] || $PY benchmark/alt_aggregate.py C5agg $GRID/${W}__${TAG}__C5-llm $A
echo "done: $GRID ($W, $TAG); cost so far: $(cat $R/cost*.tsv 2>/dev/null | awk '{s+=$NF} END {print s}')"
