#!/bin/bash
# Causal Stories with a decision model on OpenRouter (Decisions API), starting from existing extracted nodes, so only
# the decision model changes. Generator for naming: Gemma 4 31B on OpenRouter. Key in pipeline/.env.
# Usage: DECISION=google/gemini-decision-1 ./run_or_decisions.sh <chain_nodes.json> <out_folder> [N stories]
# Nodes: the newest Gemma run (grid_v8c, current method). The old run_api_or.sh is not used.
set -euo pipefail
cd "$(dirname "$0")"
set -a; [ -f .env ] && . ./.env; set +a
: "${OPENROUTER_API_KEY:?missing in .env}"
: "${DECISION:?set DECISION=<openrouter decision model>}"
export KEV_MODEL=$DECISION
NODES=$1; OUT=$2; N=${3:-}
API=https://openrouter.ai/api/alpha/decisions
W=${WORKERS:-8}; GEN=${GENERATOR:-google/gemma-4-31b-it}; PY=.venv/bin/python
mkdir -p "$OUT"
if [ ! -f "$OUT/chain_nodes.json" ]; then
  $PY -c "import json,sys; d=json.load(open('$NODES')); n='$N'; json.dump(d[:int(n)] if n else d, open('$OUT/chain_nodes.json','w'))"
fi
step() { local s=$1; shift; [ -f "$OUT/.done_$s" ] && { echo "--- skip $s"; return; }; echo "### $s"; local t=$(date +%s)
         INFER_LOG="$PWD/$OUT/inference.jsonl" INFER_STEP=$s "$@" && touch "$OUT/.done_$s" && printf "%s\t%s\n" $s $(( $(date +%s) - t )) >> "$OUT/timing.tsv"; }
G="$OUT/*__gemma-kev.chain.json"
step chain_walk    $PY run.py --mode chain-kev --kev-url $API --nodes-file "$OUT/chain_nodes.json" --tag gemma-kev --outdir "$OUT" --workers $W
step groups        $PY run.py --mode merge-kev --kev-url $API --graphs-glob "$G" --outdir "$OUT"
step names         $PY run.py --mode name-groups --backend openrouter --model "$GEN" --outdir "$OUT"
step node-happened $PY run.py --mode node-happened --kev-url $API --graphs-glob "$G" --outdir "$OUT" --workers $W
step implied       $PY run.py --mode implied --kev-url $API --graphs-glob "$G" --outdir "$OUT" --workers $W
step connect       $PY run.py --mode connect --kev-url $API --graphs-glob "$G" --outdir "$OUT" --workers $W
step matrix        $PY run.py --mode matrix --graphs-glob "$G" --outdir "$OUT"
echo "done: $OUT"
