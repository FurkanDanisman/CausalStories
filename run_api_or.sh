#!/bin/bash
# Full pipeline on OpenRouter only (one key): Gemma 4 31B for nodes/names, Jev (~typesafe/jev-latest,
# decisions endpoint) for arrows/groups. Key in pipeline/.env: OPENROUTER_API_KEY=...
# Usage:  [GENERATOR=qwen/qwen3.5-27b] [DECISION=jaredpalmer/kev-4b | llm:<model>] ./run_api_or.sh texts.json out_folder
set -euo pipefail
cd "$(dirname "$0")"
set -a; [ -f .env ] && . ./.env; set +a
: "${OPENROUTER_API_KEY:?missing in .env}"
export KEV_MODEL=${DECISION:-~typesafe/jev-latest}   # decision model on OpenRouter (System One contract)
TEXTS=$1; OUT=$2
JEV=https://openrouter.ai/api/alpha/decisions
W=${WORKERS:-8}; K=${MAX_NODES:-8}
GEN=${GENERATOR:-google/gemma-4-31b-it}     # generator LLM on OpenRouter
PY=.venv/bin/python
mkdir -p "$OUT"
start=$(date +%s)
echo "### Gemma: events + participants (K=$K)"; $PY run.py --mode chain-nodes --backend openrouter --model "$GEN" --texts-file "$TEXTS" --max-nodes $K --outdir "$OUT" --workers $W
echo "### Jev: chain walk";                 $PY run.py --mode chain-kev --kev-url $JEV --nodes-file "$OUT/chain_nodes.json" --tag gemma-kev --outdir "$OUT" --workers $W
echo "### Jev: groups (Steps 1-2)";         $PY run.py --mode merge-kev --kev-url $JEV --graphs-glob "$OUT/*__gemma-kev.chain.json" --outdir "$OUT"
echo "### Gemma: high-level names";         $PY run.py --mode name-groups --backend openrouter --model "$GEN" --outdir "$OUT"
echo "### Jev: named nodes happened? (Step 3)"; $PY run.py --mode node-happened --kev-url $JEV --graphs-glob "$OUT/*__gemma-kev.chain.json" --outdir "$OUT" --workers $W
echo "### Jev: missing groups happened? (Step 3)"; $PY run.py --mode implied --kev-url $JEV --graphs-glob "$OUT/*__gemma-kev.chain.json" --outdir "$OUT" --workers $W
echo "### Jev: connect inferred groups (Step 4)"; $PY run.py --mode connect --kev-url $JEV --graphs-glob "$OUT/*__gemma-kev.chain.json" --outdir "$OUT" --workers $W
echo "Done in $(( $(date +%s) - start )) s. Results in $OUT/"
