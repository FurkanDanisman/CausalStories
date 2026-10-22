#!/bin/bash
# Full pipeline on hosted APIs, from a laptop (no cluster): Gemma 4 31B via OpenRouter,
# Jev via TypeSafe's System One API. Keys in pipeline/.env (not in git):
#   OPENROUTER_API_KEY=...
#   TYPESAFE_API_KEY=...
# Usage:  ./run_api.sh world10.json out_api_world10        (texts file, output folder)
set -euo pipefail
cd "$(dirname "$0")"
set -a; [ -f .env ] && . ./.env; set +a
: "${OPENROUTER_API_KEY:?missing in .env}" "${TYPESAFE_API_KEY:?missing in .env}"
export KEV_MODEL=${KEV_MODEL:-jev-latest}
TEXTS=${1:-world10.json}
OUT=${2:-out_api}
JEV=https://api.typesafe.ai
W=${WORKERS:-8}
PY=.venv/bin/python
mkdir -p "$OUT"
start=$(date +%s)
echo "### Gemma: events + participants";   $PY run.py --mode chain-nodes --backend openrouter --texts-file "$TEXTS" --max-nodes 5 --outdir "$OUT" --workers $W
echo "### Jev: chain walk";                 $PY run.py --mode chain-kev --kev-url $JEV --nodes-file "$OUT/chain_nodes.json" --tag gemma-kev --outdir "$OUT" --workers $W
echo "### Jev: groups (Steps 1-2)";         $PY run.py --mode merge-kev --kev-url $JEV --graphs-glob "$OUT/*__gemma-kev.chain.json" --outdir "$OUT"
echo "### Gemma: high-level names";         $PY run.py --mode name-groups --backend openrouter --outdir "$OUT"
echo "### Jev: did it happen? (Step 3)";    $PY run.py --mode implied --kev-url $JEV --graphs-glob "$OUT/*__gemma-kev.chain.json" --outdir "$OUT" --workers $W
echo "### Steps 4-5: MICE x30, meta-analysis, network"
$PY -c "import sklearn" 2>/dev/null || .venv/bin/pip install -q scikit-learn
$PY run.py --mode combine --graphs-glob "$OUT/*__gemma-kev.chain.json" --outdir "$OUT"
echo "Done in $(( $(date +%s) - start )) s. Results in $OUT/"
