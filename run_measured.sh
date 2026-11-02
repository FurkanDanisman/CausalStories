#!/bin/bash
# Run the OpenRouter pipeline for one decision model and record wall time and $ spent (key usage delta).
# Usage: ./run_measured.sh <decision-model> <texts.json> <out_folder>
set -euo pipefail
cd "$(dirname "$0")"
set -a; . ./.env; set +a
usage() { curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" | .venv/bin/python -c "import json,sys;print(json.load(sys.stdin)['data']['usage'])"; }
u0=$(usage); t0=$(date +%s)
DECISION=$1 ./run_api_or.sh "$2" "$3" > "$3.log" 2>&1
t1=$(date +%s); sleep 20; u1=$(usage)
.venv/bin/python -c "import json;json.dump({'decision_model':'$1','generator':'google/gemma-4-31b-it','seconds':$t1-$t0,'usd':round($u1-$u0,4),'hardware':'hosted (OpenRouter)'},open('$3/run_info.json','w'),indent=1)"
cat "$3/run_info.json"
