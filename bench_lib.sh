#!/bin/bash
# Shared helpers for the benchmark jobs (bench_*.sbatch). One model on the 2 GPUs at a time:
# an LLM through a vLLM OpenAI-compatible server (tensor-parallel over both GPUs), or Kev-27B through
# kev_serve_2gpu.py. Source this file from a job script.

module load gcc opencv
source .venv/bin/activate
BASE=$HOME/projects/aip-rgrosse/furkanbd
export HF_HOME=$BASE/hf_cache
export HF_HUB_OFFLINE=1
KEV_DIR=$BASE/kev
PIPE=$PWD
KEV_PORT=8009
LLM_PORT=8000
export LLM_URL=http://127.0.0.1:$LLM_PORT/v1/chat/completions
export LLM_MODEL=llm                      # served model name used by baselines.py / the judge
declare -A MODELS=(
  [gemma]=$BASE/model-weights/gemma-4-31B-it
  [qwen]=/model-weights/Qwen3.5-27B
  [llama]=/model-weights/Meta-Llama-3.1-8B-Instruct
  [gptoss]=/model-weights/gpt-oss-20b
)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_COUNT=$(nvidia-smi -L | wc -l)

start_llm() {                             # start_llm <gemma|qwen|llama|gptoss> [max_len]
  local path=${MODELS[$1]} len=${2:-32768}
  python -m vllm.entrypoints.openai.api_server --model "$path" --served-model-name llm \
    --tensor-parallel-size 2 --max-model-len "$len" --port $LLM_PORT > "$OUT/llm_server_$1.log" 2>&1 &
  LLM_PID=$!
  for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$LLM_PORT/v1/models" >/dev/null && return 0
    kill -0 $LLM_PID 2>/dev/null || { echo "LLM server died:"; tail -40 "$OUT/llm_server_$1.log"; exit 1; }
    sleep 10
  done
  echo "LLM server did not start"; exit 1
}
stop_llm() { kill $LLM_PID 2>/dev/null; wait $LLM_PID 2>/dev/null; sleep 5; }

start_kev() {
  (cd "$KEV_DIR" && .venv/bin/python "$PIPE/kev_serve_2gpu.py" --run jaredpalmer/kev-27b --port $KEV_PORT) >> "$OUT/kev_server.log" 2>&1 &
  KEV_PID=$!
  for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$KEV_PORT/v1/models" >/dev/null && return 0
    kill -0 $KEV_PID 2>/dev/null || { echo "Kev server died:"; tail -40 "$OUT/kev_server.log"; exit 1; }
    sleep 10
  done
  echo "Kev server did not start"; exit 1
}
stop_kev() { kill $KEV_PID 2>/dev/null; wait $KEV_PID 2>/dev/null; sleep 5; }

# GPU part of our aggregation on the story graphs in $OUT (method_Oct.tex, Section 2): Kev groups, the LLM
# names the groups, Kev decides what happened (named nodes, then missing groups) and connects the inferred
# groups on top of each story graph. Usage and the averaged matrix (Steps 5-6) are computed on the laptop
# (Killarney is GPU-only). The generator for naming is $1.
our_aggregation() {
  start_kev
  python run.py --mode merge-kev --kev-url "http://127.0.0.1:$KEV_PORT" --graphs-glob "$OUT/*__gemma-kev.chain.json" --outdir "$OUT"
  stop_kev
  start_llm "$1"
  python run.py --mode name-groups --backend openai --base-url "http://127.0.0.1:$LLM_PORT/v1" --model llm --outdir "$OUT"
  stop_llm
  start_kev
  python run.py --mode node-happened --kev-url "http://127.0.0.1:$KEV_PORT" --graphs-glob "$OUT/*__gemma-kev.chain.json" --outdir "$OUT"
  python run.py --mode implied --kev-url "http://127.0.0.1:$KEV_PORT" --graphs-glob "$OUT/*__gemma-kev.chain.json" --outdir "$OUT"
  python run.py --mode connect --kev-url "http://127.0.0.1:$KEV_PORT" --graphs-glob "$OUT/*__gemma-kev.chain.json" --outdir "$OUT"
  stop_kev
}

run_info() {                              # run_info <generator> <decision/method> <seconds>
  python - "$@" <<PY
import json, sys, time
g, d, s = sys.argv[1], sys.argv[2], float(sys.argv[3])
json.dump({"world": "$WORLD", "generator": g, "decision_model": d, "seconds": s, "seconds_gpu": s, "usd_total": 0,
           "note": "seconds = GPU stages on Killarney; the local Steps 5-6 time is added when they run",
           "hardware": "Killarney $GPU_COUNT x $GPU_NAME", "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
          open("$OUT/run_info.json", "w"), indent=1)
PY
}
