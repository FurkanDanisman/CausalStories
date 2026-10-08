#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=kev-2gpu-test
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=kev-2gpu-test-%j.out
#
# Quick test: Kev-27B split over 2 GPUs (kev_serve_2gpu.py), one tiny question sent twice
# (the second time reuses Kev's prefix cache). Needs the Kev setup from run_kev_chain.sbatch.
#   sbatch smoke_kev_2gpu.sbatch
set -uo pipefail
BASE=$HOME/projects/aip-rgrosse/furkanbd
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1
KEV_DIR=$BASE/kev
PIPE=$PWD
PORT=8009
nvidia-smi --query-gpu=name,memory.total --format=csv

(cd "$KEV_DIR" && .venv/bin/python "$PIPE/kev_serve_2gpu.py" --run jaredpalmer/kev-27b --port $PORT) > kev_2gpu_server.log 2>&1 &
KEV_PID=$!
trap 'kill $KEV_PID 2>/dev/null' EXIT
for i in $(seq 1 100); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  kill -0 $KEV_PID 2>/dev/null || { echo "Kev server died:"; tail -60 kev_2gpu_server.log; exit 1; }
  sleep 10
done
echo "--- server log ---"; cat kev_2gpu_server.log

REQ='{"state": "I lost my job, so I could not pay the rent.", "model": "kev-latest",
      "questions": {"q": {"type": "choice",
        "instructions": "What is the direct causal relation from \"lost my job\" to \"could not pay the rent\"?",
        "criteria": {"enables": "the first makes the second happen or more likely",
                     "blocks": "the first prevents the second",
                     "none": "no direct causal relation"}}}}'
for k in 1 2; do
  echo "--- request $k ---"
  curl -s "http://127.0.0.1:$PORT/v1/systemone" -H 'content-type: application/json' -d "$REQ"; echo
done
nvidia-smi --query-gpu=index,memory.used --format=csv
