#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=kev-chain
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=kev-chain-%j.out
#
# Chain walk on world stories p0, p4, p8 (controlled_kev.json): Gemma picks the nodes
# and their opposite names, then Kev-27B does the walk, then Gemma merges the nodes
# of all graphs into shared concepts. Gemma runs first and exits,
# then the Kev server starts: one 80 GB GPU, used in turn (Kev-27B needs ~66 GB).
# The first run also sets up Kev inside the job (clone, env, ~55 GB of weights);
# later runs skip that step.
#   sbatch run_kev_chain.sbatch
set -uo pipefail

module load gcc opencv
source .venv/bin/activate
BASE=$HOME/projects/aip-rgrosse/furkanbd
export HF_HOME=$BASE/hf_cache
KEV_DIR=$BASE/kev
GEMMA=$BASE/model-weights/gemma-4-31B-it
OUT=out_kev
PORT=8009
mkdir -p "$OUT"

echo "############ Kev setup (skipped if already done) ############"
if [ ! -f "$KEV_DIR/.setup_done" ]; then
  set -e
  [ -d "$KEV_DIR" ] || git clone https://github.com/jaredpalmer/kev.git "$KEV_DIR"
  command -v uv >/dev/null || pip install uv          # into the active pipeline venv (pip --user fails inside a venv)
  (cd "$KEV_DIR" && uv sync --extra serve \
     && uv pip install "flash-linear-attention==0.5.2" "triton>=3.7.1" \
     && uv run python -c "
from huggingface_hub import snapshot_download
from kev.checkpoint import read_meta, resolve_run
p = resolve_run('jaredpalmer/kev-27b'); m = read_meta(p)
print('Kev-27B adapter:', p)
print('base:', m.base, '@', m.base_revision, '->', snapshot_download(m.base, revision=m.base_revision))
")
  touch "$KEV_DIR/.setup_done"
  set +e
fi
export HF_HUB_OFFLINE=1

echo "############ Gemma: nodes + opposites ############"
VLLM_MAX_LEN=8192 python run.py --mode chain-nodes --backend vllm --model "$GEMMA" \
  --texts-file controlled_kev.json --max-nodes 5 --outdir "$OUT"

echo "############ starting Kev-27B server ############"
(cd "$KEV_DIR" && .venv/bin/python -m kev.serve --run jaredpalmer/kev-27b --port $PORT) > "$OUT/kev_server.log" 2>&1 &
KEV_PID=$!
trap 'kill $KEV_PID 2>/dev/null' EXIT
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  kill -0 $KEV_PID 2>/dev/null || { echo "Kev server died:"; tail -40 "$OUT/kev_server.log"; exit 1; }
  sleep 10
done
curl -s "http://127.0.0.1:$PORT/v1/models"; echo

echo "############ Kev: chain walk on Gemma nodes ############"
python run.py --mode chain-kev --kev-url "http://127.0.0.1:$PORT" \
  --nodes-file "$OUT/chain_nodes.json" --tag gemma-kev --outdir "$OUT"

echo "############ stopping Kev (frees the GPU for Gemma) ############"
kill $KEV_PID 2>/dev/null; wait $KEV_PID 2>/dev/null

echo "############ Gemma: merge into shared concepts (Section 2, Steps 2-3) ############"
VLLM_MAX_LEN=8192 python run.py --mode abstract --backend vllm --model "$GEMMA" \
  --graphs-glob "$OUT/*__gemma-kev.chain.json" --merge-groups 2 --outdir "$OUT"

echo "Done. Saved in $OUT/: *.chain.json (graphs) and abstraction.json (concepts)"
