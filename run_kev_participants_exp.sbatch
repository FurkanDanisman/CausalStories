#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=kev-exp-part
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=kev-exp-part-%j.out
#
# Experiment: participant arrows as in Regan et al. (2023). participant -> event and
# event -> participant both get enables / blocks / none (no participant -> participant).
# Kev only, on the nodes we already have: out_kev/chain_nodes.json (run 6057822: p0, p4,
# p8) and out_gd3/chain_nodes.json (24 GD3 answers). Results in out_exp_paper/.
#   sbatch run_kev_participants_exp.sbatch
set -uo pipefail

module load gcc opencv
source .venv/bin/activate
BASE=$HOME/projects/aip-rgrosse/furkanbd
export HF_HOME=$BASE/hf_cache
KEV_DIR=$BASE/kev
GEMMA=$BASE/model-weights/gemma-4-31B-it
OUT=out_exp_paper
export PARTICIPANT_ARROWS=paper
export PROMPT_TEMPLATE_DIR=$PWD/prompt_variants/paper_participants
PIPE=$PWD
export VLLM_TP=2
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

echo "############ starting Kev-27B server ############"
(cd "$KEV_DIR" && .venv/bin/python "$PIPE/kev_serve_2gpu.py" --run jaredpalmer/kev-27b --port $PORT) > "$OUT/kev_server.log" 2>&1 &
KEV_PID=$!
trap 'kill $KEV_PID 2>/dev/null' EXIT
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  kill -0 $KEV_PID 2>/dev/null || { echo "Kev server died:"; tail -40 "$OUT/kev_server.log"; exit 1; }
  sleep 10
done
curl -s "http://127.0.0.1:$PORT/v1/models"; echo

echo "############ Kev: walk with participant arrows (paper) ############"
for NODES in out_kev/chain_nodes.json out_gd3/chain_nodes.json; do
  python run.py --mode chain-kev --kev-url "http://127.0.0.1:$PORT" \
    --nodes-file "$NODES" --tag paper --outdir "$OUT"
done

echo "############ stopping Kev ############"
kill $KEV_PID 2>/dev/null; wait $KEV_PID 2>/dev/null
echo "Done. Saved in $OUT/: *__paper.chain.json"
