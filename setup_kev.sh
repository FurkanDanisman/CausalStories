#!/bin/bash
# One-time Kev setup on the Killarney LOGIN node (needs internet; compute nodes run offline).
#   bash setup_kev.sh
# Clones github.com/jaredpalmer/kev, builds its env with uv, and downloads Kev-27B's
# adapter plus its base model (Qwen/Qwen3.8-27B, ~55 GB) into $HF_HOME.
set -euo pipefail
BASE=$HOME/projects/aip-rgrosse/furkanbd
export HF_HOME=$BASE/hf_cache
KEV_DIR=$BASE/kev

[ -d "$KEV_DIR" ] || git clone https://github.com/jaredpalmer/kev.git "$KEV_DIR"
cd "$KEV_DIR"
command -v uv >/dev/null || pip install --user uv
export PATH=$HOME/.local/bin:$PATH
uv sync --extra serve
uv pip install "flash-linear-attention==0.5.2" "triton>=3.7.1"   # fused Qwen3.5 kernels (as in Kev's own deploy image)

uv run python - <<'PY'
from huggingface_hub import snapshot_download
from kev.checkpoint import read_meta, resolve_run
p = resolve_run("jaredpalmer/kev-27b")
m = read_meta(p)
print("Kev-27B adapter:", p)
print("base:", m.base, "@", m.base_revision, "->", snapshot_download(m.base, revision=m.base_revision))
PY
echo "Kev ready in $KEV_DIR, weights in $HF_HOME"
