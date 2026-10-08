#!/usr/bin/env python
"""Serve Kev-27B split across two GPUs (e.g. 2 x L40S, 48 GB each).

Kev's own server (kev.serve) loads the whole model onto one GPU, and Kev-27B (~55 GB
in bf16) does not fit on one 48 GB card. This launcher, run with Kev's Python:
  1. loads the checkpoint on the CPU (bf16, LoRA adapter applied, pointer head loaded),
  2. spreads the backbone's decoder layers over cuda:0 and cuda:1 with accelerate,
     keeping the embeddings, rotary embedding and final norm on cuda:0 so the hidden
     states reach the pointer head (also on cuda:0) on the right device,
  3. sets the current CUDA device before each layer runs (the Triton kernels of the
     Gated DeltaNet layers launch on the current device),
  4. moves the cache-reorder index to each cache layer's device,
  5. serves the same /v1/systemone API as kev.serve.
Fused kernels and CUDA graphs (single-GPU speed-ups) are off.

    cd $KEV_DIR && .venv/bin/python /path/to/kev_serve_2gpu.py --run jaredpalmer/kev-27b --port 8009
"""

import argparse

import torch


def _tensor_device(obj, depth=0):
    """Device of the first tensor found in an object's attributes (dicts and lists included)."""
    if isinstance(obj, torch.Tensor):
        return obj.device
    if depth > 2:
        return None
    vals = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, (list, tuple)) else \
        getattr(obj, "__dict__", {}).values()
    for v in vals:
        d = _tensor_device(v, depth + 1)
        if d is not None:
            return d
    return None


def _patch_cache_reorder():
    from transformers import cache_utils

    def reorder_cache(self, beam_idx):
        for layer in self.layers:
            dev = _tensor_device(layer)
            layer.reorder_cache(beam_idx.to(dev) if dev is not None else beam_idx)

    cache_utils.Cache.reorder_cache = reorder_cache


def _split(model, max_memory):
    """Dispatch model.lm over two GPUs; embeddings, rotary embedding and final norm on cuda:0."""
    from accelerate import dispatch_model, infer_auto_device_map

    lm = model.lm
    layers_name = next(n for n, m in lm.named_modules()
                       if n.endswith("layers") and isinstance(m, torch.nn.ModuleList))
    stack = layers_name[: -len(".layers")]
    layer_cls = type(dict(lm.named_modules())[layers_name][0]).__name__
    dm = infer_auto_device_map(lm, max_memory=max_memory, no_split_module_classes=[layer_cls])
    for part in ("embed_tokens", "rotary_emb", "norm"):
        if f"{stack}.{part}" in dict(lm.named_modules()):
            dm[f"{stack}.{part}"] = 0
    used = sorted({d for d in dm.values()}, key=str)
    print(f"device map over {used}: {sum(1 for k in dm if '.layers.' in k)} layer groups placed", flush=True)
    model.lm = dispatch_model(lm, device_map=dm, main_device=0)

    # Triton kernels launch on the current device: set it to each layer's device before it runs
    for layer in dict(model.lm.named_modules())[layers_name]:
        dev = next(layer.parameters()).device
        layer.register_forward_pre_hook(lambda mod, args, dev=dev: torch.cuda.set_device(dev))

    model.head.to("cuda:0")
    model.device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-27b")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--gpu0", default="36GiB", help="memory for weights on cuda:0 (it also holds the head and caches)")
    ap.add_argument("--gpu1", default="44GiB", help="memory for weights on cuda:1")
    a = ap.parse_args()
    if torch.cuda.device_count() < 2:
        raise SystemExit(f"needs 2 GPUs, found {torch.cuda.device_count()}")

    from kev import serve
    from kev.checkpoint import Checkpoint, LoadOptions

    _patch_cache_reorder()
    ck = Checkpoint(a.run)
    opts = LoadOptions(dtype=torch.bfloat16, attn="sdpa", backend="torch", cuda_graphs=False, fused=False)
    print(f"loading {a.run} on the CPU (bf16) ...", flush=True)
    tok, model = ck.load("cpu", opts)
    model = _split(model, {0: a.gpu0, 1: a.gpu1, "cpu": "200GiB"})
    for i in range(2):
        print(f"cuda:{i} allocated {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB", flush=True)

    serve.app.state.server = serve.Server(ck, tok, model, "cuda")
    print(f"serving {a.run} on 2 GPUs at {a.host}:{a.port}", flush=True)
    import uvicorn
    uvicorn.run(serve.app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
