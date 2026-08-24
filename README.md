# Single-example causal-graph extraction

Validation step for the larger methodology: prove that an LLM can extract a
causal graph — **participants, events, and directed causal arrows between them**
— from ONE TORQUESTRA narrative, and score it against the gold human graph.

We intentionally **do not** emit ENABLES/BLOCKS labels or fine-grained
sub-relations. An arrow is enough. Node typing (event vs participant, plus
FrameNet/MAVEN event types on events) is kept so output aligns with gold nodes.

## Run it (offline, no API key)

```bash
pip install -r requirements.txt
python run.py                     # mock backend, Keating train example
```

This runs the whole pipeline with a canned model, prints the extracted graph +
eval report, and writes `out/graph.json` and `out/graph.dot`.

## Swap in a real model

The pipeline is model-agnostic — one `LLMClient.complete(task, prompt, schema)`
call per stage. Pick a backend and pass a model id:

```bash
python run.py --backend anthropic --model <model-id> --k 5   # needs ANTHROPIC_API_KEY
python run.py --backend openai    --model gpt-4o
python run.py --backend ollama    --model qwen2.5:14b        # local, offline
```

## Running on a cluster (e.g. Killarney)

Two ways to run open-source models on a GPU node:

**A) Serve with vLLM (OpenAI-compatible), then point the pipeline at it:**
```bash
# on the GPU node:
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000
# run the pipeline (any host that can reach the node):
python run.py --backend openai --base-url http://<node>:8000/v1 \
              --model Qwen/Qwen2.5-7B-Instruct
```

**B) In-process via transformers (no server), inside a batch job:**
```bash
python run.py --backend hf --model Qwen/Qwen2.5-7B-Instruct
```

### Held-out eval example + a separate evaluator

The current eval target (fully held out; the prompt's few-shot example is
disjoint from it) is `docid_CNN19980227.2130.0067_sentid_10`. Keep the
**evaluator** model (node alignment + judge) fixed and strong so a model is not
scored by itself:

```bash
python run.py \
  --backend hf --model Qwen/Qwen2.5-7B-Instruct \        # model UNDER TEST
  --eval-backend hf --eval-model Qwen/Qwen2.5-72B-Instruct \  # fixed strong judge
  --torque-id docid_CNN19980227.2130.0067_sentid_10 --split train \
  --outdir out_bomb
```
Omit the `--eval-*` flags and the test model grades itself (a warning is printed).

`--k` is self-consistency: extract nodes once, extract edges K times against the
fixed node set, arrow probability = fraction of runs that drew it. (This is the
single-text preview of the cross-narrative edge-probability aggregation in the
methodology.)

## Pipeline stages

| stage | file | what it does |
|-------|------|--------------|
| load | `pipeline/dataset.py` | pick example by `torque_id`/`split`; strip gold edge labels |
| nodes | `pipeline/extract.py` | LLM → typed nodes (event/participant + event_types) |
| edges | `pipeline/extract.py` | LLM → directed arrows over the fixed node set, K-sample |
| eval | `pipeline/evaluate.py` | LLM node alignment → unlabeled-edge P/R/F1 + LLM-judge |
| viz | `pipeline/visualize.py` | Graphviz DOT (`dot -Tpng out/graph.dot -o g.png`) |

Structured output is enforced per stage via pydantic schemas (`pipeline/schema.py`).

## Known limitations (v1, by design)

- **Eval reuses the LLM** to align extracted↔gold node names, so scores are
  bounded by the aligner — the same canonicalization problem the full pipeline
  targets, here as a single controlled pair.
- **Cross-run edge aggregation is exact-string** over a fixed node vocabulary
  (safe because nodes are extracted once); it does not merge paraphrased edges.
- **Cross-narrative concepts** (canonicalization, intervenability, contradiction,
  imputation) are out of scope here — this validates extraction only. Their
  definitions still need to be sourced before the aggregation stage is built.
