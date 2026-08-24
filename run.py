#!/usr/bin/env python3
"""Single-example causal-graph extraction + evaluation.

Usage:
  python run.py                          # mock backend, Keating train example
  python run.py --backend anthropic --model claude-... --k 5
  python run.py --backend openai --model gpt-4o
  python run.py --backend ollama --model qwen2.5:14b
  python run.py --split dev              # longer text, structure-only gold

Outputs the extracted graph (JSON), a DOT file, and an eval report. With the
mock backend it runs fully offline and needs no API key.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import dataset, evaluate, extract, visualize
from pipeline.llm_client import (AnthropicClient, HFClient, MockLLMClient,
                                 OllamaClient, OpenAIClient)


def build_client(backend: str, model: str | None, base_url: str | None = None):
    if backend == "mock":
        return MockLLMClient(fixture="keating")
    if backend == "anthropic":
        return AnthropicClient(model=model or "REPLACE_WITH_MODEL_ID")
    if backend == "openai":                      # also: any vLLM/TGI/llama.cpp server via base_url
        return OpenAIClient(model=model or "gpt-4o", base_url=base_url)
    if backend == "ollama":
        return OllamaClient(model=model or "qwen2.5:14b")
    if backend == "hf":                          # in-process transformers (cluster job, no server)
        return HFClient(model=model or "Qwen/Qwen2.5-7B-Instruct")
    raise ValueError(f"unknown backend: {backend}")


def main() -> None:
    ap = argparse.ArgumentParser()
    # model UNDER TEST (does the extraction)
    ap.add_argument("--backend", default="mock",
                    choices=["mock", "anthropic", "openai", "ollama", "hf"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible server URL (vLLM/TGI/llama.cpp); use with --backend openai")
    # EVALUATOR model (does node alignment + judging). Keep this FIXED and strong
    # across runs so models aren't graded by themselves. Defaults to the test model.
    ap.add_argument("--eval-backend", default=None,
                    choices=["mock", "anthropic", "openai", "ollama", "hf"])
    ap.add_argument("--eval-model", default=None)
    ap.add_argument("--eval-base-url", default=None)
    ap.add_argument("--split", default="train", choices=["train", "dev"])
    ap.add_argument("--torque-id", default=dataset.DEMO_TORQUE_ID)
    ap.add_argument("--k", type=int, default=1, help="self-consistency samples")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    client = build_client(args.backend, args.model, args.base_url)
    if args.eval_backend:
        eval_client = build_client(args.eval_backend, args.eval_model, args.eval_base_url)
        eval_desc = f"{args.eval_backend}:{args.eval_model}"
    else:
        eval_client = client
        eval_desc = "SAME as test model — scores may be self-graded"
    ex = dataset.get_example(args.torque_id, args.split)

    print(f"=== example {args.torque_id} [{args.split}] · backend={args.backend} ===")
    print(f"    evaluator (align+judge): {eval_desc}")
    if args.backend == "mock":
        print("!!! MOCK backend: the PREDICTED graph is canned, NOT a real model. !!!")
    print(f"\nTEXT:\n{ex.text}\n")

    # ---- STAGE 1: nodes ----
    print("STAGE 1  extract_nodes.md  ->  typed nodes")
    graph = extract.extract_graph(client, ex.text, k=args.k)   # nodes once, edges k times
    for n in graph.nodes:
        types = f"  types={n.event_types}" if n.event_types else ""
        print(f"    [{n.kind.value:11}] {n.id}{types}")
    # ---- STAGE 2: edges ----
    print(f"\nSTAGE 2  extract_edges.md  ->  arrows (k={args.k} self-consistency samples)")
    for e in graph.edges:
        print(f"    {e.head}  ->  {e.tail}   (p={e.prob:.2f})")

    # ---- STAGE 3: eval ----
    print("\nSTAGE 3  align_nodes.md + judge.md  ->  score vs gold")
    report = evaluate.evaluate(eval_client, graph, ex)
    print("    node alignment (predicted -> gold):")
    for k_, v_ in report.alignment.items():
        print(f"        {k_!r}  ->  {v_!r}")
    print(f"\n    structurally valid : {report.valid}")
    print(f"    edge precision     : {report.precision:.3f}  (correct arrows / predicted arrows)")
    print(f"    edge recall        : {report.recall:.3f}  (gold arrows recovered / gold arrows)")
    print(f"    edge F1            : {report.f1:.3f}  "
          f"({report.matched} matched / {report.n_pred} pred / {report.n_gold} gold)")
    print(f"    LLM-judge score    : {report.judge_score}/5 -- {report.judge_rationale}")

    # ---- STAGE 4: visual comparison ----
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    (outdir / "graph.json").write_text(json.dumps(
        {"nodes": [n.model_dump() for n in graph.nodes], "edges": graph.to_gold_like()},
        indent=2))
    pred_dot = visualize.predicted_dot(graph, report)
    gold_dot = visualize.gold_dot(ex, report)
    (outdir / "predicted.dot").write_text(pred_dot)
    (outdir / "gold.dot").write_text(gold_dot)
    pred_svg = visualize.render_svg(pred_dot)
    gold_svg = visualize.render_svg(gold_dot)
    meta = {"torque_id": args.torque_id, "split": args.split, "backend": args.backend,
            "text": ex.text, "is_mock": args.backend == "mock"}
    (outdir / "comparison.html").write_text(
        visualize.comparison_html(gold_svg, pred_svg, report, meta))
    if pred_svg:
        (outdir / "predicted.svg").write_text(pred_svg)
        (outdir / "gold.svg").write_text(gold_svg)

    print(f"\nSTAGE 4  wrote {outdir}/comparison.html  (open it: two graphs side by side)")
    print(f"         also: {outdir}/gold.svg, {outdir}/predicted.svg, {outdir}/graph.json")


if __name__ == "__main__":
    main()
