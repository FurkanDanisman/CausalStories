#!/usr/bin/env python3
"""Single-example causal-graph extraction + evaluation.

Modes (--mode):
  full     (default) one process: extract with the test model, then judge. Fine
           for API models. For big local models prefer extract/judge split below.
  extract  load ONE model, extract the graph, save it. No judging. One process
           per model => clean GPU teardown on exit.
  judge    load the judge model once, score every saved graph in --outdir vs gold,
           render the visual comparisons + a summary table.

Backends: mock | anthropic | openai (also vLLM/TGI/llama.cpp via --base-url) |
          ollama | hf (transformers, in-process) | vllm (in-process, guided JSON).

Examples:
  python run.py                                   # mock, offline sanity check
  python run.py --mode extract --backend vllm --model /model-weights/Qwen3.5-27B \\
                --tag qwen3.5-27b --torque-id <id> --outdir out_compare
  python run.py --mode judge   --backend vllm --model $JUDGE --torque-id <id> --outdir out_compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import dataset, evaluate, extract, visualize
from pipeline.llm_client import (AnthropicClient, HFClient, MockLLMClient,
                                 OllamaClient, OpenAIClient, VLLMClient)
from pipeline.schema import CausalGraph


def build_client(backend: str, model: str | None, base_url: str | None = None):
    if backend == "mock":
        return MockLLMClient(fixture="keating")
    if backend == "anthropic":
        return AnthropicClient(model=model or "REPLACE_WITH_MODEL_ID")
    if backend == "openai":                      # also: any vLLM/TGI/llama.cpp server via base_url
        return OpenAIClient(model=model or "gpt-4o", base_url=base_url)
    if backend == "ollama":
        return OllamaClient(model=model or "qwen2.5:14b")
    if backend == "hf":                          # in-process transformers
        return HFClient(model=model or "Qwen/Qwen2.5-7B-Instruct")
    if backend == "vllm":                        # in-process vLLM, schema-guided JSON
        return VLLMClient(model=model)
    raise ValueError(f"unknown backend: {backend}")


def _print_graph(graph: CausalGraph, k: int) -> None:
    print("STAGE 1  extract_nodes.md  ->  typed nodes")
    for n in graph.nodes:
        types = f"  types={n.event_types}" if n.event_types else ""
        print(f"    [{n.kind.value:11}] {n.id}{types}")
    print(f"\nSTAGE 2  extract_edges.md  ->  arrows (k={k} self-consistency samples)")
    for e in graph.edges:
        print(f"    {e.head}  ->  {e.tail}   (p={e.prob:.2f})")


def _print_report(report) -> None:
    print("\n    node alignment (predicted -> gold):")
    for k_, v_ in report.alignment.items():
        print(f"        {k_!r}  ->  {v_!r}")
    print(f"\n    structurally valid : {report.valid}")
    print(f"    edge precision     : {report.precision:.3f}  (correct arrows / predicted arrows)")
    print(f"    edge recall        : {report.recall:.3f}  (gold arrows recovered / gold arrows)")
    print(f"    edge F1            : {report.f1:.3f}  "
          f"({report.matched} matched / {report.n_pred} pred / {report.n_gold} gold)")
    print(f"    LLM-judge score    : {report.judge_score}/5 -- {report.judge_rationale}")


def _render(graph, report, ex, outdir: Path, tag: str, backend: str) -> None:
    pred_dot = visualize.predicted_dot(graph, report)
    gold_dot = visualize.gold_dot(ex, report)
    pred_svg = visualize.render_svg(pred_dot)
    gold_svg = visualize.render_svg(gold_dot)
    meta = {"torque_id": ex.torque_id, "split": ex.split, "backend": f"{backend}:{tag}",
            "text": ex.text, "is_mock": backend == "mock"}
    (outdir / f"{tag}.comparison.html").write_text(
        visualize.comparison_html(gold_svg, pred_svg, report, meta))


def do_extract(args, ex, outdir: Path) -> None:
    tag = args.tag or (args.model or args.backend).replace("/", "_")
    print(f"=== EXTRACT · {tag} · backend={args.backend} · {ex.torque_id} [{ex.split}] ===\n")
    client = build_client(args.backend, args.model, args.base_url)
    graph = extract.extract_graph(client, ex.text, k=args.k)
    _print_graph(graph, args.k)
    (outdir / f"{tag}.graph.json").write_text(json.dumps(graph.model_dump(mode="json"), indent=2))
    print(f"\nsaved {outdir}/{tag}.graph.json")


def do_judge(args, ex, outdir: Path) -> None:
    graph_files = sorted(outdir.glob("*.graph.json"))
    if not graph_files:
        raise SystemExit(f"no *.graph.json in {outdir}; run --mode extract first")
    print(f"=== JUDGE · {args.model} · scoring {len(graph_files)} graph(s) ===\n")
    judge = build_client(args.backend, args.model, args.base_url)
    summary = []
    for gf in graph_files:
        tag = gf.name[:-len(".graph.json")]
        graph = CausalGraph.model_validate(json.loads(gf.read_text()))
        print(f"--- {tag} ---")
        report = evaluate.evaluate(judge, graph, ex)
        _print_report(report)
        _render(graph, report, ex, outdir, tag, backend=tag)
        summary.append({"model": tag, "precision": round(report.precision, 3),
                        "recall": round(report.recall, 3), "f1": round(report.f1, 3),
                        "judge": report.judge_score, "n_pred": report.n_pred,
                        "n_gold": report.n_gold})
        print()
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("=" * 64)
    print(f"{'model':<20}{'P':>7}{'R':>7}{'F1':>7}{'judge':>7}")
    for r in summary:
        print(f"{r['model']:<20}{r['precision']:>7.3f}{r['recall']:>7.3f}"
              f"{r['f1']:>7.3f}{r['judge']:>6}/5")
    print(f"\nwrote {outdir}/summary.json + per-model *.comparison.html")


def do_full(args, ex, outdir: Path) -> None:
    client = build_client(args.backend, args.model, args.base_url)
    if args.eval_backend:
        eval_client = build_client(args.eval_backend, args.eval_model, args.eval_base_url)
        eval_desc = f"{args.eval_backend}:{args.eval_model}"
    else:
        eval_client = client
        eval_desc = "SAME as test model — scores may be self-graded"
    tag = args.tag or (args.model or args.backend).replace("/", "_")

    print(f"=== example {ex.torque_id} [{ex.split}] · backend={args.backend} ===")
    print(f"    evaluator (align+judge): {eval_desc}")
    if args.backend == "mock":
        print("!!! MOCK backend: the PREDICTED graph is canned, NOT a real model. !!!")
    print(f"\nTEXT:\n{ex.text}\n")

    graph = extract.extract_graph(client, ex.text, k=args.k)
    _print_graph(graph, args.k)
    print("\nSTAGE 3  align_nodes.md + judge.md  ->  score vs gold")
    report = evaluate.evaluate(eval_client, graph, ex)
    _print_report(report)
    (outdir / f"{tag}.graph.json").write_text(json.dumps(graph.model_dump(mode="json"), indent=2))
    _render(graph, report, ex, outdir, tag, args.backend)
    print(f"\nSTAGE 4  wrote {outdir}/{tag}.comparison.html  (two graphs side by side)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["full", "extract", "judge"])
    # model UNDER TEST (extract/full) or the JUDGE (judge mode)
    ap.add_argument("--backend", default="mock",
                    choices=["mock", "anthropic", "openai", "ollama", "hf", "vllm"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible server URL (vLLM/TGI/llama.cpp); use with --backend openai")
    ap.add_argument("--tag", default=None, help="label for output files (defaults to model name)")
    # EVALUATOR (full mode only). Keep FIXED and strong so models aren't self-graded.
    ap.add_argument("--eval-backend", default=None,
                    choices=["mock", "anthropic", "openai", "ollama", "hf", "vllm"])
    ap.add_argument("--eval-model", default=None)
    ap.add_argument("--eval-base-url", default=None)
    ap.add_argument("--split", default="train", choices=["train", "dev"])
    ap.add_argument("--torque-id", default=dataset.DEMO_TORQUE_ID)
    ap.add_argument("--k", type=int, default=1, help="self-consistency samples")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ex = dataset.get_example(args.torque_id, args.split)

    if args.mode == "extract":
        do_extract(args, ex, outdir)
    elif args.mode == "judge":
        do_judge(args, ex, outdir)
    else:
        do_full(args, ex, outdir)


if __name__ == "__main__":
    main()
