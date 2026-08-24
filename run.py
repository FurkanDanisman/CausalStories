#!/usr/bin/env python3
"""Causal-graph extraction + evaluation over one example or a held-out set.

Modes (--mode):
  full     one process: extract with the test model, then judge (single example).
  extract  load ONE model, extract graph(s), save them. No judging. One process
           per model => clean GPU teardown on exit. With --examples-file the model
           is loaded once and every example is extracted.
  judge    load the judge model once, score every saved graph in --outdir against
           its gold (each saved graph records its own torque_id), and report mean
           P/R/F1 per model over all examples.

Examples selection: --torque-id (single) OR --examples-file FILE (one
"torque_id [split]" per line; blank/#-comment lines ignored).

Backends: mock | anthropic | openai (also vLLM/TGI via --base-url) | ollama |
          hf (transformers) | vllm (in-process, schema-guided JSON).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pipeline import aggregate, dataset, evaluate, extract, prompts, visualize
from pipeline.llm_client import (AnthropicClient, HFClient, MockLLMClient,
                                 OllamaClient, OpenAIClient, VLLMClient)
from pipeline.schema import CausalGraph, Variant


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


def _tag(args) -> str:
    return args.tag or (args.model or args.backend).replace("/", "_")


def _examples(args) -> list[tuple[str, str]]:
    if args.examples_file:
        out = []
        for line in Path(args.examples_file).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            out.append((parts[0], parts[1] if len(parts) > 1 else args.split))
        return out
    return [(args.torque_id, args.split)]


def _save_graph(path: Path, tag: str, ex, graph: CausalGraph) -> None:
    data = {"tag": tag, "torque_id": ex.torque_id, "split": ex.split}
    data.update(graph.model_dump(mode="json"))
    path.write_text(json.dumps(data, indent=2))


def _load_graph(path: Path):
    d = json.loads(path.read_text())
    graph = CausalGraph.model_validate({"nodes": d["nodes"], "edges": d["edges"]})
    return d["tag"], d["torque_id"], d["split"], graph


def _render(graph, report, ex, outdir: Path, name: str, backend: str) -> None:
    pred_svg = visualize.render_svg(visualize.predicted_dot(graph, report))
    gold_svg = visualize.render_svg(visualize.gold_dot(ex, report))
    meta = {"torque_id": ex.torque_id, "split": ex.split, "backend": backend,
            "text": ex.text, "is_mock": backend == "mock"}
    (outdir / f"{name}.comparison.html").write_text(
        visualize.comparison_html(gold_svg, pred_svg, report, meta))


def do_extract(args, outdir: Path) -> None:
    tag = _tag(args)
    examples = _examples(args)
    print(f"=== EXTRACT · {tag} · backend={args.backend} · {len(examples)} example(s) ===")
    for old in outdir.glob(f"{tag}##*.graph.json"):
        old.unlink()
    client = build_client(args.backend, args.model, args.base_url)
    for idx, (tid, split) in enumerate(examples):
        ex = dataset.get_example(tid, split)
        try:
            graph = extract.extract_graph(client, ex.text, k=args.k)
        except Exception as e:  # bad/truncated output must not kill the batch
            print(f"  !! [{idx:03d}] {tid}: FAILED {type(e).__name__}: {e}")
            graph = CausalGraph(nodes=[], edges=[])
        print(f"  [{idx:03d}] {tid}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        _save_graph(outdir / f"{tag}##{idx:03d}.graph.json", tag, ex, graph)
    print(f"saved {len(examples)} graph(s) for {tag}")


def do_judge(args, outdir: Path) -> None:
    files = sorted(outdir.glob("*.graph.json"))
    if not files:
        raise SystemExit(f"no *.graph.json in {outdir}; run --mode extract first")
    print(f"=== JUDGE · {args.model} · scoring {len(files)} graph(s) ===")
    judge = build_client(args.backend, args.model, args.base_url)
    per_tag = defaultdict(list)
    detail = []
    for gf in files:
        tag, tid, split, graph = _load_graph(gf)
        ex = dataset.get_example(tid, split)
        report = evaluate.evaluate(judge, graph, ex)
        per_tag[tag].append(report)
        detail.append({"model": tag, "torque_id": tid, "precision": round(report.precision, 3),
                       "recall": round(report.recall, 3), "f1": round(report.f1, 3),
                       "paper_precision": round(report.paper_precision, 3),
                       "judge": report.judge_score, "n_pred": report.n_pred,
                       "n_valid": report.n_valid, "n_gold": report.n_gold})
        if len(files) <= 6:  # only render for small single-example-style runs
            _render(graph, report, ex, outdir, gf.name[:-len(".graph.json")], backend=tag)
        print(f"  {tag} · {tid}: goldP={report.precision:.3f} R={report.recall:.3f} "
              f"F1={report.f1:.3f} | paperP={report.paper_precision:.3f} judge={report.judge_score}")

    summary = []
    for tag, reports in per_tag.items():
        n = len(reports)
        summary.append({"model": tag, "n_examples": n,
                        "gold_precision": round(sum(r.precision for r in reports) / n, 3),
                        "recall": round(sum(r.recall for r in reports) / n, 3),
                        "gold_f1": round(sum(r.f1 for r in reports) / n, 3),
                        "paper_precision": round(sum(r.paper_precision for r in reports) / n, 3),
                        "judge": round(sum(r.judge_score for r in reports) / n, 2)})
    summary.sort(key=lambda s: -s["gold_f1"])
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    (outdir / "summary_detail.json").write_text(json.dumps(detail, indent=2))
    print("=" * 78)
    print(f"{'model':<16}{'n':>4}{'goldP':>8}{'R':>8}{'goldF1':>8}{'paperP':>8}{'judge':>8}")
    for r in summary:
        print(f"{r['model']:<16}{r['n_examples']:>4}{r['gold_precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['gold_f1']:>8.3f}{r['paper_precision']:>8.3f}{r['judge']:>8.2f}")
    print(f"\ngoldP/R/F1 = vs gold graph;  paperP = predicted edges judged causally valid")
    print(f"wrote {outdir}/summary.json (means) + summary_detail.json")


def do_generate(args, outdir: Path) -> None:
    """For each base example, generate N synthetic retellings (different narrators)
    and save a story-set JSON that downstream extraction + aggregation consumes."""
    bases = _examples(args)
    n = args.n_variants
    print(f"=== GENERATE · {args.backend}:{args.model} · {len(bases)} base(s) x {n} variants ===")
    client = build_client(args.backend, args.model, args.base_url)
    for tid, split in bases:
        ex = dataset.get_example(tid, split)
        print(f"\n[{tid}] base: {ex.text}")
        variants = []
        for idx in range(n):
            persp = prompts.PERSPECTIVES[idx % len(prompts.PERSPECTIVES)]
            try:
                v = client.complete(task="generate", schema=Variant, temperature=0.8,
                                    prompt=prompts.generate_variant_prompt(ex.text, persp))
                text = v.text.strip()
            except Exception as e:
                print(f"  v{idx}: FAILED {type(e).__name__}: {e}")
                continue
            variants.append({"perspective": persp, "text": text})
            print(f"  v{idx}: {text}")
        data = {"base_torque_id": tid, "base_split": split, "base_text": ex.text,
                "gold_edges": ex.gold_edges, "variants": variants}
        (outdir / f"{tid.replace('/', '_')}.variants.json").write_text(json.dumps(data, indent=2))
    print(f"\nwrote {len(bases)} story-set(s) to {outdir}/*.variants.json")


def do_agg_extract(args, outdir: Path) -> None:
    """Extract a causal graph from each synthetic variant with ONE model."""
    tag = _tag(args)
    synth = Path(args.synth_dir)
    files = sorted(synth.glob("*.variants.json"))
    if not files:
        raise SystemExit(f"no *.variants.json in {synth}; run --mode generate first")
    print(f"=== AGG-EXTRACT · {tag} · {len(files)} base(s) ===")
    client = build_client(args.backend, args.model, args.base_url)
    for sf in files:
        d = json.loads(sf.read_text())
        vgs = []
        for idx, v in enumerate(d["variants"]):
            try:
                g = extract.extract_graph(client, v["text"], k=args.k)
            except Exception as e:
                print(f"  {d['base_torque_id']} v{idx}: FAILED {type(e).__name__}: {e}")
                g = CausalGraph(nodes=[], edges=[])
            vgs.append({"idx": idx, **g.model_dump(mode="json")})
            print(f"  {d['base_torque_id']} v{idx}: {len(g.nodes)} nodes, {len(g.edges)} edges")
        out = {"base_torque_id": d["base_torque_id"], "base_split": d["base_split"], "tag": tag,
               "variant_graphs": vgs}
        (outdir / f"{d['base_torque_id']}__{tag}.varsgraphs.json").write_text(json.dumps(out, indent=2))
    print(f"saved variant graphs for {tag}")


def do_agg_combine(args, outdir: Path) -> None:
    """Canonicalize + aggregate each base/model's variant graphs into one
    probability graph, score it vs the base's gold graph, render true vs estimated."""
    files = sorted(outdir.glob("*.varsgraphs.json"))
    if not files:
        raise SystemExit(f"no *.varsgraphs.json in {outdir}; run --mode agg-extract first")
    print(f"=== AGG-COMBINE · judge={args.model} · {len(files)} base/model set(s) ===")
    judge = build_client(args.backend, args.model, args.base_url)
    per_model = defaultdict(list)
    detail = []
    for f in files:
        d = json.loads(f.read_text())
        base_ex = dataset.get_example(d["base_torque_id"], d["base_split"])
        graphs = [CausalGraph.model_validate({"nodes": vg["nodes"], "edges": vg["edges"]})
                  for vg in d["variant_graphs"]]
        n = len(graphs)
        node2canon = aggregate.canonicalize(judge, base_ex.text, graphs)
        agg = aggregate.aggregate(graphs, node2canon, n_variants=n, min_count=args.agg_min_count)
        report = evaluate.evaluate(judge, agg, base_ex)
        name = f"{d['base_torque_id']}__{d['tag']}"
        _render(agg, report, base_ex, outdir, name, backend=f"AGG({d['tag']}, min>={args.agg_min_count}/{n})")
        (outdir / f"{name}.aggregated.json").write_text(json.dumps(
            {"base_torque_id": d["base_torque_id"], "tag": d["tag"],
             "edges": [{"head": e.head, "tail": e.tail, "prob": e.prob} for e in agg.edges]}, indent=2))
        per_model[d["tag"]].append(report)
        detail.append({"model": d["tag"], "base": d["base_torque_id"],
                       "gold_precision": round(report.precision, 3), "recall": round(report.recall, 3),
                       "gold_f1": round(report.f1, 3), "paper_precision": round(report.paper_precision, 3),
                       "judge": report.judge_score, "n_agg_edges": report.n_pred, "n_gold": report.n_gold})
        print(f"  {d['tag']} · {d['base_torque_id']}: goldP={report.precision:.3f} "
              f"R={report.recall:.3f} F1={report.f1:.3f} | paperP={report.paper_precision:.3f} "
              f"({report.n_pred} agg edges) -> {name}.comparison.html")

    summary = []
    for tag, reps in per_model.items():
        m = len(reps)
        summary.append({"model": tag, "n_bases": m,
                        "gold_precision": round(sum(r.precision for r in reps) / m, 3),
                        "recall": round(sum(r.recall for r in reps) / m, 3),
                        "gold_f1": round(sum(r.f1 for r in reps) / m, 3),
                        "paper_precision": round(sum(r.paper_precision for r in reps) / m, 3),
                        "judge": round(sum(r.judge_score for r in reps) / m, 2)})
    summary.sort(key=lambda s: -s["gold_f1"])
    (outdir / "summary_agg.json").write_text(json.dumps(summary, indent=2))
    (outdir / "summary_agg_detail.json").write_text(json.dumps(detail, indent=2))
    print("=" * 78)
    print(f"{'model':<16}{'bases':>6}{'goldP':>8}{'R':>8}{'goldF1':>8}{'paperP':>8}{'judge':>8}")
    for r in summary:
        print(f"{r['model']:<16}{r['n_bases']:>6}{r['gold_precision']:>8.3f}{r['recall']:>8.3f}"
              f"{r['gold_f1']:>8.3f}{r['paper_precision']:>8.3f}{r['judge']:>8.2f}")
    print(f"\nwrote {outdir}/summary_agg.json + per (base,model) *.comparison.html (true vs estimated)")


def do_full(args, outdir: Path) -> None:
    ex = dataset.get_example(args.torque_id, args.split)
    client = build_client(args.backend, args.model, args.base_url)
    if args.eval_backend:
        eval_client = build_client(args.eval_backend, args.eval_model, args.eval_base_url)
        eval_desc = f"{args.eval_backend}:{args.eval_model}"
    else:
        eval_client = client
        eval_desc = "SAME as test model — scores may be self-graded"
    tag = _tag(args)

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
    _save_graph(outdir / f"{tag}.graph.json", tag, ex, graph)
    _render(graph, report, ex, outdir, tag, args.backend)
    print(f"\nSTAGE 4  wrote {outdir}/{tag}.comparison.html  (two graphs side by side)")


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
    print(f"\n    gold  precision : {report.precision:.3f}   recall : {report.recall:.3f}   "
          f"F1 : {report.f1:.3f}")
    print(f"    paper precision : {report.paper_precision:.3f}  "
          f"({report.n_valid}/{report.n_pred} predicted edges judged causally valid)")
    print(f"    LLM-judge (holistic) : {report.judge_score}/5 -- {report.judge_rationale}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full",
                    choices=["full", "extract", "judge", "generate", "agg-extract", "agg-combine"])
    ap.add_argument("--backend", default="mock",
                    choices=["mock", "anthropic", "openai", "ollama", "hf", "vllm"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--tag", default=None, help="label for output files (defaults to model name)")
    ap.add_argument("--eval-backend", default=None,
                    choices=["mock", "anthropic", "openai", "ollama", "hf", "vllm"])
    ap.add_argument("--eval-model", default=None)
    ap.add_argument("--eval-base-url", default=None)
    ap.add_argument("--split", default="train", choices=["train", "dev"])
    ap.add_argument("--torque-id", default=dataset.DEMO_TORQUE_ID)
    ap.add_argument("--examples-file", default=None,
                    help="file of 'torque_id [split]' lines; overrides --torque-id")
    ap.add_argument("--k", type=int, default=1, help="self-consistency samples")
    ap.add_argument("--n-variants", type=int, default=6, help="synthetic retellings per base (generate mode)")
    ap.add_argument("--synth-dir", default="out_synth", help="dir of *.variants.json (agg-extract)")
    ap.add_argument("--agg-min-count", type=int, default=2,
                    help="keep aggregated edges seen in >= this many variants")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "extract":
        do_extract(args, outdir)
    elif args.mode == "judge":
        do_judge(args, outdir)
    elif args.mode == "generate":
        do_generate(args, outdir)
    elif args.mode == "agg-extract":
        do_agg_extract(args, outdir)
    elif args.mode == "agg-combine":
        do_agg_combine(args, outdir)
    else:
        do_full(args, outdir)


if __name__ == "__main__":
    main()
