#!/usr/bin/env python3
"""Results PDF for the v7 benchmark runs (grid_v7/): graphs with their scores.

Individual graphs: 10 stories per world (the same stories for every method), each shown as its true graph,
then Ours, Per-story prompting and Causal Map for each generator, with node F1, arrows and story score.
Aggregated graphs: per world and generator, the true world graph, Ours A (LLM grouping) and Ours B (Kev
procedure) at every usage cut, then Causal Map and Whole-corpus prompting as they output them (no usage cut: the cut is a step of our method), with node F1, arrows,
SID, score, and KL for ours. Every graph is drawn in true-node names as the judge matched them; a node the
judge could not match keeps the method's own name (grey). Green arrow: enables; red: blocks.
Usage: python benchmark/plot_results_v7.py  ->  benchmark/stories_v7/results_graphs.pdf"""
import json, random, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
GRID = HERE.parent / "grid_v7"
OUT = HERE / "stories_v7" / "results_build"
SLUG = "gemma-4-31b-it"
CUTS = [0, 5, 10, 15, 20, 25, 30]
GENS = {"gemma": "Gemma 4 31B", "qwen": "Qwen 3.5 27B"}
WORLDS = {"homelessness": "Homelessness", "mental_health": "Mental health"}
POS = {"carries out", "enables", "risk"}
sys.path.insert(0, str(HERE)); import worlds_random as R
STYLE = '''graph [rankdir=LR, nodesep=0.35, ranksep=0.6, pad=0.15, fontname="Helvetica"];
node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin="0.12,0.06", fillcolor="#ffffff", color="#5c6763"];
edge [fontname="Helvetica", fontsize=10, arrowsize=0.8];'''
EXTRA = 'fillcolor="#eeeeee", color="#999999", fontcolor="#666666"'
PLUS, MINUS = 'color="#1f7a4d", fontcolor="#1f7a4d"', 'color="#b4462f", fontcolor="#b4462f"'


def ev(run, cut):
    f = GRID / run / f"evaluation_{SLUG}_cut{cut:02d}.json"
    return json.load(open(f)) if f.exists() else None


def dot(name, nodes, edges):
    """nodes: {name: extra?}; edges: [(u, v, '+'/'-', label)]."""
    L = [f"digraph G {{ {STYLE}"]
    for n, extra in nodes.items():
        L.append(f'"{n}" [{EXTRA if extra else ""}];')
    for u, v, s, lab in edges:
        L.append(f'"{u}" -> "{v}" [{PLUS if s == "+" else MINUS}{", label=" + chr(34) + lab + chr(34) if lab else ""}];')
    L.append("}")
    f = OUT / f"{name}.pdf"
    subprocess.run(["dot", "-Tpdf", "-o", str(f)], input="\n".join(L), text=True, check=True)
    return f.name


def esc(s):
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def fig(f, cap, h="0.27"):
    return (r"\begin{minipage}{\linewidth}\centering\includegraphics[width=\linewidth,height=" + h +
            r"\textheight,keepaspectratio]{" + f + r"}\par\small " + cap + r"\end{minipage}\par\medskip")


def individual(w, sid, run, tag):
    """A method's story graph in true-node names, with its scores."""
    e = ev(run, 10)
    g = json.load(open(GRID / run / f"{sid}__gemma-kev.chain.json"))
    r = e["individual"][sid]
    m = r["map"]
    kinds = g.get("kinds") or {}
    lab = {n: (m.get(n) or n) for n in g["nodes"]}
    nodes = {lab[n]: not m.get(n) for n in g["nodes"]}
    edges = [(lab[x["head"]], lab[x["tail"]], "+" if x["rel"] == "enables" else "-", "")
             for x in g["edges"] if kinds.get(x["tail"], "event") != "participant"]
    f = dot(f"ind_{tag}_{sid}", nodes, edges)
    return f, f"node F1 {r['node_F1']:.2f} $\\cdot$ arrows {r['arrows']:.2f} $\\cdot$ story score \\textbf{{{r['score']:.2f}}}"


def aggregated(run, cut, tag):
    e = ev(run, cut)
    if e is None or e["aggregated"] is None:
        return None, "not judged at this cut yet"
    a = e["aggregated"]
    nodes = {(v["match"] or k): not v["match"] for k, v in a["kept_groups"].items()}
    edges = []
    for x in a["arrows_list"]:
        u, v = x["head"].removeprefix("x:"), x["tail"].removeprefix("x:")
        nodes.setdefault(u, True); nodes.setdefault(v, True)
        edges.append((u, v, x["sign"], f'{x["estimate"]:.2f}'))
    f = dot(f"agg_{tag}_{cut:02d}", nodes, edges)
    cap = (f"{len(a['kept_groups'])} groups kept $\\cdot$ node F1 {a['node_F1']:.2f} $\\cdot$ arrows {a['arrows']:.2f} "
           f"$\\cdot$ SID {a['sid']:.2f} $\\cdot$ score \\textbf{{{a['score']:.2f}}}")
    if a.get("kl_score") is not None:
        cap += f" $\\cdot$ KL {a['kl_score']:.2f}"
    return f, cap


def true_story(w, s):
    nodes = {n: False for n in s["truth"]["nodes"]}
    edges = [(u, v, "+" if k in POS else "-", "") for u, v, k in s["truth"]["arrows"]]
    return dot(f"true_{s['id']}", nodes, edges)


def true_world(w):
    W = R.WORLDS[w]
    nodes = {n: False for n in list(W["root"]) + list(W["mech"])}
    edges = [(u, v, "+", f"{p:.2f}") for v, (en, bl) in W["mech"].items() for u, p in en.items()] + \
            [(u, v, "-", f"{q:.2f}") for v, (en, bl) in W["mech"].items() for u, q in bl.items()]
    return dot(f"true_world_{w}", nodes, edges)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tex = [r"\documentclass[10pt]{article}\usepackage[margin=0.6in]{geometry}\usepackage{graphicx}\usepackage[T1]{fontenc}\begin{document}",
           r"\section*{Benchmark v7 results: graphs and scores}",
           r"Judge: Gemma 4 31B. Every method's graph is drawn in the true nodes' names as the judge matched them; a grey node is one "
           r"the judge could not match to any true node (an extra node). Green arrow: enables; red arrow: blocks. Individual graphs: "
           r"10 stories per world, the same for every method. Ours A and Ours B share their story graphs (they differ only in grouping). "
           r"Aggregated score = geometric mean of node F1, arrows and SID; KL (only ours outputs conditional probabilities) is reported "
           r"separately. Aggregated arrows are labelled with the method's estimate."]
    for w, wt in WORLDS.items():
        S = {s["id"]: s for s in json.load(open(HERE / "stories_v7" / f"{w}.json"))}
        pick = sorted(random.Random(21).sample(sorted(S), 10))
        for g, gt in GENS.items():
            tex.append(r"\clearpage\section*{" + wt + " -- individual graphs -- " + gt + "}")
            runs = [("Ours", f"{w}__{g}__llm-llmgroup"), ("Per-story prompting", f"{w}__{g}__C1-llm-llmgroup"),
                    ("Causal Map", f"{w}__{g}__C5-llm-llmgroup")]
            for sid in pick:
                s = S[sid]
                tex.append(r"\subsection*{" + sid + r"}\noindent\textit{``" + esc(s["text"]) + r"''}\par\medskip")
                tex.append(fig(true_story(w, s), "\\textbf{True graph}", "0.13"))
                for lab, run in runs:
                    f, cap = individual(w, sid, run, f"{g}_{lab[:4]}")
                    tex.append(fig(f, f"\\textbf{{{lab}}}: " + cap, "0.13"))
                tex.append(r"\clearpage")
        for g, gt in GENS.items():
            tex.append(r"\clearpage\section*{" + wt + " -- aggregated graphs -- " + gt + "}")
            tex.append(fig(true_world(w), "\\textbf{True world graph} (arrow labels: cause strength / protection strength)", "0.3"))
            for lab, run in [("Ours A (LLM grouping)", f"{w}__{g}__llm-llmgroup"), ("Ours B (Kev procedure)", f"{w}__{g}__llm")]:
                tex.append(r"\subsection*{" + lab + r" -- every usage cut}")
                for c in CUTS:
                    f, cap = aggregated(run, c, f"{w}_{g}_{lab[5]}")
                    tex.append(fig(f, f"\\textbf{{{c}\\% cut}}: " + cap, "0.3") if f else f"\\textbf{{{c}\\% cut}}: {cap}\\par\\medskip")
            tex.append(r"\subsection*{Competing methods (their own output, no usage cut)}")
            for lab, run in [("Causal Map", f"agg__C5agg__{w}__{g}__C5-llm-llmgroup"), ("Whole-corpus prompting", f"{w}__{g}__C6-llm-llmgroup")]:
                f, cap = aggregated(run, 0, f"{w}_{g}_{lab[:5]}")
                tex.append(fig(f, f"\\textbf{{{lab}}}: " + cap, "0.3"))
    tex.append(r"\end{document}")
    (OUT / "results_graphs.tex").write_text("\n".join(tex))
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "results_graphs.tex"], cwd=OUT, capture_output=True)
    (OUT / "results_graphs.pdf").replace(HERE / "stories_v7" / "results_graphs.pdf")
    print("saved", HERE / "stories_v7" / "results_graphs.pdf")


if __name__ == "__main__":
    main()
