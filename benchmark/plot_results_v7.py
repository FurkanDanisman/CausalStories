#!/usr/bin/env python3
"""Results PDF for the v7 benchmark runs (grid_v7/): graphs with their scores.

Individual graphs: 10 stories per world (the same stories for every method), each shown as its true graph,
then Ours, Per-story prompting and Causal Map for each generator, with node F1, arrows and story score.
Aggregated graphs: per world and generator, the true world graph, Ours A (LLM grouping) and Ours B (Kev
procedure) at every usage cut, then Causal Map and Whole-corpus prompting as they output them (no usage cut: the cut is a step of our method), with node F1, arrows,
SID, score, and KL for ours. Every graph is drawn in true-node names as the judge matched them; a node the
judge could not match keeps the method's own name (grey). Green arrow: enables; red: blocks.
Usage: python benchmark/plot_results_v7.py  ->  benchmark/stories_v7/results_individual.pdf, results_aggregated.pdf"""
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
STYLE = '''graph [rankdir=TB, nodesep=0.25, ranksep=0.45, pad=0.15, fontname="Helvetica"];
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


def aggregated(run, cut, tag, labels=True):
    """labels: write the arrow's estimate on it (ours only: the other methods' numbers are not probabilities)."""
    e = ev(run, cut)
    if e is None or e["aggregated"] is None:
        return None, "not judged at this cut yet"
    a = e["aggregated"]
    nodes = {(v["match"] or k): not v["match"] for k, v in a["kept_groups"].items()}
    edges = []
    for x in a["arrows_list"]:
        u, v = x["head"].removeprefix("x:"), x["tail"].removeprefix("x:")
        nodes.setdefault(u, True); nodes.setdefault(v, True)
        edges.append((u, v, x["sign"], f'{x["estimate"]:.2f}' if labels else ""))
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


HEAD = r"\documentclass[10pt]{article}\usepackage[margin=0.6in]{geometry}\usepackage{graphicx}\usepackage[T1]{fontenc}\begin{document}"
NOTE = (r"Judge: Gemma 4 31B. Every method's graph is drawn in the true nodes' names as the judge matched them; a grey node is one "
        r"the judge could not match to any true node (an extra node). Green arrow: enables; red arrow: blocks.")


def cell(f, cap, h):
    body = (r"\includegraphics[width=\linewidth,height=" + h + r"\textheight,keepaspectratio]{" + f + "}") if f else ""
    return r"\begin{minipage}[t]{0.49\linewidth}\centering " + body + r"\par\small " + cap + r"\end{minipage}"


def grid(cells, h):
    """2 x 2: cells[0] top left, cells[1] top right, cells[2] bottom left, cells[3] bottom right."""
    return (r"\noindent" + cell(*cells[0], h) + r"\hfill" + cell(*cells[1], h) + r"\par\bigskip\noindent" +
            cell(*cells[2], h) + r"\hfill" + cell(*cells[3], h) + r"\par\medskip")


def build(name, tex):
    (OUT / f"{name}.tex").write_text("\n".join(tex + [r"\end{document}"]))
    subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{name}.tex"], cwd=OUT, capture_output=True)
    (OUT / f"{name}.pdf").replace(HERE / "stories_v7" / f"{name}.pdf")
    print("saved", HERE / "stories_v7" / f"{name}.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---------------- individual graphs: 2 x 2 per story (true | ours / per-story | Causal Map)
    tex = [HEAD, r"\section*{Individual graphs}", NOTE + r" 10 stories per world, the same for every method. Top left: the "
           r"story's true graph; top right: ours; bottom left: Per-story prompting; bottom right: Causal Map. Ours A and Ours B share "
           r"their story graphs (they differ only in grouping). Story score = geometric mean of node F1 and arrows."]
    for w, wt in WORLDS.items():
        S = {s["id"]: s for s in json.load(open(HERE / "stories_v7" / f"{w}.json"))}
        pick = sorted(random.Random(21).sample(sorted(S), 10))
        for g, gt in GENS.items():
            for sid in pick:
                s = S[sid]
                tex.append(r"\clearpage\subsection*{" + wt + " -- " + gt + " -- " + sid + r"}\noindent\textit{``" + esc(s["text"]) + r"''}\par\medskip")
                cells = [(true_story(w, s), r"\textbf{True graph}")]
                for lab, run in [("Ours", f"{w}__{g}__llm-llmgroup"), ("Per-story prompting", f"{w}__{g}__C1-llm-llmgroup"),
                                 ("Causal Map", f"{w}__{g}__C5-llm-llmgroup")]:
                    f, cap = individual(w, sid, run, f"{g}_{lab[:4]}")
                    cells.append((f, f"\\textbf{{{lab}}}\\\\" + cap))
                tex.append(grid(cells, "0.3"))
    build("results_individual", tex)
    # ---------------- aggregated graphs: 2 x 2 at the 15% cut (ours), appendix: every other cut
    tex = [HEAD, r"\section*{Aggregated graphs}", NOTE + r" Top left: the true world graph (arrow labels: cause strength / protection "
           r"strength); top right: ours at the 15\% usage cut (a group is kept if at least 15\% of stories use it); bottom left: Causal Map "
           r"and bottom right: Whole-corpus prompting, as they output them (the usage cut is a step of our method only). One page for "
           r"each of our two groupings. Score = geometric mean of node F1, arrows and SID; KL (only ours outputs conditional "
           r"probabilities) is reported separately. Our arrows are labelled with our estimate; the other methods' arrows carry no number (they output none: Causal Map's link weight is only the share of stories that mention the link, and Whole-corpus prompting gives no weight). Appendix: ours at every other cut."]
    OURS = [("Ours A (LLM grouping)", "llm-llmgroup", "A"), ("Ours B (Kev procedure)", "llm", "B")]
    for w, wt in WORLDS.items():
        for g, gt in GENS.items():
            comp = []
            for lab, run in [("Causal Map", f"agg__C5agg__{w}__{g}__C5-llm-llmgroup"), ("Whole-corpus prompting", f"{w}__{g}__C6-llm-llmgroup")]:
                f, cap = aggregated(run, 0, f"{w}_{g}_{lab[:5]}", labels=False)
                comp.append((f, f"\\textbf{{{lab}}}\\\\" + cap))
            for lab, suf, t in OURS:
                f, cap = aggregated(f"{w}__{g}__{suf}", 15, f"{w}_{g}_{t}")
                tex.append(r"\clearpage\subsection*{" + wt + " -- " + gt + " -- " + lab + r"}")
                tex.append(grid([(true_world(w), r"\textbf{True world graph}"), (f, f"\\textbf{{{lab}, 15\\% cut}}\\\\" + cap)] + comp, "0.38"))
    tex.append(r"\clearpage\appendix\section*{Appendix: ours at every usage cut}")
    for w, wt in WORLDS.items():
        for g, gt in GENS.items():
            for lab, suf, t in OURS:
                tex.append(r"\clearpage\subsection*{" + wt + " -- " + gt + " -- " + lab + r"}")
                cs = [c for c in CUTS if c != 15]
                for i in range(0, len(cs), 4):
                    cells = []
                    for c in cs[i:i + 4]:
                        f, cap = aggregated(f"{w}__{g}__{suf}", c, f"{w}_{g}_{t}")
                        cells.append((f, f"\\textbf{{{c}\\% cut}}\\\\" + cap))
                    while len(cells) < 4: cells.append((None, ""))
                    tex.append(grid(cells, "0.36") + r"\clearpage")
    build("results_aggregated", tex)


if __name__ == "__main__":
    main()
