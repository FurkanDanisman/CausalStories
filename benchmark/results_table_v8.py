#!/usr/bin/env python3
"""All v8 results in one table PDF (scores only; graphs are in results_individual.pdf / results_aggregated.pdf).

Rows per world: Causal Stories with Gemma 4 31B (grouping rerun with the new opposite question, grid_v8c), with
Qwen 3.5 27B (grid_v8), and with Gemma 4 31B nodes and each OpenRouter decision model (grid_v8_or); then the competing
approaches with Gemma and Qwen (grid_v8). A run that is not scored yet shows "pending"; a run that was not started
shows "not run". Rerun this script as results arrive.
Main table: individual scores, and aggregated scores at the 10% cut for Causal Stories (competitors at no cut: the
cut is a step of our method). Appendix: Causal Stories' aggregated score at every cut.
Usage: python benchmark/results_table_v8.py  ->  benchmark/stories_v8/results_table.pdf"""
import json, re, subprocess
from pathlib import Path

HERE = Path(__file__).parent
P = HERE.parent
SLUG = "gemma-4-31b-it"
CUTS = [0, 5, 10, 15, 20, 25, 30]
MAIN = 10
WORLDS = {"homelessness": "Homelessness", "mental_health": "Mental health"}
DECISION = [("Gemini Decision 1", "google_gemini-decision-1"), ("GPT-6 Luna Decisions", "openai_gpt-6-luna-decisions"),
            ("Jev 1.14", "typesafe_jev-1_14")]   # Kev 4B dropped (too slow on OpenRouter)
OUT = HERE / "stories_v8"
GENERATORS = [("gemma", "Gemma 4 31B"), ("qwen", "Qwen 3.5 27B"), ("llama70", "Llama 3.3 70B")]


def ours_rows(w):
    def newest(g, grids):                                    # the newest finished run of ours for this generator
        for gr in grids:
            if (P / gr / f"{w}__{g}__llm" / f"evaluation_{SLUG}_flip_cut{MAIN:02d}.json").exists():
                return P / gr / f"{w}__{g}__llm"
        return P / grids[-1] / f"{w}__{g}__llm"
    rows = [("Gemma 4 31B", "Gemma 4 31B", newest("gemma", ["grid_v10", "grid_v9", "grid_v8c"])),
            ("Qwen 3.5 27B", "Qwen 3.5 27B", newest("qwen", ["grid_v10", "grid_v8"])),
            ("Llama 3.3 70B", "Llama 3.3 70B", newest("llama70", ["grid_v10", "grid_v8"]))]
    rows += [("Gemma 4 31B", lab, P / "grid_v8_or" / f"{w}__gemma__{s}") for lab, s in DECISION]
    return rows


def comp_rows(w):
    rows = []
    for g, gl in GENERATORS:
        G = P / "grid_v8"
        rows += [("Per-story prompting", gl, G / f"{w}__{g}__C1-llm-llmgroup", None),
                 ("Causal Map", gl, G / f"{w}__{g}__C5-llm-llmgroup", G / f"agg__C5agg__{w}__{g}__C5-llm-llmgroup"),
                 ("Whole-corpus prompting", gl, None, G / f"{w}__{g}__C6-llm-llmgroup")]
    return rows


def ev(run, cut, flip=False):
    """flip=True (Causal Stories): the evaluation with KL computed correctly, i.e. a group stated the other way round
    (e.g. 'employment' for 'job loss') is compared with its true node with the values flipped. Only KL differs from
    the plain evaluation."""
    if run is None:
        return None
    f = run / f"evaluation_{SLUG}{'_flip' if flip else ''}_cut{cut:02d}.json"
    return json.load(open(f)) if f.exists() else None


def status(run):
    if "grid_v8c" in str(run) and not run.exists():
        return "pending (cluster)"
    return "pending" if run.exists() else "not run"


def f2(x):
    return "--" if x is None else f"{x:.2f}"


def indiv(e):
    """Mean $\\pm$ the half-width of the 95% confidence interval (1.96 standard errors) over the scored stories."""
    rows = [r for r in ((e or {}).get("individual") or {}).values() if isinstance(r, dict) and r.get("score") is not None]
    out = []
    for k in ("node_F1", "arrows", "score"):
        x = [r[k] for r in rows if r.get(k) is not None]
        if len(x) < 2:
            out.append("--"); continue
        m = sum(x) / len(x); sd = (sum((v - m) ** 2 for v in x) / (len(x) - 1)) ** 0.5
        out.append(f"{m:.2f} $\\pm$ {1.96 * sd / len(x) ** 0.5:.2f}")
    return out


def agg(e, kl):
    a = (e or {}).get("aggregated") or {}
    with_kl = None
    if kl and None not in (a.get("node_F1"), a.get("arrows"), a.get("sid"), a.get("kl_score")):
        with_kl = (a["node_F1"] * a["arrows"] * a["sid"] * a["kl_score"]) ** 0.25
    return [f2(a.get("node_F1")), f2(a.get("arrows")), f2(a.get("sid")), f2(a.get("score")), f2(with_kl) if kl else "--"]


SCORE_COLS = (2, 6, 7)                       # individual score, aggregated score, score with KL


def bold_best(rows, cols=SCORE_COLS):
    """rows: [(label, cells)]; in each score column, the highest number is bold (all of them on a tie)."""
    out = [(lab, list(c)) for lab, c in rows]
    def num(x):                                   # the mean of "0.80 $\pm$ 0.01", or None
        m = re.match(r"\d+\.\d+", x)
        return float(m.group(0)) if m else None
    for j in cols:
        vals = [num(c[j]) for _, c in rows if j < len(c) and num(c[j]) is not None]
        if not vals:
            continue
        for _, c in out:
            if j < len(c) and num(c[j]) == max(vals):
                c[j] = r"\textbf{" + c[j] + "}"
    return out


def tex():
    L = [r"\documentclass[10pt]{article}", r"\usepackage[margin=1.6cm,landscape]{geometry}", r"\usepackage{booktabs,multirow}",
         r"\pagestyle{empty}", r"\begin{document}", r"\section*{Results (benchmark v8, judge: Gemma 4 31B)}",
         r"Individual: mean $\pm$ 95\% confidence interval (1.96 standard errors) over the stories with at least one true arrow; score: geometric mean of node F1 and arrows. "
         r"Aggregated: the world graph; Causal Stories at the " + f"{MAIN}" + r"\% usage cut, competing approaches as they output them. "
         r"Score: geometric mean of node F1, arrows and SID, comparable across methods. "
         r"Score with KL: Causal Stories only (the only method with conditional probabilities): geometric mean of node F1, "
         r"arrows, SID and the KL score $\exp(-\mathrm{KL})$, where a group stated the other way round (e.g. `employment' for "
         r"`job loss') is compared with its true node with the values flipped. "
         r"Causal Stories with Gemma 4 31B and Qwen 3.5 27B: the v10 runs (new causal-edge definition, grouping tie rule); "
         r"Llama 3.3 70B and the decision models: the earlier runs (old definition), until their v10 runs are finished. "
         r"Bold: the best score among the methods with the same generator (part 1) and among the decision models (part 2).", ""]
    HEAD = [r"\toprule", r"& \multicolumn{3}{c}{Individual stories} & \multicolumn{5}{c}{Aggregated (world graph)} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-9}",
            "{} & Node F1 & Arrows & Score & Node F1 & Arrows & SID & Score & Score with KL \\\\", r"\midrule"]

    HEAD1 = [h.replace(r"\multicolumn{5}{c}", r"\multicolumn{4}{c}").replace(r"\cmidrule(lr){5-9}", r"\cmidrule(lr){5-8}")
             .replace(" & Score with KL", "") for h in HEAD]

    def ours_cells(run):
        e = ev(run, MAIN, flip=True)
        return indiv(e) + agg(e, True) if e else [status(run)] + [""] * 7

    L.append(r"\section*{1. Methods with the same generator}")
    for w, wl in WORLDS.items():
        L += [r"\subsection*{" + wl + "}", r"\begin{tabular}{lcccccccc}"] + [h.format("Method") if "{}" in h else h for h in HEAD1]
        comp = comp_rows(w)
        for k, (g, gl) in enumerate(GENERATORS):
            if k: L.append(r"\midrule")
            L.append(r"\multicolumn{8}{l}{\textit{Generator: " + gl + r"}} \\")
            block = [("Causal Stories", ours_cells(ours_rows(w)[k][2])[:7])]
            for m, _, ri, ra in [c for c in comp if c[1] == gl]:
                ei, ea = ev(ri, 0), ev(ra, 0)
                ci = indiv(ei) if ri is not None else ["--"] * 3
                if ri is not None and ei is None: ci = [status(ri), "", ""]
                ca = agg(ea, False) if ra is not None else ["--"] * 5
                if ra is not None and ea is None: ca = [status(ra), "", "", "", ""]
                block.append((m, (ci + ca)[:7]))
            for lab, cells in bold_best(block):                 # best of the methods with this generator
                L.append(r"\quad " + lab + " & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", ""]

    L += [r"\section*{2. Causal Stories: decisions by the generator or by a decision model}",
          r"Every row uses the same nodes extracted by Gemma 4 31B and Gemma 4 31B for naming the groups; only the model "
          r"that answers the decision questions (arrows, grouping, did it happen, connect) differs.", ""]
    for w, wl in WORLDS.items():
        L += [r"\subsection*{" + wl + "}", r"\begin{tabular}{lccccccccc}"] + [h.format("Decisions by") if "{}" in h else h for h in HEAD]
        rows = [r for r in ours_rows(w) if r[0] == "Gemma 4 31B"]
        block = []
        for gen, dec, run in rows:
            block.append((dec, ours_cells(run)))
        for i, (lab, cells) in enumerate(bold_best(block)):
            L.append(lab + " & " + " & ".join(cells) + r" \\")
            if i == 0: L.append(r"\midrule")
        L += [r"\bottomrule", r"\end{tabular}", ""]
    L += [r"\clearpage", r"\section*{Appendix: Causal Stories' aggregated score at every usage cut (Score / Score with KL)}"]
    for w, wl in WORLDS.items():
        L += [r"\subsection*{" + wl + "}", r"\begin{tabular}{l" + "c" * len(CUTS) + "}", r"\toprule",
              "Generator / decisions & " + " & ".join(f"{c}\\%" for c in CUTS) + r" \\", r"\midrule"]
        for gen, dec, run in ours_rows(w):
            lab = gen if gen == dec else f"{gen} / {dec}"
            cells = []
            for c in CUTS:
                a = agg(ev(run, c, flip=True), True)
                cells.append(f"{a[3]} / {a[4]}" if ev(run, c, flip=True) else "")
            if not any(cells): cells = [status(run)] + [""] * (len(CUTS) - 1)
            L.append(lab + " & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", ""]
    L.append(r"\end{document}")
    return "\n".join(L)


if __name__ == "__main__":
    b = OUT / "results_table_build"; b.mkdir(exist_ok=True)
    (b / "results_table.tex").write_text(tex())
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "results_table.tex"], cwd=b, check=True, capture_output=True)
    (b / "results_table.pdf").replace(OUT / "results_table.pdf")
    print(OUT / "results_table.pdf")
