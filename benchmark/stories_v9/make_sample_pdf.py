"""The first 10 addiction stories with their true graphs (green: happened, red: did not happen; dashed: protection)."""
import json, subprocess, os
D = os.path.dirname(os.path.abspath(__file__)); F = json.load(open(f"{D}/addiction_facts.json"))[:10]; T = json.load(open(f"{D}/addiction_10_texts.json"))
STYLE = 'graph [rankdir=LR, nodesep=0.35, ranksep=0.7, pad=0.15, fontname="Helvetica"]; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin="0.12,0.06"]; edge [arrowsize=0.8];'
GREEN = 'fillcolor="#cfeccf", color="#2e7d32"'; RED = 'fillcolor="#f6cfcf", color="#b23b3b"'
os.makedirs(f"{D}/_sample", exist_ok=True)
tex = [r"\documentclass[11pt]{article}\usepackage[margin=0.8in]{geometry}\usepackage{graphicx}\usepackage[T1]{fontenc}\pagestyle{empty}\begin{document}",
       r"\section*{Addiction relapse: the first 10 stories}"]
for s in F:
    L = [f"digraph G {{ {STYLE}"] + [f'"{n}" [{GREEN if v else RED}];' for n, v in s["truth"]["nodes"].items()]
    for u, v, k in s["truth"]["arrows"]:
        st = 'color="#b4462f", style=dashed' if k == "blocks" else 'color="#1f7a4d"'
        L.append(f'"{u}" -> "{v}" [{st}];')
    L.append("}")
    subprocess.run(["dot", "-Tpdf", "-o", f"{D}/_sample/{s['id']}.pdf"], input="\n".join(L), text=True, check=True)
    tex += [r"\subsection*{" + s["id"] + r"}\noindent\textit{``" + T[s["id"]] + r"''}\par\medskip",
            r"\begin{center}\includegraphics[width=0.95\linewidth,height=0.28\textheight,keepaspectratio]{" + f"{D}/_sample/{s['id']}.pdf" + r"}\end{center}"]
tex.append(r"\end{document}"); open(f"{D}/_sample/s.tex", "w").write("\n".join(tex))
subprocess.run(["pdflatex", "-interaction=nonstopmode", "s.tex"], cwd=f"{D}/_sample", capture_output=True)
os.replace(f"{D}/_sample/s.pdf", f"{D}/addiction_10_stories.pdf"); print("saved")
