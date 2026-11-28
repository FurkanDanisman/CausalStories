"""stories_v9/all_400_stories.pdf: every story of benchmark v9 (mental health, addiction) with its true graph
(green: happened, red: did not happen; solid green: cause, dashed red: protection)."""
import json, os, subprocess
D = os.path.dirname(os.path.abspath(__file__))
STYLE = 'graph [rankdir=LR, nodesep=0.3, ranksep=0.6, pad=0.1, fontname="Helvetica"]; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin="0.1,0.05"]; edge [arrowsize=0.7];'
GREEN = 'fillcolor="#cfeccf", color="#2e7d32"'; RED = 'fillcolor="#f6cfcf", color="#b23b3b"'
def esc(s): return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
B = f"{D}/_all"; os.makedirs(B, exist_ok=True)
tex = [r"\documentclass[10pt]{article}\usepackage[margin=0.6in]{geometry}\usepackage{graphicx}\usepackage[T1]{fontenc}\pagestyle{plain}\begin{document}"]
for w, title in [("mental_health", "Mental health"), ("addiction", "Addiction relapse")]:
    S = json.load(open(f"{D}/{w}.json")); tex.append(r"\section*{" + title + f" ({len(S)} stories)" + "}")
    for s in S:
        L = [f"digraph G {{ {STYLE}"] + [f'"{n}" [{GREEN if v else RED}];' for n, v in s["truth"]["nodes"].items()]
        for u, v, k in s["truth"]["arrows"]:
            st = 'color="#b4462f", style=dashed' if k == "blocks" else 'color="#1f7a4d"'
            L.append(f'"{u}" -> "{v}" [{st}];')
        L.append("}"); subprocess.run(["dot", "-Tpdf", "-o", f"{B}/{s['id']}.pdf"], input="\n".join(L), text=True, check=True)
        tex += [r"\noindent\begin{minipage}{\linewidth}\textbf{" + s["id"] + r"}\quad\textit{" + esc(s["text"]) + r"}\par\smallskip",
                r"\centering\includegraphics[width=0.9\linewidth,height=0.17\textheight,keepaspectratio]{" + f"{B}/{s['id']}.pdf" + r"}\end{minipage}\par\bigskip"]
tex.append(r"\end{document}"); open(f"{B}/all.tex", "w").write("\n".join(tex))
subprocess.run(["pdflatex", "-interaction=nonstopmode", "all.tex"], cwd=B, capture_output=True)
os.replace(f"{B}/all.pdf", f"{D}/all_400_stories.pdf"); print("saved all_400_stories.pdf")
