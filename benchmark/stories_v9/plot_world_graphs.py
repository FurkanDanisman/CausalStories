"""stories_v9/world_graphs.pdf: the true world graphs of benchmark v9 (mental health, addiction), drawn exactly as
stories_v8/world_graphs.pdf: participants blue, causes green, protections dashed red; P(1) / P(0) among storytellers."""
import os, subprocess, sys
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(D)); import worlds_random as R
STYLE = '''graph [rankdir=LR, nodesep=0.5, ranksep=1.0, pad=0.3, fontname="Helvetica", labelloc=t, fontsize=18];
node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin="0.14,0.07"];
edge [fontname="Helvetica", fontsize=12, arrowsize=0.8];'''
EV = 'fillcolor="#ffffff", color="#5c6763"'; PA = 'fillcolor="#eaf0fa", color="#2d5fa8"'
def pct(p): return f"{p:.1%}" if p < 0.1 else f"{p:.0%}"
pages = []
for w, title in [("mental_health", "Mental health"), ("addiction", "Addiction relapse")]:
    root, mech = R.graph(w); vals = [s["values"] for s in R.draw_stories(w, 100_000, seed=1)]
    P = {k: sum(v[k] for v in vals) / len(vals) for k in vals[0]}
    L = [f'digraph G {{ {STYLE} label="{title}";']
    for n in list(root) + list(mech):
        L.append(f'"{n}" [label="{n}\\nP(1) {pct(P[n])} · P(0) {pct(1 - P[n])}", {PA if n.isupper() else EV}];')
    for v, (en, bl) in mech.items():
        for u, p in en.items(): L.append(f'"{u}" -> "{v}" [label=" {p:.2f} ", color="#1f7a4d", fontcolor="#1f7a4d"];')
        for u, q in bl.items(): L.append(f'"{u}" -> "{v}" [label=" −{q:.2f} ", color="#b4462f", fontcolor="#b4462f", style=dashed];')
    L.append("}"); f = f"{D}/_wg_{w}.pdf"
    subprocess.run(["dot", "-Tpdf", "-Gdpi=160", "-o", f], input="\n".join(L), text=True, check=True); pages.append(f)
subprocess.run(["pdfunite", *pages, f"{D}/world_graphs.pdf"], check=True); [os.remove(f) for f in pages]; print("saved world_graphs.pdf")
