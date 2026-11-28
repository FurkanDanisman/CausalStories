"""DRAFT worlds for the next benchmark version (not used by any run yet): two new worlds with the shape of mental
health, and mental health as it is. Same model as worlds_random.py (noisy-OR with blockers; a starting event carried
out by a participant is drawn ROLE -> event). Usage: python draft_worlds.py -> world_graphs.pdf (P(happens) per node,
among the storytellers of 200,000 simulated people, as stories_v8/world_graphs.pdf)."""
import random, subprocess, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import worlds_random as R

DRAFT = {
    "Eyewitness misidentification": {   # 200 eyewitnesses, each to their own crime; every witness tells a story
        "start": [],
        "root": {"seeing a gun": 0.40, "seeing him only for seconds": 0.50, "hinting at one photo": 0.30, "OFFICER": 0.50},
        "actors": {"hinting at one photo": "DETECTIVE"},
        "mech": {
            "saying he might not be there": ({"OFFICER": 0.80}, {}),
            "not getting a good look at his face": ({"seeing a gun": 0.7, "seeing him only for seconds": 0.8}, {}),
            "picking the wrong person": ({"not getting a good look at his face": 0.7, "hinting at one photo": 0.6},
                                         {"saying he might not be there": 0.6}),
        },
    },
    "Addiction relapse": R.WORLDS["addiction"],
    "Mental health": R.WORLDS["mental_health"],
}


def marginals(W, n=200_000, seed=0):
    rng = random.Random(seed); c = {}; m = 0
    for _ in range(n):
        x = {r: int(rng.random() < p) for r, p in W["root"].items()}
        for v, (en, bl) in W["mech"].items():
            fired = [u for u, p in en.items() if x[u] and rng.random() < p]
            stop = [b for b, q in bl.items() if x[b] and rng.random() < q]
            x[v] = int(bool(fired) and not stop)
        if W["start"] and not any(x[s] for s in W["start"]):   # storytellers: a starting event happened (all, if none listed)
            continue
        m += 1
        for k, v in x.items(): c[k] = c.get(k, 0) + v
    p = {k: v / m for k, v in c.items()}
    for e, a in W.get("actors", {}).items(): p[a] = p[e]
    return p


# drawing: exactly as stories_v8/world_graphs.pdf (plot_final.py): participants blue, protections dashed red
STYLE = '''graph [rankdir=LR, nodesep=0.5, ranksep=1.0, pad=0.3, fontname="Helvetica", labelloc=t, fontsize=18];
node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin="0.14,0.07"];
edge [fontname="Helvetica", fontsize=12, arrowsize=0.8];'''
EV = 'fillcolor="#ffffff", color="#5c6763"'; PA = 'fillcolor="#eaf0fa", color="#2d5fa8"'
def pct(p): return f"{p:.1%}" if p < 0.1 else f"{p:.0%}"
pages = []
for name, W in DRAFT.items():
    P = marginals(W)
    act = W.get("actors", {})
    root = {act.get(r, r): q for r, q in W["root"].items()}        # as worlds_random.graph(): ROLE -> event
    mech = {e: ({a: 1.0}, {}) for e, a in act.items()}; mech.update(W["mech"])
    L = [f'digraph G {{ {STYLE} label="{name}";']
    for n in list(root) + list(mech):
        L.append(f'"{n}" [label="{n}\\nP(1) {pct(P[n])} · P(0) {pct(1 - P[n])}", {PA if n.isupper() else EV}];')
    for v, (en, bl) in mech.items():
        for u, q in en.items(): L.append(f'"{u}" -> "{v}" [label=" {q:.2f} ", color="#1f7a4d", fontcolor="#1f7a4d"];')
        for u, q in bl.items(): L.append(f'"{u}" -> "{v}" [label=" −{q:.2f} ", color="#b4462f", fontcolor="#b4462f", style=dashed];')
    L.append("}")
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"_{len(pages)}.pdf")
    subprocess.run(["dot", "-Tpdf", "-Gdpi=160", "-o", f], input="\n".join(L), text=True, check=True); pages.append(f)
    print(name, {k: round(v, 2) for k, v in sorted(P.items(), key=lambda kv: -kv[1])})
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "world_graphs.pdf")
subprocess.run(["pdfunite", *pages, out], check=True); [os.remove(f) for f in pages]
print("saved", out)
