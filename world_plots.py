#!/usr/bin/env python3
"""Render the world experiment's stage plots.

Two input sources:
  python world_plots.py masked      # INPUT = the true world table, ~25% masked (LLM-free)
  python world_plots.py out_w       # INPUT = Qwen's extracted table in out_w/

'masked' mode stages (every person plotted over the SAME event vocabulary):
  0_true      the known truth              green=1, red=0
  1_masked    input to imputation          green=1, red=0, gray=MASKED (missing)
  2_imputed   after MICE                   grays filled; blue border = imputed cell
  3_aggregated  edge label = fraction of people with that causal step

Colours: GREEN=1 (happened), RED=0 (did not), GRAY=missing.
Needs scikit-learn + Graphviz. Seeded, reproducible.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

PNG = Path("out_w_png"); PNG.mkdir(exist_ok=True)
GREEN, RED, GRAY, IMP_BORDER = "#A7D08C", "#E06666", "#CCCCCC", "#1F5FBF"
KIND_FILL = {"participant": "#F6C177", "event": "#EA9999"}
esc = lambda s: str(s).replace("\\", " ").replace('"', "'")


def render(name: str, title: str, clusters: list) -> None:
    """clusters: (label, nodes, edges); nodes: (id, fill, border|None); edges: (h, t, elabel|None)."""
    L = [f'digraph G {{ label="{esc(title)}"; labelloc=t; fontsize=17; rankdir=LR;',
         '  node [style=filled, shape=box, fontname=Helvetica, fontsize=10];']
    for ci, (label, nodes, edges) in enumerate(clusters):
        L.append(f'  subgraph cluster_{ci} {{ label="{esc(label)}"; style=dashed; color=gray55;')
        for nid, fill, border in nodes:
            b = f', color="{border}", penwidth=3' if border else ''
            L.append(f'    "{ci}:{esc(nid)}" [fillcolor="{fill}", label="{esc(nid)}"{b}];')
        for edge in edges:
            h, t = edge[0], edge[1]
            el = edge[2] if len(edge) > 2 else None
            style = edge[3] if len(edge) > 3 else None
            attrs = []
            if el:
                attrs.append(f'label="{esc(el)}"')
            if style == "dashed":                       # dashed = BLOCKS, solid = ENABLES
                attrs.append("style=dashed")
            a = f' [{", ".join(attrs)}]' if attrs else ""
            L.append(f'    "{ci}:{esc(h)}" -> "{ci}:{esc(t)}"{a};')
        L.append("  }")
    L.append("}")
    subprocess.run(["dot", "-Tpng", "-o", str(PNG / f"{name}.png")], input="\n".join(L), text=True, check=True)
    print(f"wrote {PNG/name}.png")


def do_masked(p_mask: float = 0.25) -> None:
    D = json.load(open("world_homelessness.variants.json"))
    KEYS = ["crisis", "support", "rent_trouble", "eviction", "homeless", "health_decline"]
    DISP = {"crisis": "crisis", "support": "support", "rent_trouble": "couldn't afford rent",
            "eviction": "eviction", "homeless": "became homeless", "health_decline": "health declined"}
    NODES = [DISP[k] for k in KEYS]
    # (head, tail, relation): ENABLES = solid arrow, BLOCKS = dashed arrow.
    WEDGES = [("crisis", "couldn't afford rent", "enable"),
              ("couldn't afford rent", "eviction", "enable"),
              ("eviction", "became homeless", "enable"),
              ("became homeless", "health declined", "enable"),
              ("support", "eviction", "block")]   # support BLOCKS eviction (protective)
    people = [f"p{i}" for i in range(len(D["variants"]))]
    T = np.array([[D["variants"][i]["true_events"][k] for k in KEYS] for i in range(len(people))], float)

    rng = random.Random(0)
    M = np.array([[rng.random() < p_mask for _ in KEYS] for _ in people])   # True = masked
    O = T.copy(); O[M] = np.nan
    Xb = (IterativeImputer(max_iter=30, random_state=0, min_value=0, max_value=1).fit_transform(O) >= 0.5).astype(int)

    def edge_tuples(labelfn=None):
        out = []
        for h, t, rel in WEDGES:
            style = "dashed" if rel == "block" else None
            out.append((h, t, (labelfn(h, t, rel) if labelfn else None), style))
        return out

    def nodes_from(colourfn):
        return [(people[i], [(NODES[j], *colourfn(i, j)) for j in range(len(KEYS))], edge_tuples())
                for i in range(len(people))]

    sub = "  (solid = enables, dashed = blocks)"
    render("0_true", "0) TRUE table  (green=1, red=0)" + sub,
           nodes_from(lambda i, j: (GREEN if T[i, j] == 1 else RED, None)))
    render("1_masked", f"1) MASKED input to MICE  (~{int(p_mask*100)}% hidden = gray)" + sub,
           nodes_from(lambda i, j: (GRAY if M[i, j] else (GREEN if T[i, j] == 1 else RED), None)))
    render("2_imputed", "2) after MICE  (grays filled; blue border = imputed cell)" + sub,
           nodes_from(lambda i, j: (GREEN if Xb[i, j] == 1 else RED, IMP_BORDER if M[i, j] else None)))

    # stage 3: aggregate the completed table into a probability graph.
    n = len(people)
    idx = {DISP[k]: j for j, k in enumerate(KEYS)}

    def agg_label(h, t, rel):
        if rel == "enable":                                 # P(both endpoints happen)
            c = sum(1 for i in range(n) if Xb[i, idx[h]] == 1 and Xb[i, idx[t]] == 1)
            return f"{c/n:.2f}"
        sup = [i for i in range(n) if Xb[i, idx[h]] == 1]   # BLOCKS: among head=1, how often tail=0
        if not sup:
            return "blocks n/a"
        return f"blocks {sum(1 for i in sup if Xb[i, idx[t]] == 0) / len(sup):.2f}"

    render("3_aggregated", "3) aggregated  (enable = fraction with the step; blocks = block rate)" + sub,
           [("aggregated", [(nm, KIND_FILL["event"], None) for nm in NODES], edge_tuples(agg_label))])
    print("done (masked input) -> out_w_png/{0_true,1_masked,2_imputed,3_aggregated}.png")


def do_pipeline(W: Path) -> None:
    def load(sub):
        return {json.loads(f.read_text())["id"]: json.loads(f.read_text())
                for f in sorted((W / sub).glob("*.graph.json"))}
    raw, std = load("raw"), load("std")
    table = json.loads((W / "table.json").read_text())
    events, rows = table["events"], table["rows"]
    ids = [r["id"] for r in rows]
    val = {r["id"]: r["row"] for r in rows}

    freq: Counter = Counter()
    for g in std.values():
        ev = {n["id"] for n in g["nodes"] if n["kind"] == "event"}
        seen = set()
        for e in g["edges"]:
            k = (e["head"], e["tail"])
            if e["head"] in ev and e["tail"] in ev and k not in seen:
                freq[k] += 1; seen.add(k)
    adj: dict[str, list] = defaultdict(list)
    def reaches(s, d):
        st, sn = [s], set()
        while st:
            x = st.pop()
            if x == d: return True
            for y in adj[x]:
                if y not in sn: sn.add(y); st.append(y)
        return False
    for (u, v), _ in sorted(freq.items(), key=lambda kv: -kv[1]):
        if u != v and not reaches(v, u): adj[u].append(v)
    indeg: Counter = Counter()
    for u in adj:
        for v in adj[u]: indeg[v] += 1
    q = [e for e in events if indeg[e] == 0]; order = []
    while q:
        x = q.pop(0); order.append(x)
        for y in adj[x]:
            indeg[y] -= 1
            if indeg[y] == 0: q.append(y)
    order += [e for e in events if e not in order]

    render("0_raw", "0) individual graphs — NO naming standardization",
           [(pid, [(n["id"], KIND_FILL[n["kind"]], None) for n in g["nodes"]],
             [(e["head"], e["tail"], None) for e in g["edges"]]) for pid, g in raw.items()])
    ce = [(order[i], order[i + 1], None) for i in range(len(order) - 1)]
    col = lambda v: GREEN if v == 1 else RED if v == 0 else GRAY
    render("1_standardized", "1) standardized — same nodes; green=present, red=refuted, gray=missing",
           [(pid, [(ev, col(val[pid][events.index(ev)]), None) for ev in order], ce) for pid in ids])
    X = np.array([[np.nan if c is None else float(c) for c in r["row"]] for r in rows])
    Xb = (IterativeImputer(max_iter=30, random_state=0, min_value=0, max_value=1).fit_transform(X) >= 0.5).astype(int)
    def n2(pid, ev):
        j = events.index(ev); i = ids.index(pid)
        return (ev, GREEN if Xb[i, j] == 1 else RED, IMP_BORDER if val[pid][j] is None else None)
    render("2_imputed", "2) after MICE — same nodes; grays filled (blue border = imputed)",
           [(pid, [n2(pid, ev) for ev in order], ce) for pid in ids])
    N = len(std); ec: Counter = Counter(); kind = {}
    for g in std.values():
        for nn in g["nodes"]: kind[nn["id"]] = nn["kind"]
        seen = set()
        for e in g["edges"]:
            k = (e["head"], e["tail"])
            if k not in seen: ec[k] += 1; seen.add(k)
    agg_nodes = {x for (h, t) in ec for x in (h, t)}
    render("3_aggregated", "3) aggregated graph — edge label = fraction of the accounts supporting it",
           [("aggregated", [(u, KIND_FILL.get(kind.get(u, "event"), "#EA9999"), None) for u in agg_nodes],
             [(h, t, f"{c/N:.2f}") for (h, t), c in ec.items()])])
    print("done (out_w pipeline) -> out_w_png/{0_raw,1_standardized,2_imputed,3_aggregated}.png")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "masked"
    if arg == "masked":
        do_masked()
    else:
        do_pipeline(Path(arg))
