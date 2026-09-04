#!/usr/bin/env python3
"""Local half of the world experiment: render the 4 stages from out_w/.

Key idea (per Furkan): after standardization the event VOCABULARY is fixed, so every
person is plotted over the SAME set of canonical events. A node is coloured by its
value in that person's row:
    GREEN = 1 (present / happened)   RED = 0 (refuted / did not)   GRAY = missing
Stage 1 shows present/refuted/missing; MICE then fills the grays -> stage 2 has no
gray, imputed cells get a thick blue border.

Stages: 0_raw (raw extracted graphs), 1_standardized, 2_imputed, 3_aggregated.
Usage:  python world_plots.py [out_w]     (needs scikit-learn + Graphviz)
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

W = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out_w")
PNG = Path("out_w_png"); PNG.mkdir(exist_ok=True)
GREEN, RED, GRAY, IMP_BORDER = "#A7D08C", "#E06666", "#CCCCCC", "#1F5FBF"
KIND_FILL = {"participant": "#F6C177", "event": "#EA9999"}
esc = lambda s: str(s).replace("\\", " ").replace('"', "'")


def load(sub: str) -> dict:
    return {json.loads(f.read_text())["id"]: json.loads(f.read_text())
            for f in sorted((W / sub).glob("*.graph.json"))}


def render(name: str, title: str, clusters: list) -> None:
    """clusters: list of (label, nodes, edges); nodes: (id, fill, border|None); edges: (h, t, elabel|None)."""
    L = [f'digraph G {{ label="{esc(title)}"; labelloc=t; fontsize=17; rankdir=LR;',
         '  node [style=filled, fontname=Helvetica, fontsize=10];']
    for ci, (label, nodes, edges) in enumerate(clusters):
        L.append(f'  subgraph cluster_{ci} {{ label="{esc(label)}"; style=dashed; color=gray55;')
        for nid, fill, border in nodes:
            b = f', color="{border}", penwidth=3' if border else ''
            L.append(f'    "{ci}:{esc(nid)}" [shape=box, fillcolor="{fill}", label="{esc(nid)}"{b}];')
        for h, t, el in edges:
            lab = f' [label="{el}"]' if el else ''
            L.append(f'    "{ci}:{esc(h)}" -> "{ci}:{esc(t)}"{lab};')
        L.append("  }")
    L.append("}")
    subprocess.run(["dot", "-Tpng", "-o", str(PNG / f"{name}.png")], input="\n".join(L), text=True, check=True)
    print(f"wrote {PNG/name}.png")


raw, std = load("raw"), load("std")
table = json.loads((W / "table.json").read_text())
events, rows = table["events"], table["rows"]
ids = [r["id"] for r in rows]
val = {r["id"]: r["row"] for r in rows}   # 1 / 0 / None per event

# ----- consensus event order (topological over aggregated event-edges) -----
freq: Counter = Counter()
for g in std.values():
    ev = {n["id"] for n in g["nodes"] if n["kind"] == "event"}
    seen = set()
    for e in g["edges"]:
        k = (e["head"], e["tail"])
        if e["head"] in ev and e["tail"] in ev and k not in seen:
            freq[k] += 1; seen.add(k)
adj: dict[str, list] = defaultdict(list)
def _reaches(s, d):
    st, sn = [s], set()
    while st:
        x = st.pop()
        if x == d: return True
        for y in adj[x]:
            if y not in sn: sn.add(y); st.append(y)
    return False
for (u, v), _ in sorted(freq.items(), key=lambda kv: -kv[1]):
    if u != v and not _reaches(v, u): adj[u].append(v)
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

# ----- stage 0: raw extracted graphs (varying, raw names) -----
render("0_raw", "0) individual graphs — NO naming standardization",
       [(pid, [(n["id"], KIND_FILL[n["kind"]], None) for n in g["nodes"]],
         [(e["head"], e["tail"], None) for e in g["edges"]]) for pid, g in raw.items()])

# ----- stage 1: standardized, full vocab, coloured by table value (green/red/gray) -----
def colour(v): return GREEN if v == 1 else RED if v == 0 else GRAY
chain_edges = [(order[i], order[i + 1], None) for i in range(len(order) - 1)]
render("1_standardized", "1) standardized — same nodes; green=present, red=refuted, gray=missing",
       [(pid, [(ev, colour(val[pid][events.index(ev)]), None) for ev in order], chain_edges) for pid in ids])

# ----- stage 2: MICE-impute the grays; full vocab; imputed cells get a blue border -----
X = np.array([[np.nan if c is None else float(c) for c in r["row"]] for r in rows])
Xb = (IterativeImputer(max_iter=30, random_state=0, min_value=0, max_value=1).fit_transform(X) >= 0.5).astype(int)
def node2(pid, ev):
    j = events.index(ev); i = ids.index(pid)
    was_missing = val[pid][j] is None
    return (ev, GREEN if Xb[i, j] == 1 else RED, IMP_BORDER if was_missing else None)
render("2_imputed", "2) after MICE — same nodes; grays filled (blue border = imputed)",
       [(pid, [node2(pid, ev) for ev in order], chain_edges) for pid in ids])

# ----- stage 3: aggregated probability graph over standardized graphs -----
N = len(std); ec: Counter = Counter(); kind = {}
for g in std.values():
    for n in g["nodes"]: kind[n["id"]] = n["kind"]
    seen = set()
    for e in g["edges"]:
        k = (e["head"], e["tail"])
        if k not in seen: ec[k] += 1; seen.add(k)
agg_nodes = {x for (h, t) in ec for x in (h, t)}
render("3_aggregated", "3) aggregated graph — edge label = fraction of the 10 supporting it",
       [("aggregated", [(u, KIND_FILL.get(kind.get(u, "event"), "#EA9999"), None) for u in agg_nodes],
         [(h, t, f"{c/N:.2f}") for (h, t), c in ec.items()])])

print("done -> out_w_png/{0_raw,1_standardized,2_imputed,3_aggregated}.png")
