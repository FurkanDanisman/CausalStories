#!/usr/bin/env python3
"""Local half of the world experiment: MICE imputation + the 4 plots.

Consumes what `run.py --mode world-extract` produced on the cluster (out_w/):
  raw/*.graph.json         extracted graphs (no naming standardization)
  std/*.graph.json         after canonical naming standardization
  table.json               event x person table (1 present, 0 refuted, null missing)

Produces 4 PNGs in out_w_png/ via render_png.py:
  0_raw            individual graphs, raw names
  1_standardized   individual graphs, standardized names
  2_imputed        individual graphs after MICE (imputed nodes in blue)
  3_aggregated     one probability-weighted aggregated graph

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
PLOTS = Path("out_w_plots")
PNG = Path("out_w_png")
PLOTS.mkdir(exist_ok=True)


def load(sub: str) -> dict:
    out = {}
    for f in sorted((W / sub).glob("*.graph.json")):
        d = json.loads(f.read_text())
        out[d["id"]] = d
    return out


raw, std = load("raw"), load("std")
table = json.loads((W / "table.json").read_text())
events, rows = table["events"], table["rows"]
ids = [r["id"] for r in rows]

# ---- stages 0 & 1: pass graphs through with a stage tag ----
for pid, g in raw.items():
    g = {**g, "tag": "0_raw"}
    (PLOTS / f"0-{pid}.graph.json").write_text(json.dumps(g))
for pid, g in std.items():
    g = {**g, "tag": "1_standardized"}
    (PLOTS / f"1-{pid}.graph.json").write_text(json.dumps(g))

# ---- stage 2: MICE impute the event x person table ----
X = np.array([[np.nan if c is None else float(c) for c in r["row"]] for r in rows])
if X.shape[1]:
    imp = IterativeImputer(max_iter=20, random_state=0, min_value=0, max_value=1)
    Xb = (imp.fit_transform(X) >= 0.5).astype(int)
else:
    Xb = X.astype(int)

# consensus event order = topological sort of the aggregated event DAG.
# Build the DAG greedily from highest-frequency edges, skipping any that would
# create a cycle, then Kahn topo-sort.
freq: Counter = Counter()
for g in std.values():
    ev = {n["id"] for n in g["nodes"] if n["kind"] == "event"}
    seen = set()
    for e in g["edges"]:
        k = (e["head"], e["tail"])
        if e["head"] in ev and e["tail"] in ev and k not in seen:
            freq[k] += 1
            seen.add(k)

adj: dict[str, list] = defaultdict(list)


def _reaches(src: str, dst: str) -> bool:
    stack, seen = [src], set()
    while stack:
        x = stack.pop()
        if x == dst:
            return True
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return False


for (u, v), _ in sorted(freq.items(), key=lambda kv: -kv[1]):
    if u != v and not _reaches(v, u):            # keep edge only if it stays acyclic
        adj[u].append(v)
indeg: Counter = Counter()
for u in adj:
    for v in adj[u]:
        indeg[v] += 1
queue = [e for e in events if indeg[e] == 0]
order = []
while queue:
    x = queue.pop(0)
    order.append(x)
    for y in adj[x]:
        indeg[y] -= 1
        if indeg[y] == 0:
            queue.append(y)
order += [e for e in events if e not in order]   # any leftovers

for i, pid in enumerate(ids):
    orig = {n["id"] for n in std[pid]["nodes"] if n["kind"] == "event"}
    completed = [events[j] for j in range(len(events)) if Xb[i, j] == 1]
    chain = [e for e in order if e in completed]
    nodes = [{"id": e, "kind": "event", "event_types": [], "imputed": e not in orig} for e in chain]
    nodes += [n for n in std[pid]["nodes"] if n["kind"] != "event"]
    edges = [{"head": chain[k], "tail": chain[k + 1], "prob": 1.0} for k in range(len(chain) - 1)]
    (PLOTS / f"2-{pid}.graph.json").write_text(json.dumps(
        {"tag": "2_imputed", "id": pid, "text": std[pid].get("text", ""), "nodes": nodes, "edges": edges}))

# ---- stage 3: aggregated probability graph over the standardized graphs ----
N = len(std)
edge_count: Counter = Counter()
kind = {}
for g in std.values():
    for n in g["nodes"]:
        kind[n["id"]] = n["kind"]
    seen = set()
    for e in g["edges"]:
        key = (e["head"], e["tail"])
        if key not in seen:
            edge_count[key] += 1
            seen.add(key)
agg_edges = [{"head": h, "tail": t, "prob": round(c / N, 2)} for (h, t), c in edge_count.items()]
used = {x for e in agg_edges for x in (e["head"], e["tail"])}
(PLOTS / "3-aggregated.graph.json").write_text(json.dumps(
    {"tag": "3_aggregated", "id": "aggregated", "text": "",
     "nodes": [{"id": u, "kind": kind.get(u, "event"), "event_types": []} for u in used],
     "edges": agg_edges}))

subprocess.run([sys.executable, "render_png.py", str(PLOTS), str(PNG)], check=True)
print(f"done -> {PNG}/0_raw.png 1_standardized.png 2_imputed.png 3_aggregated.png")
