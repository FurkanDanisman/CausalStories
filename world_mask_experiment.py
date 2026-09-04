#!/usr/bin/env python3
"""Masking + MICE recovery on the TRUE world table (no LLM).
Take the known 0/1 truth, mask ~25% of cells (gray = missing), MICE-impute them,
and check recovery. Renders: true_masked.png (input) and recovered.png (output)."""
import json, random, subprocess
from pathlib import Path
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

D = json.load(open("world_homelessness.variants.json"))
KEYS = ["crisis", "support", "rent_trouble", "eviction", "homeless", "health_decline"]
DISP = {"crisis": "crisis", "support": "support", "rent_trouble": "couldn't afford rent",
        "eviction": "eviction", "homeless": "became homeless", "health_decline": "health declined"}
NODES = [DISP[k] for k in KEYS]
EDGES = [("crisis", "couldn't afford rent"), ("couldn't afford rent", "eviction"),
         ("eviction", "became homeless"), ("became homeless", "health declined"), ("support", "eviction")]
people = [f"p{i}" for i in range(len(D["variants"]))]
T = np.array([[D["variants"][i]["true_events"][k] for k in KEYS] for i in range(len(people))], float)

rng = random.Random(0)
P_MASK = 0.25
M = np.array([[rng.random() < P_MASK for _ in KEYS] for _ in people])
O = T.copy(); O[M] = np.nan

imp = IterativeImputer(max_iter=30, random_state=0, min_value=0, max_value=1)
I = (imp.fit_transform(O) >= 0.5).astype(int)

masked = np.argwhere(M)
acc = sum(int(I[r, c] == T[r, c]) for r, c in masked) / max(len(masked), 1)

print("per person:  value  or  [true->imputed ✓/✗] for masked cells")
print("events:", KEYS)
for i, p in enumerate(people):
    cells = []
    for j in range(len(KEYS)):
        if M[i, j]:
            ok = "✓" if I[i, j] == T[i, j] else "✗"
            cells.append(f"[{int(T[i,j])}->{I[i,j]}{ok}]")
        else:
            cells.append(f" {int(T[i,j])} ")
    print(f"  {p}: " + " ".join(cells))
print(f"\nmasked cells: {len(masked)}   MICE recovery accuracy: {acc:.2f}")

GREEN, RED, GRAY = "#A7D08C", "#E06666", "#CCCCCC"
Path("out_w_png").mkdir(exist_ok=True)

def render(name, fillfn, title):
    L = [f'digraph G {{ label="{title}"; labelloc=t; fontsize=17; rankdir=LR;',
         '  node [style=filled, fontname=Helvetica, fontsize=10];']
    for i, p in enumerate(people):
        L.append(f'  subgraph cluster_{i} {{ label="{p}"; style=dashed; color=gray55;')
        for j, node in enumerate(NODES):
            fill, extra = fillfn(i, j)
            shape = "ellipse" if node == "support" else "box"
            L.append(f'    "{p}::{node}" [shape={shape}, fillcolor="{fill}", label="{node}"{extra}];')
        for h, t in EDGES:
            L.append(f'    "{p}::{h}" -> "{p}::{t}";')
        L.append("  }")
    L.append("}")
    subprocess.run(["dot", "-Tpng", "-o", f"out_w_png/{name}.png"], input="\n".join(L), text=True, check=True)

render("true_masked", lambda i, j: (GRAY if M[i, j] else (GREEN if T[i, j] == 1 else RED), ""),
       "Masked TRUE plots  (green=1, red=0, gray=MASKED)")

def recovered_fill(i, j):
    val = I[i, j] if M[i, j] else T[i, j]
    fill = GREEN if val == 1 else RED
    if M[i, j]:
        col = "#1b6b34" if I[i, j] == T[i, j] else "#8b0000"
        return fill, f', color="{col}", penwidth=4'
    return fill, ""
render("recovered", recovered_fill,
       "MICE-RECOVERED  (thick border = imputed cell; dark-green=correct, dark-red=wrong)")
print("wrote out_w_png/true_masked.png and out_w_png/recovered.png")
