#!/usr/bin/env python3
"""True per-person plots from world_homelessness.variants.json true_events.
Node colour: GREEN = happened (1), RED = did not (0), GRAY = not applicable (missing)."""
import json, subprocess
from pathlib import Path

D = json.load(open("world_homelessness.variants.json"))
KEY2NODE = {"crisis": "crisis", "support": "support", "rent_trouble": "couldn't afford rent",
            "eviction": "eviction", "homeless": "became homeless", "health_decline": "health declined"}
NODES = ["crisis", "support", "couldn't afford rent", "eviction", "became homeless", "health declined"]
EDGES = [("crisis", "couldn't afford rent"), ("couldn't afford rent", "eviction"),
         ("eviction", "became homeless"), ("became homeless", "health declined"), ("support", "eviction")]
GREEN, RED, GRAY = "#A7D08C", "#E06666", "#CCCCCC"
esc = lambda s: s.replace('"', "'")

L = ['digraph G { label="TRUE individual plots  (green=happened, red=did not, gray=n/a)"; labelloc=t;',
     '  fontsize=18; rankdir=LR; node [style=filled, fontname=Helvetica, fontsize=10];']
for i, v in enumerate(D["variants"]):
    te = v.get("true_events", {})
    L.append(f'  subgraph cluster_{i} {{ label="p{i}"; style=dashed; color=gray55;')
    for node in NODES:
        key = next(k for k, nm in KEY2NODE.items() if nm == node)
        val = te.get(key)
        fill = GREEN if val == 1 else RED if val == 0 else GRAY
        shape = "ellipse" if node == "support" else "box"
        L.append(f'    "p{i}::{node}" [shape={shape}, fillcolor="{fill}", label="{esc(node)}"];')
    for h, t in EDGES:
        L.append(f'    "p{i}::{h}" -> "p{i}::{t}";')
    L.append("  }")
L.append("}")
Path("out_w_png").mkdir(exist_ok=True)
subprocess.run(["dot", "-Tpng", "-o", "out_w_png/true_individuals.png"], input="\n".join(L), text=True, check=True)
print("wrote out_w_png/true_individuals.png")
