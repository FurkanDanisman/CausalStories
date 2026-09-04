#!/usr/bin/env python3
"""Render one PNG per model from raw-extract graph JSONs (out_raw/*.graph.json).

Each model's PNG shows all input narratives as separate labeled clusters. Run
LOCALLY where Graphviz `dot` is installed (e.g. your Mac). No cluster/model needed.

Usage:  python render_png.py            # reads out_raw/, writes out_png/<model>.png
        python render_png.py DIR OUTDIR
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

FILL = {"participant": "#F6C177", "event": "#EA9999"}


def esc(s: str) -> str:
    return s.replace("\\", " ").replace('"', "'")


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out_raw")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out_png")
    out.mkdir(exist_ok=True)

    by_tag: dict[str, list] = defaultdict(list)
    for f in sorted(src.glob("*.graph.json")):
        d = json.loads(f.read_text())
        by_tag[d["tag"]].append(d)
    if not by_tag:
        raise SystemExit(f"no *.graph.json in {src}")

    for tag, graphs in by_tag.items():
        L = [f'digraph G {{ label="{esc(tag)}"; labelloc=t; fontsize=20; fontname=Helvetica;',
             '  rankdir=LR; node [style=filled, fontname=Helvetica, fontsize=10];',
             '  edge [fontname=Helvetica, fontsize=9];']
        for gi, d in enumerate(graphs):
            cid = d["id"]
            L.append(f'  subgraph cluster_{gi} {{ label="{esc(cid)}"; style=dashed; color=gray55; fontsize=14;')
            for n in d["nodes"]:
                nid = esc(f'{cid}::{n["id"]}')
                shape = "ellipse" if n["kind"] == "participant" else "box"
                label = n["id"][len("Entity::"):] if n["id"].startswith("Entity::") else n["id"]
                fill = "#7FB3D5" if n.get("imputed") else FILL[n["kind"]]   # blue = MICE-imputed
                L.append(f'    "{nid}" [shape={shape}, fillcolor="{fill}", label="{esc(label)}"];')
            for e in d["edges"]:
                p = e.get("prob", 1.0)
                attrs = f' [label="{p:.2f}", penwidth={1 + 2 * p:.1f}]' if p < 0.999 else ""
                L.append(f'    "{esc(cid + "::" + e["head"])}" -> "{esc(cid + "::" + e["tail"])}"{attrs};')
            L.append("  }")
        L.append("}")
        png = out / f"{tag}.png"
        subprocess.run(["dot", "-Tpng", "-o", str(png)], input="\n".join(L), text=True, check=True)
        print(f"wrote {png}  ({len(graphs)} narrative(s))")


if __name__ == "__main__":
    main()
