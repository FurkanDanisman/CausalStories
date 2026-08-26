#!/usr/bin/env python3
"""Render TRUE (gold) vs ESTIMATED (aggregated) graphs locally from *.aggregated.json.

Reads each aggregated graph (base_torque_id, tag, edges[head,tail,prob]), pulls the
base's gold graph from the LOCAL dataset, renders both with Graphviz, and writes one
self-contained HTML you can open. No model / cluster needed.

Usage:
  python render_local.py agg_pasted/*.aggregated.json     # or no args -> agg_pasted/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pipeline import dataset, visualize


def _kind(nid: str) -> str:
    return "participant" if nid.startswith("Entity::") else "event"


def _panel(agg_path: Path) -> str:
    d = json.loads(agg_path.read_text())
    tid, tag = d["base_torque_id"], d.get("tag", "?")
    ex = dataset.get_example(tid, d.get("base_split", "train"))

    gnodes = {n: _kind(n) for n in ex.gold_node_ids}
    gedges = []
    for e in ex.gold_edges:
        gnodes.setdefault(e["head"], _kind(e["head"]))
        gnodes.setdefault(e["tail"], _kind(e["tail"]))
        gedges.append((e["head"], e["tail"], "#2E8B57", ""))
    gold_svg = visualize.render_svg(visualize._dot("TRUE / gold", list(gnodes.items()), gedges))

    aids = {n for e in d["edges"] for n in (e["head"], e["tail"])}
    anodes = [(n, _kind(n)) for n in sorted(aids)]
    aedges = [(e["head"], e["tail"], "#E69138", f"{e['prob']:.2f}") for e in d["edges"]]
    agg_svg = visualize.render_svg(visualize._dot(f"ESTIMATED / aggregated ({tag})", anodes, aedges))

    return (f"<h2>{tid} &middot; {tag}</h2><p style='color:#555'>{ex.text}</p>"
            f"<div style='display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap'>"
            f"<div>{gold_svg or '(no graphviz)'}</div><div>{agg_svg or '(no graphviz)'}</div></div><hr>")


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or sorted(Path("agg_pasted").glob("*.aggregated.json"))
    if not paths:
        raise SystemExit("no *.aggregated.json given (put them in agg_pasted/ or pass as args)")
    body = "\n".join(_panel(p) for p in paths)
    out = Path("agg_local.html")
    out.write_text(f"<html><body style='font-family:sans-serif;max-width:1300px;margin:auto'>{body}</body></html>")
    print(f"wrote {out}  ({len(paths)} panel(s))")


if __name__ == "__main__":
    main()
