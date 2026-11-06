#!/usr/bin/env python3
"""Where the time goes, per run: seconds per step (timing.tsv), and the connect step's size (inferred nodes added
per story, questions asked). Compares with an older results folder when given.
Usage: python benchmark/step_stats.py [grid_cluster] [grid_cluster_v3]"""
import json, sys
from collections import defaultdict
from pathlib import Path


def stats(root):
    rows = {}
    for d in sorted(Path(root).glob("*__*__*")):
        if d.name.startswith(("agg__", "_")) or "_n" in d.name.split("__")[-1] or not (d / "timing.tsv").exists():
            continue
        t = defaultdict(float)
        for line in open(d / "timing.tsv"):
            p = line.split("\t")
            if len(p) >= 2:
                try: t[p[0]] += float(p[1])
                except ValueError: pass
        n = nodes = q = 0                         # one connect file at a time (login nodes have little memory)
        for f in d.glob("*__connect.json"):
            if f.name.startswith("._"): continue
            with open(f) as fh: c = json.load(fh)
            n += 1; nodes += len(c["new_nodes"]); q += len(c["asked"]); del c
        rows[d.name] = {"steps": dict(t), "total": sum(t.values()), "new_nodes": nodes / n if n else None,
                        "connect_q": q if n else None, "stories": n}
    return rows


def main():
    new = stats(sys.argv[1] if len(sys.argv) > 1 else "grid_cluster")
    old = stats(sys.argv[2]) if len(sys.argv) > 2 else {}
    print(f"{'run':<34} {'total min':>9} {'connect min':>11} {'connect %':>9} {'groups min':>10} {'new nodes/story':>15} {'connect questions':>17}")
    for name, r in new.items():
        c, g = r["steps"].get("connect", 0), r["steps"].get("groups", 0)
        nn = f"{r['new_nodes']:.1f}" if r["new_nodes"] is not None else "-"
        o = old.get(name)
        if o and o["new_nodes"] is not None: nn += f" (was {o['new_nodes']:.1f})"
        cq = str(r["connect_q"]) if r["connect_q"] is not None else "-"
        if o and o["connect_q"]: cq += f" (was {o['connect_q']})"
        print(f"{name:<34} {r['total']/60:>9.0f} {c/60:>11.0f} {100*c/r['total'] if r['total'] else 0:>8.0f}% {g/60:>10.0f} {nn:>15} {cq:>17}")


if __name__ == "__main__":
    main()
