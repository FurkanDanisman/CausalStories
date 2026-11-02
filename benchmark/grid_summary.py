#!/usr/bin/env python3
"""Summary of every finished grid run so far (works while the grid is still running).

For each run: time, $, and per judge the individual story score (Table 1) and the aggregated world
score at each usage cut (Table 2). Writes pipeline/grid/summary.json and summary.md and prints the md.
Usage: python benchmark/grid_summary.py"""
import json, re
from pathlib import Path

HERE = Path(__file__).parent
PIPE = HERE.parent if (HERE.parent / "run.py").exists() else HERE.parent / "pipeline"   # repo root (laptop or cluster)
GRID = PIPE / "grid"
JUDGES = {"qwen3-235b-a22b-2507": "Qwen3 235B", "gemma-4-31b-it": "Gemma 4 31B"}
CUTS = [5, 10, 15, 20, 25, 30]
NAMES = {"gemma": "Gemma 4 31B", "qwen": "Qwen 3.5 27B", "llama": "Llama 3.1 8B",
         "jev": "Jev 1.14", "kev4b": "Kev 4B", "luna": "GPT-6 Luna", "same": "same LLM"}


def main():
    rows = []
    for d in sorted(p for p in GRID.glob("*__*__*") if p.is_dir()):
        w, g, dec = d.name.split("__")
        info = json.load(open(d / "run_info.json")) if (d / "run_info.json").exists() else None
        row = {"world": w, "generator": NAMES[g], "decision": NAMES[dec], "status": "done" if info else "running/queued",
               "seconds": info.get("seconds") if info else None, "usd": info.get("usd_total") if info else None, "judges": {}}
        for slug, jn in JUDGES.items():
            ev = {}
            for c in CUTS:
                f = d / f"evaluation_{slug}_cut{c:02d}.json"
                if f.exists():
                    e = json.load(open(f))
                    ev["individual"] = {k: round(v, 3) for k, v in e["individual_summary"]["all"].items()}
                    ev["individual_outcome"] = round(e["individual_summary"]["outcome"]["score"], 3)
                    ev["individual_stopped"] = round(e["individual_summary"]["stopped"]["score"], 3)
                    a = e["aggregated"]
                    ev.setdefault("aggregated", {})[c] = {"score": round(a["score"], 3), "nodes": round(a["node_F1"], 3),
                                                          "arrows": round(a["arrows"], 3), "sid": round(a["sid"], 3),
                                                          "kl": round(a["kl_score"], 3)}
            if ev:
                row["judges"][jn] = ev
        rows.append(row)
    json.dump(rows, open(GRID / "summary.json", "w"), indent=1)

    md = ["| World | Generator | Decision | Time (s) | $ | Judge | Story score (out / stop) | " +
          " | ".join(f"Agg {c}%" for c in CUTS) + " |", "|" + "---|" * (7 + len(CUTS))]
    for r in rows:
        base = f"| {r['world']} | {r['generator']} | {r['decision']} | {r['seconds'] or '…'} | " \
               f"{'' if r['usd'] is None else r['usd']} |"
        if not r["judges"]:
            md.append(base + f" ({r['status']}) | | " + " | ".join("" for _ in CUTS) + " |")
        for jn, ev in r["judges"].items():
            ind = ev.get("individual", {})
            agg = ev.get("aggregated", {})
            md.append(base + f" {jn} | {ind.get('score', '…')} ({ev.get('individual_outcome', '')} / {ev.get('individual_stopped', '')}) | " +
                      " | ".join(str(agg[c]["score"]) if c in agg else "…" for c in CUTS) + " |")
    (GRID / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    done = sum(r["status"] == "done" for r in rows)
    print(f"\n{done} runs finished (of 27); {len(rows) - done} started but not finished.")


if __name__ == "__main__":
    main()
