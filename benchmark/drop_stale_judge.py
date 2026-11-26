#!/usr/bin/env python3
"""Drop judge answers made against an older ground truth, so the next judge run asks again.
Usage: python benchmark/drop_stale_judge.py <grid dir> benchmark/stories_v7/judge_stale_<world>.json
Removes, in every run of that world in <grid dir> (aggregation outputs included): the individual answers of the listed
stories and every aggregated answer (the true node list changed)."""
import json, sys
from pathlib import Path

grid, spec = Path(sys.argv[1]), json.load(open(sys.argv[2]))
w, ids = spec["world"], set(spec["stories"])
for run in sorted(grid.iterdir()):
    name = run.name.split("__", 2)[-1] if run.name.startswith("agg__") else run.name
    if not name.startswith(w + "__"):
        continue
    for f in run.glob("judge_cache_*.json"):
        c = json.load(open(f))
        drop = [k for k in c if k.startswith("agg:") or (k.startswith("ind:") and k[4:] in ids)]
        for k in drop:
            del c[k]
        json.dump(c, open(f, "w"), indent=1)
        (run / ".judged").unlink(missing_ok=True)
        print(f"{run.name}: dropped {len(drop)} answers")
