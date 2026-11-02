#!/usr/bin/env python3
"""Run the benchmark grid on OpenRouter, ONE pipeline run at a time (clean timings).

Grid: worlds x generators x decision models ("same LLM" only for homelessness x Gemma). Each finished run gets run_info.json (wall seconds,
$ from per-call OpenRouter costs, models) and is then judged in the background (not timed) by every
judge in JUDGES at every usage cut. Finished runs are skipped, so the script can be restarted.
Results at any time:  python ../benchmark/grid_summary.py
Usage:  nohup .venv/bin/python run_grid.py > grid/run_grid.log 2>&1 &"""
import json, os, subprocess, time
from pathlib import Path

HERE = Path(__file__).parent
GRID = HERE / "grid"
WORLDS = ["homelessness", "career", "mental_health"]
GENS = {"gemma": "google/gemma-4-31b-it", "qwen": "qwen/qwen3.5-27b", "llama": "meta-llama/llama-3.1-8b-instruct"}
DECS = {"jev": "~typesafe/jev-latest", "kev4b": "jaredpalmer/kev-4b", "luna": "openai/gpt-6-luna-decisions", "same": None}
SAME_ONLY = set()   # "same LLM" runs stopped on request (partial run kept in grid_stopped/)
JUDGES = ["qwen/qwen3-235b-a22b-2507", "google/gemma-4-31b-it"]   # main judge + one sensitivity judge
CUTS = "5,10,15,20,25,30"


def judge_bg(out, world):
    """Judge one finished run with every judge (skips judges already done); background, not timed."""
    cmds = " && ".join(f"JUDGE={j} {HERE}/.venv/bin/python {HERE.parent}/benchmark/evaluate_run.py {world} {out} {CUTS}"
                       for j in JUDGES)
    log = open(out / "judging.log", "a")
    subprocess.Popen(["bash", "-c", cmds], stdout=log, stderr=subprocess.STDOUT, cwd=HERE.parent)


def main():
    GRID.mkdir(exist_ok=True)
    env0 = {**os.environ}
    order = [(w, g, d) for phase in (("jev", "kev4b", "same"), ("luna",))   # Jev and Kev 4B first, Luna last
             for w in WORLDS for g in GENS for d in phase]
    for w, g, d in order:
        gm, dm = GENS[g], DECS[d]
        if True:
            if True:
                if d == "same" and (w, g) not in SAME_ONLY:
                    continue                                 # "same LLM" only for the listed world/generator
                out = GRID / f"{w}__{g}__{d}"
                if (out / "run_info.json").exists():
                    if not (out / "judging.log").exists():
                        judge_bg(out, w)
                    continue
                out.mkdir(exist_ok=True)
                dec = dm or f"llm:{gm}"
                cost_log = out / "cost.log"
                cost_log.unlink(missing_ok=True)
                env = {**env0, "GENERATOR": gm, "DECISION": dec, "COST_LOG": str(cost_log)}
                print(f"=== {time.strftime('%H:%M:%S')} start {out.name}", flush=True)
                t0 = time.time()
                with open(GRID / f"{out.name}.log", "w") as lg:
                    rc = subprocess.call([str(HERE / "run_api_or.sh"), str(HERE / f"{w}_v3.json"), str(out)],
                                         stdout=lg, stderr=subprocess.STDOUT, env=env)
                secs = round(time.time() - t0, 1)
                logtxt = (GRID / f"{out.name}.log").read_text(errors="ignore")
                if any(k in logtxt for k in ("HTTP 402", "HTTP 403", "Key limit exceeded", "Insufficient credits")):
                    print(f"=== STOPPED: billing error in {out.name}; grid halted (no empty runs recorded)", flush=True)
                    return
                bad = 0
                if (out / "chain_nodes.json").exists():
                    items = json.load(open(out / "chain_nodes.json"))
                    items = items if isinstance(items, list) else []
                    bad = sum(1 for it in items if not it.get("nodes"))
                if bad > 0.2 * 200 or "nodename nor servname" in logtxt or "URLError" in logtxt:
                    print(f"=== FAILED {out.name}: {bad} stories without nodes / network error; not recorded", flush=True)
                    continue
                if rc != 0 or not (out / "network.json").exists():
                    print(f"=== FAILED {out.name} (rc {rc}) after {secs}s; see grid/{out.name}.log", flush=True)
                    continue
                usd = {}
                for line in cost_log.read_text().splitlines() if cost_log.exists() else []:
                    k, v = line.split("\t"); usd[k] = usd.get(k, 0) + float(v)
                info = {"world": w, "generator": gm, "decision_model": dec, "seconds": secs,
                        "usd_total": round(sum(usd.values()), 4), "usd_by_part": {k: round(v, 4) for k, v in usd.items()},
                        "hardware": "hosted (OpenRouter)", "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
                json.dump(info, open(out / "run_info.json", "w"), indent=1)
                print(f"=== {time.strftime('%H:%M:%S')} done {out.name}: {secs}s ${info['usd_total']}", flush=True)
                judge_bg(out, w)


if __name__ == "__main__":
    main()
