"""Section 2, Steps 5-6 of the agreed method (method_Oct.tex), ported from the results page that
defined it (results/api_world100_matrix/build_report.py). No model calls, no imputation.

  usage   a group is used in a story if its node has an arrow in that story's connected graph
          (chain walk + influence walk + connect walk); usage = share of stories.
  matrix  for each ordered pair of groups (a, b): the decision model's enables / blocks / none
          probabilities, averaged over the stories where both nodes are in the story's graph (its own
          nodes + inferred nodes that got an arrow) and a -> b was asked (chain walk or connect walk).
          One answer per story per pair (the least "none"); opposite members flip enables <-> blocks.
          Two views: "chain" (no influence arrows; the world graph) and "all" (with influence arrows).
  graph   an arrow a -> b wherever the highest average is enables or blocks; its estimate and 95%
          interval come from the random-effects meta-analysis of that answer (combine.meta).
  table   did it happen, per story and group (implied.json: 1 / 0 / None = cannot tell).
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from .combine import meta

ANSWERS = ("enables", "blocks", "none")


def _story(outdir: Path, abst: dict, g: dict) -> tuple[dict, dict, dict, dict]:
    """Per story: connect file, node kinds, node -> group, node -> sign."""
    j = g["id"]
    con = json.loads((outdir / f"{j}__connect.json").read_text())
    k = dict(g["kinds"]); k.update({n: "event" for n in con["new_nodes"]})
    of, sg = {}, {}
    for c, d in abst["concepts"].items():
        for m in d["members"]:
            if m[0] == j:
                of[m[1]], sg[m[1]] = c, m[2]
    for n, d in con["new_nodes"].items():
        of[n], sg[n] = d["group"], 1
    return con, k, of, sg


def usage(outdir: Path, abst: dict, graphs: list[dict]) -> dict[str, float]:
    """Share of stories in which the group's node has an arrow (with influence) in the connected graph."""
    cnt = collections.Counter()
    for g in graphs:
        con, _, of, _ = _story(outdir, abst, g)
        linked = {x for e in g["edges"] + con["edges"] for x in (e["head"], e["tail"])}
        for c in {of[n] for n in linked if n in of}:
            cnt[c] += 1
    return {c: cnt[c] / len(graphs) for c in abst["concepts"]}


def cells(outdir: Path, abst: dict, graphs: list[dict]) -> dict[str, dict[tuple[str, str], list[dict]]]:
    """Per view, per ordered group pair: one probability dict per story."""
    out = {"chain": collections.defaultdict(list), "all": collections.defaultdict(list)}
    for g in graphs:
        con, k, of, sg = _story(outdir, abst, g)

        def in_graph(view):          # the story's own nodes + inferred nodes with an arrow in that view
            es = [e for e in con["edges"] if view == "all" or k.get(e["tail"]) != "participant"]
            linked = {x for e in es for x in (e["head"], e["tail"])}
            return set(g["nodes"]) | {n for n in con["new_nodes"] if n in linked}
        graph_nodes = {"chain": in_graph("chain"), "all": in_graph("all")}
        best = {}
        for q in g["asked"] + con["asked"]:
            h, t = q["head"], q["tail"]
            if h not in of or t not in of or of[h] == of[t]:
                continue
            pr = {r: q["probs"].get(r, 0.0) for r in ANSWERS}
            if sg[h] * sg[t] < 0:
                pr["enables"], pr["blocks"] = pr["blocks"], pr["enables"]
            key = (of[h], of[t], k.get(t) == "participant")   # last: event -> participant (influence)
            if key not in best or pr["none"] < best[key][0]["none"]:
                best[key] = (pr, h, t)                     # one answer per story per pair
        for (x, y, infl), (pr, h, t) in best.items():
            if h in graph_nodes["all"] and t in graph_nodes["all"]:
                out["all"][(x, y)].append(pr)
            if not infl and h in graph_nodes["chain"] and t in graph_nodes["chain"]:
                out["chain"][(x, y)].append(pr)
    return out


def arrows(cell: dict[tuple[str, str], list[dict]]) -> list[dict]:
    """Arrow wherever the highest average is enables or blocks; estimate and interval from combine.meta."""
    res = []
    for (x, y), prs in cell.items():
        m = {r: sum(p[r] for p in prs) / len(prs) for r in ANSWERS}
        top = max(m, key=m.get)
        if top != "none":
            mm = meta([[p[top]] for p in prs])             # one answer per story (repeats = 1)
            res.append({"head": x, "tail": y, "rel": top, "average": round(m[top], 4),
                        "estimate": mm["estimate"], "ci95": mm["ci95"] if len(prs) > 1 else None, "n": len(prs)})
    return res


def aggregate(outdir: Path, graphs: list[dict]) -> dict:
    abst = json.loads((outdir / "abstraction.json").read_text())
    imp = json.loads((outdir / "implied.json").read_text())
    cl = cells(outdir, abst, graphs)
    matrix = {v: [{"head": x, "tail": y, "n": len(prs),
                   "average": {r: round(sum(p[r] for p in prs) / len(prs), 4) for r in ANSWERS}}
                  for (x, y), prs in c.items()] for v, c in cl.items()}
    return {"usage": usage(outdir, abst, graphs),
            "table": {j: d["values"] for j, d in imp.items()},
            "matrix": matrix,
            "network": {"nodes": [{"id": c, "kind": d["kind"]} for c, d in abst["concepts"].items()],
                        "arrows": arrows(cl["chain"]), "arrows_with_influence": arrows(cl["all"])}}
