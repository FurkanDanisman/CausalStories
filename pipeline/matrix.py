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
  model   the causal model, the output (ported from results/calc_model.py): every node gets
          P(node = 1 | its parents in the graph), counted from the table; a story where the node or a
          parent is None is skipped for that count; only parent combinations that occur are stored. Arrows
          are added strongest first and an arrow that would close a cycle is left out of the model. Each
          arrow also carries P(tail = 1 | head = 1) in the stories (both known).
"""

from __future__ import annotations

import collections
import math
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


def cells(outdir: Path, abst: dict, graphs: list[dict], ids: bool = False) -> dict[str, dict[tuple[str, str], list]]:
    """Per view, per ordered group pair: one probability dict per story (with ids=True: (story id, dict))."""
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
            item = (g["id"], pr) if ids else pr
            if h in graph_nodes["all"] and t in graph_nodes["all"]:
                out["all"][(x, y)].append(item)
            if not infl and h in graph_nodes["chain"] and t in graph_nodes["chain"]:
                out["chain"][(x, y)].append(item)
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


# ---------------------------------------------------------------- Step 7: resolving cycles (method_Oct.tex)

ALPHA = 0.05
MAX_KNOT_ARROWS = 20          # listing the candidates of a knot is exponential in its arrows; larger knots are reported


def evidence(pr: dict) -> float:
    """Evidence for an arrow from one story's answer: log(p_arrow / p_neither), probabilities kept in [0.001, 0.999]."""
    clip = lambda x: min(max(x, 0.001), 0.999)
    return math.log(clip(pr["enables"] + pr["blocks"]) / clip(pr["none"]))


def knots(edges: set[tuple[str, str]]) -> list[set[str]]:
    """Strongly connected components with at least two concepts (Tarjan): every cycle lies inside one."""
    adj = collections.defaultdict(list)
    nodes = sorted({x for e in edges for x in e})
    for h, t in sorted(edges):
        adj[h].append(t)
    index, low, on, stack, out, n = {}, {}, set(), [], [], [0]

    def visit(v):
        index[v] = low[v] = n[0]; n[0] += 1; stack.append(v); on.add(v)
        for w in adj[v]:
            if w not in index:
                visit(w); low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = set()
            while True:
                w = stack.pop(); on.discard(w); comp.add(w)
                if w == v:
                    break
            if len(comp) > 1:
                out.append(comp)
    import sys
    sys.setrecursionlimit(max(1000, 10 * len(nodes) + 100))
    for v in nodes:
        if v not in index:
            visit(v)
    return out


def _acyclic(edges) -> bool:
    adj = collections.defaultdict(list)
    for h, t in edges:
        adj[h].append(t)
    state = {}

    def dfs(u):
        state[u] = 1
        for v in adj[u]:
            if state.get(v) == 1 or (v not in state and not dfs(v)):
                return False
        state[u] = 2
        return True
    return all(u in state or dfs(u) for u in list(adj))


def candidates(knot_arrows: list[tuple[str, str]]) -> list[frozenset]:
    """The maximal acyclic sets of a knot's arrows: leaving out any other arrow of the knot would be needed,
    putting one back closes a cycle."""
    arr, out = sorted(knot_arrows), []
    for mask in range(1 << len(arr)):
        keep = [a for i, a in enumerate(arr) if mask >> i & 1]
        if not _acyclic(keep):
            continue
        if all(not _acyclic(keep + [a]) for i, a in enumerate(arr) if not mask >> i & 1):
            out.append(frozenset(keep))
    return out


def _wilcoxon(d: list[float]) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on the nonzero differences: (p-value, W+ - W-)."""
    d = [x for x in d if x != 0]
    if not d:
        return 1.0, 0.0
    from scipy.stats import wilcoxon
    order = sorted(range(len(d)), key=lambda i: abs(d[i]))
    ranks, i = [0.0] * len(d), 0
    while i < len(order):                                  # average ranks for ties
        j = i
        while j + 1 < len(order) and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    direction = sum(r if x > 0 else -r for r, x in zip(ranks, d))
    return float(wilcoxon(d, alternative="two-sided").pvalue), direction


def _holm(ps: list[float]) -> list[float]:
    order, adj, run = sorted(range(len(ps)), key=lambda i: ps[i]), [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def resolve(arrows_: list[dict], id_cells: dict, keep: set | None = None, alpha: float = ALPHA) -> dict:
    """Step 7. arrows_: the combined graph (arrows of Step 6); id_cells: cells(..., ids=True)["chain"].
    Every knot is resolved by the narratives' evidence: all pairs of candidates compared by a two-sided Wilcoxon
    test on the per-narrative differences, Holm-adjusted per knot; a candidate is ruled out if another is
    significantly better; the main candidate is the one with the largest evidence left. Returns the main graph,
    the settled arrows and, per knot, its candidates and tests."""
    arr = [a for a in arrows_ if keep is None or (a["head"] in keep and a["tail"] in keep)]
    edges = {(a["head"], a["tail"]) for a in arr}
    stories = sorted({j for prs in id_cells.values() for j, _ in prs})
    ell = {e: collections.defaultdict(float) for e in edges}     # per arrow, per story: evidence (0 if not asked)
    for e in edges:
        for j, pr in id_cells.get(e, []):
            ell[e][j] = evidence(pr)
    E = lambda F: sum(sum(ell[e].values()) for e in F)
    kn_out, settled, main = [], set(edges), set(edges)
    for comp in knots(edges):
        ka = sorted(e for e in edges if e[0] in comp and e[1] in comp)
        settled -= set(ka); main -= set(ka)
        info = {"concepts": sorted(comp), "arrows": [list(e) for e in ka]}
        if len(ka) > MAX_KNOT_ARROWS:
            info["too_large"] = True                     # reported; its arrows are left out of the main graph
            kn_out.append(info); continue
        cands = sorted(candidates(ka), key=lambda F: sorted(F))
        tests, ps = [], []
        for x in range(len(cands)):
            for y in range(x + 1, len(cands)):
                C, D = cands[x], cands[y]
                d = [sum(ell[e][j] for e in C - D) - sum(ell[e][j] for e in D - C) for j in stories]
                p, direction = _wilcoxon(d)
                tests.append({"pair": [x, y], "p": p, "better": x if direction > 0 else (y if direction < 0 else None)})
                ps.append(p)
        for t, pa in zip(tests, _holm(ps) if ps else []):
            t["p_holm"] = pa
        ruled = {(t["pair"][1] if t["better"] == t["pair"][0] else t["pair"][0])
                 for t in tests if t["p_holm"] < alpha and t["better"] is not None}
        conf = [i for i in range(len(cands)) if i not in ruled] or list(range(len(cands)))
        b = max(conf, key=lambda i: (E(cands[i]), -i))
        common = set.intersection(*(set(cands[i]) for i in conf))
        settled |= common; main |= set(cands[b])
        info.update({"candidates": [{"arrows": sorted(map(list, F)), "evidence": round(E(F), 3), "in_set": i in conf}
                                    for i, F in enumerate(cands)],
                     "main": b, "settled": len(conf) == 1, "tests": tests})
        kn_out.append(info)
    return {"main_arrows": [a for a in arr if (a["head"], a["tail"]) in main],
            "settled": sorted(map(list, settled)), "knots": kn_out}


def cond_prob(table: dict, x: str, y: str) -> tuple[int, int]:
    """P(y = 1 | x = 1) in the stories: stories where x and y happened / stories where x happened (y known)."""
    rows = [v for v in table.values() if v.get(x) == 1 and v.get(y) is not None]
    return sum(r[y] == 1 for r in rows), len(rows)


def model(arrows_: list[dict], table: dict, keep: set | None = None) -> dict:
    """The causal model over the graph (calc_model.build): parents, P(node = 1 | parents) per parent setting
    that occurs, its count, the node's marginal, and the arrows left out because they would close a cycle."""
    arr = [a for a in arrows_ if keep is None or (a["head"] in keep and a["tail"] in keep)]
    nodes = sorted({x for a in arr for x in (a["head"], a["tail"])})
    parents = {c: [] for c in nodes}
    removed = []

    def reaches(u, v):
        stack, seen = [u], set()
        while stack:
            w = stack.pop()
            if w == v:
                return True
            if w in seen:
                continue
            seen.add(w)
            stack += [c for c in nodes if w in parents[c]]
        return False
    for a in sorted(arr, key=lambda a: -a["estimate"]):
        if reaches(a["tail"], a["head"]):
            removed.append([a["head"], a["tail"]])
        else:
            parents[a["tail"]].append(a["head"])
    rows = list(table.values())
    p, cnt, marg = {}, {}, {}
    for c in nodes:
        known = [r for r in rows if r.get(c) is not None]
        marg[c] = sum(r[c] == 1 for r in known) / len(known) if known else 0.0
        tally = {}
        for r in known:
            if all(r.get(q) in (0, 1) for q in parents[c]):
                key = "".join(str(r[q]) for q in parents[c])
                k, n = tally.get(key, (0, 0))
                tally[key] = (k + (r[c] == 1), n + 1)
        p[c] = {key: k / n for key, (k, n) in tally.items()}
        cnt[c] = {key: n for key, (k, n) in tally.items()}
    return {"parents": parents, "p": p, "n": cnt, "marginal": marg, "removed": removed}


def aggregate(outdir: Path, graphs: list[dict]) -> dict:
    abst = json.loads((outdir / "abstraction.json").read_text())
    imp = json.loads((outdir / "implied.json").read_text())
    cl = cells(outdir, abst, graphs)
    matrix = {v: [{"head": x, "tail": y, "n": len(prs),
                   "average": {r: round(sum(p[r] for p in prs) / len(prs), 4) for r in ANSWERS}}
                  for (x, y), prs in c.items()] for v, c in cl.items()}
    table = {j: d["values"] for j, d in imp.items()}
    arr = arrows(cl["chain"])
    res7 = resolve(arr, cells(outdir, abst, graphs, ids=True)["chain"])          # Step 7 (all concepts)
    for a in arr:
        k, n = cond_prob(table, a["head"], a["tail"])
        a["p_tail_given_head"] = round(k / n, 4) if n else None
        a["n_head"] = n
    return {"usage": usage(outdir, abst, graphs),
            "table": table,
            "matrix": matrix,
            "network": {"nodes": [{"id": c, "kind": d["kind"]} for c, d in abst["concepts"].items()],
                        "arrows": arr, "arrows_with_influence": arrows(cl["all"]), "resolved": res7},
            "model": model(res7["main_arrows"], table)}
