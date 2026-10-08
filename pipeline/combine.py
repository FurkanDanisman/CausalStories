"""Section 2, Steps 4-6 and the output (see method_Oct.tex).

  Step 4  implied   (Kev)  for each narrative and each event concept it does not mention,
                           Kev reads the text: happened (1) / not happened (0) / cannot tell
                           (stays missing). The subgraph extraction is then run again for the
                           narrative with the concepts set to 1 or 0 added as nodes.
                           Participant concepts are present (1) or absent (0).
  Step 5  imputed   (MICE) the still-missing cells of the narrative x concept table are
                           imputed; arrows to or from an imputed concept are unknown.
  Step 6  combine          for every ordered pair of concepts (a, b), each narrative where
                           a -> b was asked and both are known (not imputed) gives Kev's
                           probabilities p_r, r in {enables, blocks, none}. Per answer,
                           Rubin's rules: estimate = mean, W = mean p(1-p), B = variance
                           across narratives, T = W + (1 + 1/n) B, 95% interval =
                           estimate +- 1.96 sqrt(T) clipped to [0, 1]. The displayed answer
                           is the one with the highest estimate.
  Output  causal Bayesian network: arrows = pairs displayed as enables or blocks;
          P(b | parents of b) counted from the completed table.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import product

from . import chain, prompts
from .kev_client import KevClient

ANSWERS = ("enables", "blocks", "none")


def concept_map(abst: dict) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """(narrative, node) -> concept, and concept -> kind."""
    of, kind = {}, {}
    for c, d in abst["concepts"].items():
        kind[c] = d["kind"]
        for j, v in d["members"]:
            of[(j, v)] = c
    return of, kind


# ---------------------------------------------------------------- Step 4

def implied(kev: KevClient, graphs: list[dict], abst: dict, seed: int = 0, log=print) -> dict:
    of, kind = concept_map(abst)
    out = {}
    for g in graphs:
        j, text = g["id"], g["text"]
        present = {of[(j, v)] for v in g["nodes"] if (j, v) in of}
        values = {c: 1 for c in present}
        for c, k in kind.items():
            if c not in present and k == "participant":
                values[c] = 0                                   # participants: present or absent
        missing = [c for c, k in kind.items() if k == "event" and c not in present]
        log(f"[{j}] {len(missing)} missing event concept(s)")
        if missing:
            ans = kev.ask(text, {f"c{i}": prompts.kev_implied_question(c) for i, c in enumerate(missing)})
            for i, c in enumerate(missing):
                a = ans[f"c{i}"]
                values[c] = {"happened": 1, "not happened": 0}.get(a["choice"])
                pr = " ".join(f"{k}={v:.2f}" for k, v in a["probabilities"].items())
                log(f"  {c!r}: {a['choice']}   ({pr})")
        added = [c for c in missing if values.get(c) is not None]
        rerun = None
        if added:
            nodes = list(g["nodes"]) + added
            kinds = dict(g.get("kinds") or {}) | {c: "event" for c in added}
            log(f"  rerun subgraph extraction with {added}")
            rerun = chain.run_chain(kev, text, nodes, seed=seed, kinds=kinds, log=log).to_json()
        out[j] = {"values": values, "added": added, "graph": rerun}
    return out


# ---------------------------------------------------------------- Step 5

def impute(table: dict[str, dict[str, int | None]], concepts: list[str]) -> tuple[dict, set]:
    """MICE on the narrative x concept table. Returns the completed table and the set of
    imputed (narrative, concept) cells. Columns with no known value stay missing."""
    import numpy as np
    narr = list(table)
    X = np.array([[np.nan if table[j].get(c) is None else table[j][c] for c in concepts] for j in narr],
                 dtype=float)
    imputed = {(narr[r], concepts[k]) for r, k in zip(*np.where(np.isnan(X)))}
    known_cols = [k for k in range(len(concepts)) if not np.all(np.isnan(X[:, k]))]
    if imputed and known_cols:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        sub = X[:, known_cols]
        filled = IterativeImputer(max_iter=30, random_state=0, min_value=0, max_value=1,
                                  keep_empty_features=True).fit_transform(sub)
        X[:, known_cols] = (filled >= 0.5).astype(float)
    done = {j: {c: (None if np.isnan(X[r, k]) else int(X[r, k])) for k, c in enumerate(concepts)}
            for r, j in enumerate(narr)}
    imputed = {(j, c) for j, c in imputed if done[j][c] is not None}
    return done, imputed


# ---------------------------------------------------------------- Step 6

def rubin(ps: list[float]) -> dict:
    n = len(ps)
    est = sum(ps) / n
    W = sum(p * (1 - p) for p in ps) / n
    B = sum((p - est) ** 2 for p in ps) / (n - 1) if n > 1 else 0.0
    T = W + (1 + 1 / n) * B if n > 1 else W
    h = 1.96 * math.sqrt(T)
    return {"estimate": round(est, 4), "W": round(W, 4), "B": round(B, 4), "T": round(T, 4),
            "ci95": [round(max(0.0, est - h), 4), round(min(1.0, est + h), 4)]}


def combine(graphs: list[dict], abst: dict, imp: dict, log=print) -> dict:
    of, kind = concept_map(abst)
    concepts = list(kind)
    narr = [g["id"] for g in graphs]

    # Step 5: completed table
    table = {j: {c: imp[j]["values"].get(c) for c in concepts} for j in narr}
    done, imputed = impute(table, concepts)
    log(f"STEP 5  imputed {len(imputed)} cell(s): {sorted(imputed)}")

    # Step 6: per narrative, the probabilities of each concept-level pair
    per_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for g in graphs:
        j = g["id"]
        graph = imp[j]["graph"] or g
        local = {(j, v): of[(j, v)] for v in graph["nodes"] if (j, v) in of}
        local |= {(j, c): c for c in imp[j]["added"]}
        best: dict[tuple[str, str], dict] = {}
        for a in graph.get("asked", []):
            ca, cb = local.get((j, a["head"])), local.get((j, a["tail"]))
            if ca is None or cb is None or ca == cb:
                continue
            if (j, ca) in imputed or (j, cb) in imputed:
                continue                                     # arrows of imputed concepts are unknown
            pr = {r: a["probs"].get(r, 0.0) for r in ANSWERS}
            if (ca, cb) not in best or 1 - pr["none"] > 1 - best[(ca, cb)]["none"]:
                best[(ca, cb)] = pr
        for pair, pr in best.items():
            per_pair[pair].append(pr)

    pairs = []
    for (a, b), prs in per_pair.items():
        stats = {r: rubin([p[r] for p in prs]) for r in ANSWERS}
        shown = max(ANSWERS, key=lambda r: stats[r]["estimate"])
        pairs.append({"head": a, "tail": b, "n": len(prs), "answers": stats, "displayed": shown})
    log(f"STEP 6  {len(pairs)} concept pair(s) combined")

    # Output: causal Bayesian network
    arrows = [p for p in pairs if p["displayed"] != "none"]
    parents: dict[str, list[str]] = defaultdict(list)
    for p in arrows:
        parents[p["tail"]].append(p["head"])
    cpds = {}
    for b in concepts:
        pa = sorted(parents.get(b, []))
        rows = []
        for cfg in product((0, 1), repeat=len(pa)):
            match = [j for j in narr if done[j][b] is not None
                     and all(done[j][x] == v for x, v in zip(pa, cfg))]
            if match:
                rows.append({"parents": dict(zip(pa, cfg)),
                             "p_1": round(sum(done[j][b] for j in match) / len(match), 4),
                             "n": len(match)})
        cpds[b] = {"parents": pa, "rows": rows}

    return {"table": done, "imputed": sorted(imputed), "pairs": pairs,
            "network": {"nodes": [{"id": c, "kind": kind[c]} for c in concepts],
                        "arrows": [{"head": p["head"], "tail": p["tail"], "rel": p["displayed"],
                                    "estimate": p["answers"][p["displayed"]]["estimate"],
                                    "ci95": p["answers"][p["displayed"]]["ci95"], "n": p["n"]}
                                   for p in arrows],
                        "cpds": cpds}}
