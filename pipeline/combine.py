"""Section 2, Steps 4-6 and the output (see method_Oct.tex).

  Step 4  implied   (Kev)  for each narrative, every event node it names and every event
                           concept it does not mention: Kev reads the text, happened (1) /
                           not happened (0) / cannot tell (missing). Being named is not
                           happening (p8: "the eviction never happened"). A named concept is
                           1 if any of its nodes happened, 0 if all did not, else missing.
                           Participant concepts are present (1) or absent (0).
  Step 5  imputed   (MICE) the still-missing cells of the narrative x concept table are
                           imputed M = 30 times (draws from the predictive distribution);
                           arrows to or from an imputed concept are unknown.
  Step 6  combine          random-effects meta-analysis (DerSimonian-Laird) per ordered pair
                           (a, b) and answer r, on the log-odds scale: each narrative where
                           a -> b was asked and neither concept is imputed gives y_j = mean
                           logit over the R repeats and v_j = Var/R (floor 1e-4); tau^2 from
                           Cochran's Q; weights 1/(v_j + tau^2); estimate and 95% interval
                           back-transformed. The displayed answer has the highest estimate.
  Output  causal Bayesian network: arrows = pairs displayed as enables or blocks;
          P(b | parents of b) counted on each of the M tables, combined with Rubin's rules.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import product

from . import prompts
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
        values = {c: 1 for c in present if kind[c] == "participant"}   # participants: present or absent
        for c, k in kind.items():
            if c not in present and k == "participant":
                values[c] = 0
        named = [v for v in g["nodes"] if (j, v) in of and kind[of[(j, v)]] == "event"]
        log(f"[{j}] {len(named)} named event node(s)")
        node_val, node_ans = {}, {}
        if named:
            ans = kev.ask(text, {f"n{i}": prompts.kev_happened_question(v) for i, v in enumerate(named)})
            for i, v in enumerate(named):
                a = ans[f"n{i}"]
                node_val[v] = {"happened": 1, "not happened": 0}.get(a["choice"])
                node_ans[v] = {"choice": a["choice"], "probabilities": a["probabilities"]}
                pr = " ".join(f"{k}={x:.2f}" for k, x in a["probabilities"].items())
                log(f"  node {v!r}: {a['choice']}   ({pr})")
        for c in present:
            if kind[c] != "event":
                continue
            vs = [node_val[v] for v in named if of[(j, v)] == c]
            values[c] = 1 if 1 in vs else (0 if vs and all(x == 0 for x in vs) else None)
        missing = [c for c, k in kind.items() if k == "event" and c not in present]
        log(f"  {len(missing)} missing event concept(s)")
        if missing:
            ans = kev.ask(text, {f"c{i}": prompts.kev_implied_question(c) for i, c in enumerate(missing)})
            for i, c in enumerate(missing):
                a = ans[f"c{i}"]
                values[c] = {"happened": 1, "not happened": 0}.get(a["choice"])
                pr = " ".join(f"{k}={v:.2f}" for k, v in a["probabilities"].items())
                log(f"  {c!r}: {a['choice']}   ({pr})")
        out[j] = {"values": values, "nodes": node_ans}
    return out


# ---------------------------------------------------------------- Step 5

M_IMPUTATIONS = 30


def impute(table: dict[str, dict[str, int | None]], concepts: list[str], m: int = M_IMPUTATIONS):
    """MICE on the narrative x concept table, m times with draws from the predictive
    distribution. Returns the m completed tables and the set of imputed (narrative, concept)
    cells. Columns with no known value stay missing."""
    import numpy as np
    narr = list(table)
    X = np.array([[np.nan if table[j].get(c) is None else table[j][c] for c in concepts] for j in narr],
                 dtype=float)
    miss = {(narr[r], concepts[k]) for r, k in zip(*np.where(np.isnan(X)))}
    known_cols = [k for k in range(len(concepts)) if not np.all(np.isnan(X[:, k]))]
    tables = []
    for i in range(m):
        Xi = X.copy()
        if miss and known_cols:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
            filled = IterativeImputer(max_iter=30, random_state=i, sample_posterior=True, min_value=0,
                                      max_value=1, keep_empty_features=True).fit_transform(Xi[:, known_cols])
            Xi[:, known_cols] = (filled >= 0.5).astype(float)
        tables.append({j: {c: (None if np.isnan(Xi[r, k]) else int(Xi[r, k])) for k, c in enumerate(concepts)}
                       for r, j in enumerate(narr)})
    imputed = {(j, c) for j, c in miss if tables[0][j][c] is not None}
    return tables, imputed


# ---------------------------------------------------------------- Step 6

EPS, V_MIN = 1e-3, 1e-4


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def _expit(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def meta(reps_per_narrative: list[list[float]]) -> dict:
    """Random-effects meta-analysis (DerSimonian-Laird) on the log-odds scale. Input: per
    narrative, the R repeat probabilities for one answer."""
    ys, vs = [], []
    for reps in reps_per_narrative:
        ls = [_logit(p) for p in reps]
        y = sum(ls) / len(ls)
        var = sum((x - y) ** 2 for x in ls) / (len(ls) - 1) if len(ls) > 1 else 0.0
        ys.append(y)
        vs.append(max(var / len(ls), V_MIN))
    n = len(ys)
    w = [1 / v for v in vs]
    ybar = sum(wi * yi for wi, yi in zip(w, ys)) / sum(w)
    Q = sum(wi * (yi - ybar) ** 2 for wi, yi in zip(w, ys))
    denom = sum(w) - sum(wi * wi for wi in w) / sum(w)
    tau2 = max(0.0, (Q - (n - 1)) / denom) if n > 1 and denom > 0 else 0.0
    ws = [1 / (v + tau2) for v in vs]
    mu = sum(wi * yi for wi, yi in zip(ws, ys)) / sum(ws)
    se = 1 / math.sqrt(sum(ws))
    return {"estimate": round(_expit(mu), 4), "ci95": [round(_expit(mu - 1.96 * se), 4), round(_expit(mu + 1.96 * se), 4)],
            "tau2": round(tau2, 4), "v_mean": round(sum(vs) / n, 4), "n": n}


def rubin_cpd(qs: list[float], us: list[float]) -> dict:
    """Rubin's rules over the m completed tables for one conditional probability."""
    m = len(qs)
    qbar = sum(qs) / m
    ubar = sum(us) / m
    B = sum((q - qbar) ** 2 for q in qs) / (m - 1) if m > 1 else 0.0
    T = ubar + (1 + 1 / m) * B
    h = 1.96 * math.sqrt(T)
    return {"p_1": round(qbar, 4), "U": round(ubar, 4), "B": round(B, 4), "T": round(T, 4),
            "ci95": [round(max(0.0, qbar - h), 4), round(min(1.0, qbar + h), 4)]}


def combine(graphs: list[dict], abst: dict, imp: dict, log=print) -> dict:
    of, kind = concept_map(abst)
    concepts = list(kind)
    narr = [g["id"] for g in graphs]

    # Step 5: completed table
    table = {j: {c: imp[j]["values"].get(c) for c in concepts} for j in narr}
    tables, imputed = impute(table, concepts)
    done = tables[0]
    log(f"STEP 5  imputed {len(imputed)} cell(s) x {len(tables)} imputations: {sorted(imputed)}")

    # Step 6: per narrative, the probabilities of each concept-level pair
    per_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for g in graphs:
        j = g["id"]
        local = {(j, v): of[(j, v)] for v in g["nodes"] if (j, v) in of}
        best: dict[tuple[str, str], dict] = {}
        for a in g.get("asked", []):
            ca, cb = local.get((j, a["head"])), local.get((j, a["tail"]))
            if ca is None or cb is None or ca == cb:
                continue
            if (j, ca) in imputed or (j, cb) in imputed:
                continue                                     # arrows of imputed concepts are unknown
            pr = {r: a["probs"].get(r, 0.0) for r in ANSWERS}
            reps = a.get("reps") or [a["probs"]]
            if (ca, cb) not in best or 1 - pr["none"] > 1 - best[(ca, cb)][0]["none"]:
                best[(ca, cb)] = (pr, reps)
        for pair, pr in best.items():
            per_pair[pair].append(pr)

    pairs = []
    for (a, b), prs in per_pair.items():
        stats = {r: meta([[x.get(r, 0.0) for x in reps] for _, reps in prs]) for r in ANSWERS}
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
            qs, us, ns = [], [], []
            for t in tables:
                match = [j for j in narr if t[j][b] is not None and all(t[j][x] == v for x, v in zip(pa, cfg))]
                if match:
                    q = sum(t[j][b] for j in match) / len(match)
                    qs.append(q)
                    us.append(q * (1 - q) / len(match))
                    ns.append(len(match))
            if qs:
                rows.append({"parents": dict(zip(pa, cfg)), **rubin_cpd(qs, us),
                             "n": round(sum(ns) / len(ns), 2), "tables": len(qs)})
        cpds[b] = {"parents": pa, "rows": rows}

    return {"table": done, "tables": tables, "imputed": sorted(imputed), "pairs": pairs,
            "network": {"nodes": [{"id": c, "kind": kind[c]} for c in concepts],
                        "arrows": [{"head": p["head"], "tail": p["tail"], "rel": p["displayed"],
                                    "estimate": p["answers"][p["displayed"]]["estimate"],
                                    "ci95": p["answers"][p["displayed"]]["ci95"], "n": p["n"]}
                                   for p in arrows],
                        "cpds": cpds}}
