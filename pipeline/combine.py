"""Section 2 helpers of the agreed method (method_Oct.tex).

  implied   Step 3 (decision model): for each story, every event node it names and every event
            group it does not name: happened (1) / not happened (0) / cannot tell (missing).
            Being named is not happening ("the eviction never happened"). A named group is 1 if
            any of its nodes happened, 0 if all did not, else missing. Participant groups are
            present (1) or absent (0). Cannot tell stays missing: there is no imputation.
  meta      the 95% interval of an arrow: random-effects meta-analysis (DerSimonian-Laird) on
            the log-odds scale, one answer per story.
"""

from __future__ import annotations

import math

from . import prompts
from .kev_client import KevClient

ANSWERS = ("enables", "blocks", "none")


def concept_map(abst: dict) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """(narrative, node) -> concept, and concept -> kind."""
    of, kind = {}, {}
    for c, d in abst["concepts"].items():
        kind[c] = d["kind"]
        for m in d["members"]:
            of[(m[0], m[1])] = c
    return of, kind


def sign_map(abst: dict) -> dict[tuple[str, str], int]:
    """(narrative, node) -> +1, or -1 if the node joined its concept as the opposite."""
    return {(m[0], m[1]): (m[2] if len(m) > 2 else 1) for d in abst["concepts"].values() for m in d["members"]}


def _flip(pr: dict) -> dict:
    out = dict(pr)
    out["enables"], out["blocks"] = pr.get("blocks", 0.0), pr.get("enables", 0.0)
    return out


def implied(kev: KevClient, graphs: list[dict], abst: dict, seed: int = 0, log=print) -> dict:
    of, kind = concept_map(abst)
    sign = sign_map(abst)
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
                if node_val[v] is not None and sign.get((j, v), 1) < 0:
                    node_val[v] = 1 - node_val[v]                      # opposite member: value flipped
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
