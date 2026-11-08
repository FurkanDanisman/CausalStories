#!/usr/bin/env python3
"""Score one pipeline run against the benchmark truth (the two tables in three_worlds.md §9).

Node matching is done by a separate LLM judge on OpenRouter (env JUDGE; default Qwen3 235B), never by hand:
- individual graphs: each story's extracted nodes are matched to that story's own true nodes;
- aggregated graph: the groups kept at the 10% usage cut are matched to the world's nodes.
Several extracted nodes / groups may map to one true node (merged); unmapped ones are extra nodes.

Usage: python benchmark/evaluate_run.py homelessness pipeline/out_api_v3_homelessness [5,10,15,20,25,30]
Writes <run>/evaluation_<judge>_cutXX.json and prints both tables. Judge answers are cached in <run>/judge_cache_<judge>.json."""
import itertools, json, math, os, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
PIPE = HERE.parent if (HERE.parent / "run.py").exists() else HERE.parent / "pipeline"   # repo root (laptop or cluster)
JUDGE = os.environ.get("JUDGE", "qwen/qwen3-235b-a22b-2507")   # set JUDGE=<openrouter model> to change
SLUG = os.environ.get("JUDGE_SLUG") or re.sub(r"[^a-z0-9]+", "-", JUDGE.rstrip("/").split("/")[-1].lower())
POS = {"carries out", "enables", "risk"}          # true arrow types read as "+" (enables)
NEG = {"stopped", "failed"}                       # true arrow types read as "-" (blocks)
CUT = 0.10


JUDGE_URL = os.environ.get("JUDGE_URL", "https://openrouter.ai/api/v1/chat/completions")   # local: vLLM server


def env_key():
    if "openrouter.ai" not in JUDGE_URL:
        return "EMPTY"                                        # a local vLLM judge needs no key
    for line in (PIPE / ".env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    return os.environ["OPENROUTER_API_KEY"]


JUDGE_USAGE = {"calls": 0, "usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}


def judge_call(prompt, key):
    body = json.dumps({"model": JUDGE, "temperature": 0, "response_format": {"type": "json_object"},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for attempt in range(5):
        try:
            req = urllib.request.Request(JUDGE_URL, data=body,
                                         headers={"authorization": f"Bearer {key}", "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read())
            u = resp.get("usage", {})
            JUDGE_USAGE["calls"] += 1; JUDGE_USAGE["usd"] += u.get("cost", 0) or 0
            JUDGE_USAGE["prompt_tokens"] += u.get("prompt_tokens", 0); JUDGE_USAGE["completion_tokens"] += u.get("completion_tokens", 0)
            txt = resp["choices"][0]["message"]["content"]
            obj = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
            m = obj.get("map", obj)                       # some judges drop the {"map": ...} wrapper
            if not isinstance(m, dict) or not all(isinstance(v, str) for v in m.values()):
                raise ValueError("bad judge answer")
            return {"map": m}
        except Exception:
            if attempt == 4:
                raise


def match_prompt(context, true_nodes, extracted):
    lines = "\n".join(f"- {n}" for n in true_nodes)
    ex = "\n".join(f"- {e}" for e in extracted)
    return (f"{context}\n\nTRUE NODES (the reference names):\n{lines}\n\nEXTRACTED NODES:\n{ex}\n\n"
            "For each extracted node, decide which true node it means, or \"none\" if it means none of them. "
            "Uppercase true nodes are participants (people or organisations who act); the others are events "
            "or states. An extracted node matches a true node only if it refers to the same thing in this "
            "context (wording may differ: \"sleeping on the street\" = homelessness; \"my employer\" = EMPLOYER). "
            "A participant is not the same as the event it carries out (e.g. \"family member\" is FAMILY MEMBER, "
            "\"offering shelter\" is the event). Several extracted nodes may map to the same true node. "
            "Answer only JSON: {\"map\": {\"<extracted node>\": \"<true node or none>\", ...}}")


def signed(edges):
    out = {}
    for h, t, s, w in edges:
        if h == t:
            continue
        k = (h, t)
        if k not in out or w > out[k][1]:
            out[k] = (s, w)
    return {k: v[0] for k, v in out.items()}, {k: v[1] for k, v in out.items()}


def shd(T, E):
    res = {"missing": 0, "extra": 0, "reversed": 0, "sign": 0}
    for p in {frozenset(k) for k in T} | {frozenset(k) for k in E}:
        a, b = tuple(p)
        t = [(k, T[k]) for k in ((a, b), (b, a)) if k in T]
        e = [(k, E[k]) for k in ((a, b), (b, a)) if k in E]
        if t and not e: res["missing"] += 1
        elif e and not t: res["extra"] += 1
        elif t[0][0] != e[0][0]: res["reversed"] += 1
        elif t[0][1] != e[0][1]: res["sign"] += 1
    return res


def f1(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def gmean(xs):
    return 0.0 if min(xs) <= 0 else math.exp(sum(math.log(x) for x in xs) / len(xs))


# ---------- SID helpers ----------
def anc(E, S):
    out = set(S); ch = True
    while ch:
        ch = False
        for u, v in E:
            if v in out and u not in out: out.add(u); ch = True
    return out


def desc(E, x):
    out, st = set(), [x]
    while st:
        n = st.pop()
        for u, v in E:
            if u == n and v not in out: out.add(v); st.append(v)
    return out


def dsep(E, X, Y, Z):
    A = anc(E, X | Y | Z); sub = {(u, v) for u, v in E if u in A and v in A}
    und = {frozenset(e) for e in sub}
    for c in A:
        ps = [u for u, v in sub if v == c]
        und |= {frozenset(pq) for pq in itertools.combinations(ps, 2)}
    seen, st = set(X), list(X)
    while st:
        n = st.pop()
        for e in und:
            if n in e and len(e) == 2:
                m = next(iter(e - {n}))
                if m in Z or m in seen: continue
                if m in Y: return False
                seen.add(m); st.append(m)
    return True


def break_cycles(E, w):
    E = set(E)
    while True:
        cyc = None
        for u, v in E:
            if u in desc(E, v):
                cyc = True; break
        if not cyc: return E
        on = [(a, b) for a, b in E if a in desc(E, b)]          # arrows that lie on a cycle
        E.remove(min(on, key=lambda k: w.get(k, 0)))


def contract(E, node):
    ins = [u for u, v in E if v == node]; outs = [v for u, v in E if u == node]
    return {(u, v) for u, v in E if node not in (u, v)} | {(u, v) for u in ins for v in outs if u != v}


def sid(true_E, est_E, nodes):
    bad = []
    for i in nodes:
        Pa = {u for u, v in est_E if v == i}; Di = desc(true_E, i)
        for j in nodes:
            if i == j: continue
            if j in Pa: wrong = j in Di
            elif Pa & Di: wrong = True
            else: wrong = not dsep({(u, v) for u, v in true_E if u != i}, {i}, {j}, Pa)
            if wrong: bad.append((i, j))
    return bad


def topo(nodes, E):
    order, left = [], set(nodes)
    while left:
        free = sorted(n for n in left if not any(u in left and v == n for u, v in E))
        order += free; left -= set(free)
    return order


def true_values(story, nodes):
    """A story's true events, from the generator's record (story_v2.lived_graph): both ends of an enabling or
    carrying-out arrow happened; a blocker happened; the target of a 'stopped' or 'risk' arrow did not happen,
    the target of a 'failed' blocker did; a distractor in the story happened. Events not in the story: None."""
    v = {n: None for n in nodes}
    for u, t, k in story["truth"]["arrows"]:
        v[u] = 1
        v[t] = 0 if k in ("stopped", "risk") else 1
    for d in story["truth"].get("distractors", []):
        if d in v: v[d] = 1
    return v


def kl2(p, q):
    """KL between Bernoulli(p) and Bernoulli(q); exactly 0 when they are equal."""
    if p == q: return 0.0
    q = min(max(q, 1e-6), 1 - 1e-6)
    return sum(pp * math.log(pp / qq) for pp, qq in ((p, q), (1 - p, 1 - q)) if pp > 0)


def main(world, run, cut=None):
    cut = CUT if cut is None else cut
    run = Path(run); key = env_key()
    mod = {"homelessness": "world_homelessness", "career": "world_career", "mental_health": "world_mental"}[world]
    sys.path.insert(0, str(HERE)); W = __import__(mod)
    stories = {s["id"]: s for s in json.load(open(HERE / "stories_v3" / f"{world}.json"))}
    cache_p = run / f"judge_cache_{SLUG}.json"
    cache = json.load(open(cache_p)) if cache_p.exists() else {}

    # ---------------- Table 1: individual graphs ----------------
    graphs = {}
    for f in run.glob("*__gemma-kev.chain.json"):
        if f.name.startswith("._"): continue          # macOS metadata files, not graphs
        g = json.load(open(f)); graphs[g["id"]] = g

    def one(sid_):
        s = stories[sid_]; g = graphs.get(sid_)
        tnodes = sorted({n for a in s["truth"]["arrows"] for n in a[:2]})
        if g is None:
            return sid_, None
        ext = list(g["nodes"])
        ck = f"ind:{sid_}"
        if ck not in cache:
            cache[ck] = judge_call(match_prompt(f"STORY:\n{s['text']}", tnodes, ext), key)["map"]
        m = {e: (v if v in tnodes else None) for e, v in cache[ck].items()}
        mapped = [e for e in ext if m.get(e)]
        P = len(mapped) / len(ext) if ext else 0.0
        R = len({m[e] for e in mapped}) / len(tnodes)
        T, _ = signed([(h, t, "+" if k in POS else "-", 1) for h, t, k in s["truth"]["arrows"]])
        kinds = g.get("kinds") or {}                     # influence arrows (event -> participant) are a separate
        edges = [e for e in g["edges"] if kinds.get(e["tail"], "event") != "participant"]   # layer: never scored
        E, _ = signed([(m.get(e["head"]) or f"x:{e['head']}", m.get(e["tail"]) or f"x:{e['tail']}",
                        "+" if e["rel"] == "enables" else "-", e.get("prob", 0)) for e in edges])
        d = shd(T, E); s_ = sum(d.values())
        arrows = 1 - s_ / (len(T) + len(E)) if (len(T) + len(E)) else 1.0
        node = f1(P, R)
        return sid_, {"group": s["group"], "node_P": P, "node_R": R, "node_F1": node, "shd": d,
                      "true_arrows": len(T), "est_arrows": len(E), "arrows": arrows,
                      "score": gmean([node, arrows]), "map": m}

    with ThreadPoolExecutor(8) as ex:
        ind = dict(ex.map(one, sorted(stories)))
    json.dump(cache, open(cache_p, "w"), indent=1)
    scored = [v for v in ind.values() if v]

    def mean(k, rows=scored): return sum(r[k] for r in rows) / len(rows) if rows else None

    # ---------------- Table 2: aggregated graph ----------------
    if not (run / "abstraction.json").exists():           # story graphs only (C1-C5): Table 1, no aggregated graph
        if os.environ.get("JUDGE_ONLY"):
            print(f"judged {run.name} (individual graphs only)"); return
        S = {g: {k: mean(k, [r for r in scored if g == "all" or r["group"] == g]) for k in ("node_P", "node_R", "node_F1", "arrows", "score")}
             for g in ("all", "outcome", "stopped")}
        json.dump({"world": world, "run": str(run), "judge": JUDGE, "individual": ind, "aggregated": None, "individual_summary": S},
                  open(run / f"evaluation_{SLUG}_cut{round(cut * 100):02d}.json", "w"), indent=1)
        print(f"Table 1 (individual, {len(scored)} stories): story score {S['all']['score']:.2f}"); return
    abst = json.load(open(run / "abstraction.json"))
    n_st = len(graphs) or len(stories)                   # aggregated-only methods have no story graphs
    if graphs and all((run / f"{j}__connect.json").exists() for j in graphs):   # our method: agreed usage rule
        sys.path.insert(0, str(PIPE)); from pipeline import matrix   # (node has an arrow in the connected graph)
        usage = matrix.usage(run, abst, list(graphs.values()))
    else:                                               # other aggregations: share of stories naming the group
        usage = {c: len({m[0] for m in v["members"]}) / n_st for c, v in abst["concepts"].items()}
    kept = sorted(c for c, u in usage.items() if u >= cut)
    TN = sorted(set(W.ROOT) - set(W.DISTRACT) | set(W.MECH))
    ck = "agg:" + "|".join(kept)
    if ck not in cache:
        desc_ = [f"{c}  (e.g. " + "; ".join(sorted({m[1] for m in abst['concepts'][c]['members']})[:4]) + ")" for c in kept]
        cache[ck] = judge_call(match_prompt(f"These are groups of nodes extracted from {n_st} first-person stories "
                                            f"about {world.replace('_', ' ')}; each group is shown with example "
                                            "wordings. Match each group by the name before the brackets.",
                                            TN, desc_), key)["map"]
        json.dump(cache, open(cache_p, "w"), indent=1)
    if os.environ.get("JUDGE_ONLY"):                      # GPU part only (cluster): judge answers cached, scored later
        print(f"judged {run.name} at cut {cut:.0%} (cache only)")
        return
    net = json.load(open(run / "network.json"))
    gm = {}
    for k, v in cache[ck].items():
        name = k.split("  (e.g.")[0].strip()
        gm[name] = v if v in TN else None
    gm = {c: gm.get(c) for c in kept}
    mapped_true = {v for v in gm.values() if v}
    est_nodes = {gm[c] or f"x:{c}" for c in kept}
    P = len([c for c in kept if gm[c]]) / len(kept); R = len(mapped_true) / len(TN)
    T, _ = signed([(u, v, "+", 1) for v, (en, bl) in W.MECH.items() for u in en] +
                  [(u, v, "-", 1) for v, (en, bl) in W.MECH.items() for u in bl])
    kind = {c: d.get("kind", "event") for c, d in abst["concepts"].items()}
    arr = [a for a in net["network"]["arrows"] if a["head"] in kept and a["tail"] in kept
           and kind.get(a["tail"], "event") != "participant"]          # influence arrows are never scored
    knots7 = None
    if graphs and all((run / f"{j}__connect.json").exists() for j in graphs):   # our aggregation (ours, C1-C4):
        res7 = matrix.resolve(arr, matrix.cells(run, abst, list(graphs.values()), ids=True)["chain"], set(kept))
        arr, knots7 = res7["main_arrows"], res7["knots"]                   # score Step 7's main graph at this cut
    E, Ew = signed([(gm[a["head"]] or f"x:{a['head']}", gm[a["tail"]] or f"x:{a['tail']}",
                     "+" if a["rel"] == "enables" else "-", a["estimate"]) for a in arr])
    d = shd(T, E); s_ = sum(d.values())
    arrows_norm = 1 - s_ / (len(T) + len(E))
    # SID on the true node set: extra nodes contracted, cycles broken by dropping the weakest arrow
    Eg = break_cycles(set(E), Ew)
    for x in [n for n in est_nodes if n.startswith("x:")]:
        Eg = contract(Eg, x)
    Eg = break_cycles({(u, v) for u, v in Eg if u in TN and v in TN}, Ew)
    Etrue = set(T)
    bad = sid(Etrue, Eg, TN)
    sid_norm = 1 - len(bad) / (len(TN) * (len(TN) - 1))
    # the method's own story table, over its kept groups (judge-matched to true nodes)
    tab = net["table"]
    def val(row, n):                                          # merged node = any of its groups happened;
        vs = [row.get(c) for c in kept if (gm[c] or f"x:{c}") == n]   # None = cannot tell (no imputation)
        return 1 if 1 in vs else (0 if vs and all(v == 0 for v in vs) else None)
    rows = [{n: val(r, n) for n in est_nodes} for r in tab.values()]
    def p1(n, cond=None):                                     # counted only over stories where all are known
        rs = [r for r in rows if r[n] is not None and (cond is None or all(r[k] == v for k, v in cond.items()))]
        return (sum(r[n] for r in rs) / len(rs)) if rs else 0.0
    # KL of the arrows' conditional probabilities (three_worlds.md, section 9), which is what the methods output:
    # for every true arrow u -> v, the true P(v = 1 | u = 1), counted from the TRUE story table (each story's true
    # events, from the generator), against the method's P(v = 1 | u = 1) for its arrow u -> v, counted the same
    # way from its OWN story table (method_Oct.tex, Step 8, per arrow). A true arrow the method does not have
    # (or a node it misses) gets 0.5; a method without a story table uses its arrow's own number. KL is the mean
    # over the true arrows; a method with the true graph that reads every story right scores exactly 0.
    def cond(table, v, u):                                    # P(v = 1 | u = 1) over rows with u = 1 and v known
        rs = [r for r in table if r.get(u) == 1 and r.get(v) is not None]
        return sum(r[v] for r in rs) / len(rs) if rs else None
    true_rows = [true_values(stories[j], TN) for j in stories]
    KL_MODEL = "per-arrow P(child | parent), counted tables" if rows else "per-arrow, the arrow's own number (no story table)"
    kl_nodes = {}                                             # keyed "u -> v"
    for v, (en, bl) in W.MECH.items():
        for u in list(en) + list(bl):
            p_ref = cond(true_rows, v, u)
            if p_ref is None: continue                        # no story shows u happening with v known
            if (u, v) not in E: q = 0.5
            elif rows: q = cond(rows, v, u); q = 0.5 if q is None else q
            else: q = Ew[(u, v)]
            kl_nodes[f"{u} -> {v}"] = kl2(p_ref, q)
    kl = sum(kl_nodes.values()) / len(kl_nodes) if kl_nodes else 0.0
    agg = {"cut": cut, "kept_groups": {c: {"usage": round(usage[c], 3), "match": gm[c]} for c in kept},
           "node_P": P, "node_R": R, "node_F1": f1(P, R), "shd": d, "true_arrows": len(T), "est_arrows": len(E),
           "arrows": arrows_norm, "sid_wrong": len(bad), "sid_pairs": len(TN) * (len(TN) - 1), "sid": sid_norm,
           "kl": kl, "kl_by_arrow": kl_nodes, "kl_score": math.exp(-kl), "kl_model": KL_MODEL, "step7_knots": knots7,
           "arrows_list": [{"head": h, "tail": t, "sign": sg, "estimate": Ew[(h, t)]} for (h, t), sg in E.items()]}
    agg["score"] = gmean([agg["node_F1"], arrows_norm, sid_norm, agg["kl_score"]])
    out = {"world": world, "run": str(run), "judge": JUDGE, "judge_usage_so_far": dict(JUDGE_USAGE), "individual": ind, "aggregated": agg,
           "individual_summary": {g: {k: mean(k, [r for r in scored if g == "all" or r["group"] == g])
                                      for k in ("node_P", "node_R", "node_F1", "arrows", "score")}
                                  for g in ("all", "outcome", "stopped")}}
    json.dump(out, open(run / f"evaluation_{SLUG}_cut{round(cut * 100):02d}.json", "w"), indent=1)
    S = out["individual_summary"]["all"]
    if scored:
        print(f"Table 1 (individual, {len(scored)} stories): node P {S['node_P']:.2f} R {S['node_R']:.2f} "
              f"F1 {S['node_F1']:.2f} | arrows {S['arrows']:.2f} | story score {S['score']:.2f}")
    else:
        print("Table 1: no story graphs (aggregated-only method)")
    print(f"Table 2 (aggregated): node P {P:.2f} R {R:.2f} F1 {agg['node_F1']:.2f} | SHD {s_} {d} -> {arrows_norm:.2f} "
          f"| SID {len(bad)}/{agg['sid_pairs']} -> {sid_norm:.2f} | KL {kl:.3f} -> {agg['kl_score']:.2f} | score {agg['score']:.2f}")


if __name__ == "__main__":
    cuts = [float(c) / 100 for c in sys.argv[3].split(",")] if len(sys.argv) > 3 else [CUT]   # e.g. 5,10,15
    for c in cuts:
        print(f"--- usage cut {c:.0%}")
        main(sys.argv[1], sys.argv[2], c)
