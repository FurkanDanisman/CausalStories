#!/usr/bin/env python3
"""Score an estimated aggregated graph against the true world graph (pipeline/world_model.json) with
node precision/recall, SHD (with sign errors), SID (Peters & Buhlmann 2015) and KL (noisy-OR).
Needs a node matching (group -> true node, MAP below), which is not built yet; unmapped groups are extra nodes.
Usage: python benchmark/score_graph.py   (scores the four homelessness aggregated graphs)"""
import itertools, json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import score_pathways as sp

CPD = json.loads((ROOT / "pipeline" / "world_model.json").read_text())["cpd"]
TRUE_NODES = ["crisis", "support", "rent", "eviction", "homelessness", "health"]
TRUE_ARROWS = {("crisis", "rent"): "enables", ("rent", "eviction"): "enables", ("support", "eviction"): "blocks",
               ("eviction", "homelessness"): "enables", ("homelessness", "health"): "enables"}
EPS = 0.01        # noisy-OR leak: chance a node happens with none of its estimated causes

MAP = {}   # group -> true node; to be filled by the node-matching layer (not built yet)


def p_true(st):
    pr = 1.0 if st["crisis"] else 0.0
    pr *= CPD["support"] if st["support"] else 1 - CPD["support"]
    q = CPD["rent_trouble | crisis"] if st["crisis"] else 0.0
    pr *= q if st["rent"] else 1 - q
    q = (CPD["eviction | rent_trouble,support"] if st["support"] else CPD["eviction | rent_trouble,no_support"]) if st["rent"] else 0.0
    pr *= q if st["eviction"] else 1 - q
    q = CPD["homeless | eviction"] if st["eviction"] else 0.0
    pr *= q if st["homelessness"] else 1 - q
    q = CPD["health_decline | homeless"] if st["homelessness"] else 0.0
    return pr * (q if st["health"] else 1 - q)


def merged_graph(edges):
    """Groups -> true nodes; arrows inside one true node dropped; duplicates keep the strongest."""
    best = {}
    for x, y, rel, p in edges:
        u, v = MAP.get(x, x), MAP.get(y, y)
        if u != v and ((u, v) not in best or p > best[(u, v)][1]):
            best[(u, v)] = (rel, p)
    return best


def shd(est):
    """Missing, extra, reversed and wrong-sign arrows; one error per node pair."""
    pairs = {frozenset(k) for k in list(TRUE_ARROWS) + list(est)}
    err = {"missing": 0, "extra": 0, "reversed": 0, "sign": 0}
    for pr in pairs:
        t = [(k, s) for k, s in TRUE_ARROWS.items() if frozenset(k) == pr]
        e = [(k, s[0]) for k, s in est.items() if frozenset(k) == pr]
        if t and not e: err["missing"] += 1
        elif e and not t: err["extra"] += 1
        elif t and e:
            if t[0][0] not in [k for k, _ in e]: err["reversed"] += 1
            elif dict(e)[t[0][0]] != t[0][1]: err["sign"] += 1
    return sum(err.values()), err


def restrict(est):
    """Graph on the true nodes: extra nodes contracted (their parents joined to their children), then
    cycles broken by dropping the weakest arrow."""
    E = {k: v for k, v in est.items()}
    extras = {n for k in E for n in k} - set(TRUE_NODES)
    for x in extras:
        ins = [(u, E[(u, w)]) for (u, w) in list(E) if w == x]
        outs = [(w, E[(u2, w)]) for (u2, w) in list(E) if u2 == x]
        for k in [k for k in E if x in k]:
            del E[k]
        for (u, (r1, p1)) in ins:
            for (w, (r2, p2)) in outs:
                if u != w and (u, w) not in E:
                    E[(u, w)] = ("blocks" if (r1 == "blocks") != (r2 == "blocks") else "enables", min(p1, p2))
    es = [(u, v, r, p) for (u, v), (r, p) in E.items()]
    kept, _ = sp.break_cycles(es)
    return {(u, v): (r, p) for u, v, r, p in kept}


def parents(arrows, n):
    return [u for (u, v) in arrows if v == n]


def descendants(arrows, n):
    out, st = set(), [n]
    while st:
        m = st.pop()
        for (u, v) in arrows:
            if u == m and v not in out:
                out.add(v); st.append(v)
    return out


def dsep(arrows, x, y, Z):
    """d-separation via the moralised ancestral graph."""
    anc, st = set(), [x, y, *Z]
    while st:
        m = st.pop()
        if m in anc: continue
        anc.add(m); st += parents(arrows, m)
    und = {n: set() for n in anc}
    for (u, v) in arrows:
        if u in anc and v in anc:
            und[u].add(v); und[v].add(u)
    for n in anc:
        ps = [p for p in parents(arrows, n) if p in anc]
        for a, b in itertools.combinations(ps, 2):
            und[a].add(b); und[b].add(a)
    seen, st = {x}, [x]
    while st:
        m = st.pop()
        for k in und[m]:
            if k == y: return False
            if k not in seen and k not in Z:
                seen.add(k); st.append(k)
    return True


def valid_adjustment(G, x, y, Z):
    """Generalised adjustment criterion (Shpitser et al. 2010) for P(y | do(x)) in the true DAG G."""
    if y not in descendants(G, x):
        return True if x not in Z else True          # no causal effect: any set gives P(y), see SID text
    on_path = {w for w in descendants(G, x) | {x} if w == y or y in descendants(G, w)} - {x}
    forb = set(on_path)
    for w in on_path:
        forb |= descendants(G, w)
    if set(Z) & forb:
        return False
    pbd = {(u, v) for (u, v) in G if not (u == x and v in on_path)}
    return dsep(pbd, x, y, set(Z))


def sid(G, H):
    """Pairs (x, y) whose P(y | do(x)) is inferred wrongly from H (parent adjustment)."""
    wrong = 0
    for x, y in itertools.permutations(TRUE_NODES, 2):
        if y in parents(H, x):                       # H says x does not affect y
            ok = y not in descendants(G, x)
        else:
            ok = valid_adjustment(G, x, y, parents(H, x))
        wrong += not ok
    return wrong


def story_states(D):
    im = json.loads((D / "implied.json").read_text())
    out = []
    for v in im.values():
        st = {}
        for n in TRUE_NODES:
            vals = [x for g, x in v["values"].items() if MAP.get(g) == n]
            st[n] = 1 if 1 in vals else (0 if vals and all(x == 0 for x in vals) else None)
        out.append(st)
    return out


def kl(H, states):
    """KL(P_true || P_est), P_est = noisy-OR over H with per-arrow probabilities counted from the stories."""
    def freq(n, cond=None):
        rows = [s for s in states if s[n] is not None and all(s[k] == v for k, v in (cond or {}).items())]
        return (sum(s[n] for s in rows) / len(rows)) if rows else None
    clip = lambda p: min(max(p, 1e-3), 1 - 1e-3)
    model = {}
    for n in TRUE_NODES:
        ps = parents(H, n)
        if not ps:
            f = freq(n); model[n] = ("root", clip(f if f is not None else 0.5))
        else:
            en, bl = [], []
            for u in ps:
                pba = freq(n, {u: 1}); pb = freq(n)
                if H[(u, n)][0] == "enables":
                    en.append((u, clip(pba if pba is not None else 0.5)))
                else:
                    bl.append((u, min(max(1 - (pba / pb) if pba is not None and pb else 0.0, 0.0), 0.999)))
            model[n] = ("noisyor", en, bl)
    def p_est(st):
        pr = 1.0
        for n in TRUE_NODES:
            m = model[n]
            if m[0] == "root": q = m[1]
            else:
                f = 1 - EPS
                for u, p in m[1]:
                    if st[u]: f *= 1 - p
                q = 1 - f
                for u, b in m[2]:
                    if st[u]: q *= 1 - b
            pr *= q if st[n] else 1 - q
        return pr
    total = 0.0
    for bits in itertools.product((0, 1), repeat=len(TRUE_NODES)):
        st = dict(zip(TRUE_NODES, bits)); pt = p_true(st)
        if pt > 0:
            total += pt * math.log(pt / max(p_est(st), 1e-300))
    return total


def score(D, th):
    g = sp.combined(D, th)
    groups = sorted({x for e in g["edges"] for x in e[:2]})
    matched = [x for x in groups if x in MAP]
    found = {MAP[x] for x in matched}
    est = merged_graph(g["edges"])
    H = restrict(est)
    s, err = shd(est)
    states = story_states(D)
    return dict(groups=len(groups), matched=len(matched), precision=len(matched) / len(groups),
                recall=len(found) / len(TRUE_NODES), missing_nodes=sorted(set(TRUE_NODES) - found),
                extra=sorted(set(groups) - set(MAP)), shd=s, shd_parts=err, sid=sid(set(TRUE_ARROWS), set(H)),
                kl=kl(H, states), arrows_on_true_nodes=sorted(f"{u}{' ⊣ ' if r == 'blocks' else ' → '}{v}" for (u, v), (r, p) in H.items()))


if __name__ == "__main__":
    P = ROOT / "pipeline"
    for name, D, th in (("10 stories, all groups", P / "out_api_world10_hl", None), ("10 stories, >= 50%", P / "out_api_world10_hl", 0.5),
                        ("100 stories, >= 20%", P / "out_api_world100", 0.2), ("100 stories, >= 50%", P / "out_api_world100", 0.5)):
        r = score(D, th)
        print(f"== {name}")
        for k, v in r.items():
            print(f"   {k}: {round(v, 3) if isinstance(v, float) else v}")
    # references for reading KL: the true graph itself, and a graph with no arrows
    states = story_states(P / "out_api_world100")
    print("== reference KL (100 stories' frequencies): no arrows at all =", round(kl({}, states), 3))
