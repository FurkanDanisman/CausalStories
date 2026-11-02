#!/usr/bin/env python3
"""Homelessness world for the benchmark (draft graph: benchmark/draft_graphs/1_homelessness.png).

Independent causes with blockers (noisy-OR with inhibitors), closed world (no leak):
P(event | parents) = [1 - prod over present enablers (1 - p)] * prod over present blockers (1 - q).
Participants are roots that carry out their event (an enabler arrow into it).

Stories: 100 people who became homeless + 100 saved by a named blocker (homeless if every blocker
event is switched off, same random draws). A node appears in a story when it has a true arrow there:
- enabling or carrying-out arrow A -> B: A and B both happened;
- blocking arrow C -| B: C happened and B was threatened (B would happen with all blockers off);
- enabling arrow A -> B in someone stopped: B did not happen but would have, and A (happened or
  would have happened) would have brought it on, so the chain down to the outcome is in the story;
- a blocker event (and the participant or cause behind it) appears only when it stood in the way of
  something that was about to happen (whether it stopped it or not).
Requirement: every node of the true graph appears in >= 10% of the 200 stories in >= 95% of
draws; distractors are mentioned in <= 5% of stories (they have no arrows).
Usage: python benchmark/world_homelessness.py"""
import random

OUTCOME = "homelessness"
ROOT = {   # rate per year (participants: involved / acting); smaller world 2026-10-29
    "EMPLOYER": 0.047, "FAMILY MEMBER": 0.80,
    "divorce": 0.109, "house fire": 0.047, "savings": 0.63,
    "getting a pet": 0.04, "taking up running": 0.04}
DISTRACT = ("getting a pet", "taking up running")
MECH = {   # event: ({enabler: p}, {blocker: q}) in causal order
    "job loss": ({"EMPLOYER": 0.98}, {}),
    "providing a place to stay": ({"FAMILY MEMBER": 0.9}, {}),
    "can't pay rent": ({"job loss": 0.95, "divorce": 0.6}, {"savings": 0.80}),
    "eviction": ({"can't pay rent": 0.65}, {}),
    "homelessness": ({"eviction": 0.9, "house fire": 0.4}, {"providing a place to stay": 0.95, "savings": 0.95}),
}
BLOCKER_EVENTS = ('savings', 'providing a place to stay')


def draw(rng):
    s = {r: int(rng.random() < p) for r, p in ROOT.items()}
    t = {v: ({u: rng.random() < p for u, p in en.items()}, {u: rng.random() < q for u, q in bl.items()})
         for v, (en, bl) in MECH.items()}
    return s, t


def run(s, t, off=()):
    s = dict(s)
    for b in off:
        s[b] = 0
    for v, (en, bl) in MECH.items():
        if v in off:
            s[v] = 0
            continue
        s[v] = int(any(s[u] and t[v][0][u] for u in en) and not any(s[u] and t[v][1][u] for u in bl))
    return s


def arrows_in_story(s, t):
    """True arrows of one person's graph, by the rule in the module docstring."""
    free = run(s, t, off=BLOCKER_EVENTS)          # what would happen with every blocker off
    out = []
    blockers = {b for _, (_, bl) in MECH.items() for b in bl}
    useful = {b for b in blockers if any(b in bl and free[w] for w, (_, bl) in MECH.items())}
    for v, (en, bl) in MECH.items():
        if v in blockers and v not in useful:      # a blocker event appears only when it stood in the way
            continue
        for u in en:
            if s[u] and s[v]:
                out.append((u, v))
            elif free[u] and free[v] and not s[v] and t[v][0][u]:   # stopped: u would have brought on v
                out.append((u, v))
        for u in bl:
            if s[u] and free[v]:
                out.append((u, v))
    return out


def population(n, seed=0):
    rng = random.Random(seed)
    homeless, saved = [], []
    for _ in range(n):
        s0, t = draw(rng)
        s = run(s0, t)
        rec = (s, set(x for a in arrows_in_story(s, t) for x in a))
        if s["homelessness"]:
            homeless.append(rec)
        elif run(s0, t, off=BLOCKER_EVENTS)["homelessness"]:
            saved.append(rec)
    return homeless, saved


def calib(n=1_000_000, seed=5):
    rng = random.Random(seed); ev = evh = 0
    for _ in range(n):
        s = run(*draw(rng)); ev += s["eviction"]; evh += s["eviction"] and s["homelessness"]
    return ev / n, evh / max(ev, 1)


if __name__ == "__main__":
    homeless, saved = population(3_000_000)
    print(f"population 3,000,000: homeless {len(homeless)} ({len(homeless) / 3e6:.3%}), saved by a blocker {len(saved)}")
    ev, evh = calib()
    print(f"P(eviction) {ev:.3%}   P(homeless | eviction) {evh:.3f}   (real: ~2.3-3%, ~0.22)")
    nodes = [n for n in list(ROOT) + list(MECH) if n not in DISTRACT]
    pool = [(h, 1) for h in homeless] + [(x, 0) for x in saved]
    share = {n: (sum(n in u for (s, u) in homeless) / len(homeless) + sum(n in u for (s, u) in saved) / len(saved)) / 2 for n in nodes}
    dshare = {d: (sum(s[d] for s, u in homeless) / len(homeless) + sum(s[d] for s, u in saved) / len(saved)) / 2 for d in DISTRACT}
    rng = random.Random(1)
    fails = {n: 0 for n in nodes}; whole = 0; dfail = 0
    for _ in range(1000):
        mix = rng.sample(homeless, 100) + rng.sample(saved, 100)
        ok = True
        for n in nodes:
            if sum(n in u for s, u in mix) / 200 < 0.10:
                fails[n] += 1; ok = False
        whole += ok
        dfail += any(sum(s[d] for s, u in mix) / 200 >= 0.10 for d in DISTRACT)
    print("share of stories in which each node has a true arrow (100 homeless + 100 saved):")
    for n in sorted(nodes, key=lambda n: share[n]):
        print(f"   {n:28s} {share[n]:6.1%}   below 10% in {fails[n] / 10:5.1f}% of draws")
    for d in DISTRACT:
        print(f"   {d:28s} mentioned in {dshare[d]:5.1%} (distractor)")
    print(f"whole graph survives the 10% cut in {whole / 10:.1f}% of 1,000 draws; a distractor reaches 10% in {dfail / 10:.1f}%")
