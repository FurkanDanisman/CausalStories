#!/usr/bin/env python3
"""Career-success world for the benchmark (draft graph: benchmark/draft_graphs/2_career_success.png).

Same construction as world_homelessness.py: independent causes with blockers (noisy-OR with
inhibitors), closed world. Participants are roots that carry out their event; COMPANY is present
for everyone (everyone has an employer). Smaller world (2026-10-29): 8 events, 2 participants.

Stories: 100 people promoted + 100 who would have been promoted with every blocker off but were not.
A node appears in a story when it has a true arrow there (see world_homelessness.py).
Requirement: every node in >= 10% of the 200 stories in >= 95% of draws; distractors <= 5%.
Rates: promotion ~9%/yr (ADP 2019); degree 38.6% (Census 2024); mentor ~26-38% (Accountemps,
Environics); discrimination ~19% (ADP 2025); layoffs ~1.1%/month (JOLTS).
Usage: python benchmark/world_career.py"""
import random

OUTCOME = "promotion"
ROOT = {   # smaller world 2026-10-29
    "MENTOR": 0.30, "COMPANY": 1.0,
    "degree": 0.386, "facing discrimination": 0.19,
    "moving apartment": 0.04, "buying a car": 0.04}
DISTRACT = ("moving apartment", "buying a car")
MECH = {
    "receiving guidance": ({"MENTOR": 0.95}, {}),
    "gaining skills": ({"degree": 0.5, "receiving guidance": 0.4}, {}),
    "getting noticed": ({"receiving guidance": 0.4}, {}),
    "layoffs": ({"COMPANY": 0.12}, {}),
    "strong performance": ({"gaining skills": 0.6}, {}),
    "promotion": ({"strong performance": 0.56, "getting noticed": 0.2},   # 0.56: promotion ~9%/yr in the smaller world
                  {"facing discrimination": 0.7, "layoffs": 0.8}),
}
BLOCKER_EVENTS = ('facing discrimination', 'layoffs')


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
        if s[OUTCOME]:
            homeless.append(rec)
        elif run(s0, t, off=BLOCKER_EVENTS)[OUTCOME]:
            saved.append(rec)
    return homeless, saved


def calib(n=1_000_000, seed=5):
    rng = random.Random(seed); pr = bo = 0
    for _ in range(n):
        s = run(*draw(rng)); pr += s[OUTCOME]; bo += 0
    return pr / n, bo / n


if __name__ == "__main__":
    homeless, saved = population(1_000_000)
    print(f"population 1,000,000: promoted {len(homeless)} ({len(homeless) / 1e6:.3%}), stopped by a blocker {len(saved)}")
    pr, bo = calib()
    print(f"P(promotion) {pr:.3%}   (real: ~9%)")
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
    print("share of stories in which each node has a true arrow (100 promoted + 100 stopped):")
    for n in sorted(nodes, key=lambda n: share[n]):
        print(f"   {n:28s} {share[n]:6.1%}   below 10% in {fails[n] / 10:5.1f}% of draws")
    for d in DISTRACT:
        print(f"   {d:28s} mentioned in {dshare[d]:5.1%} (distractor)")
    print(f"whole graph survives the 10% cut in {whole / 10:.1f}% of 1,000 draws; a distractor reaches 10% in {dfail / 10:.1f}%")
