#!/usr/bin/env python3
"""Benchmark worlds with plain random sampling (2026-11-23; replaces the outcome-selected samples of
world_homelessness.py / world_mental.py, which are kept for the earlier results).

Each world is a causal Bayesian network over binary nodes: roots with a rate, and every other event
from its causes by a noisy-OR with blockers (the same mechanism as before):
    P(v = 1 | parents) = [1 - prod over present enablers u of (1 - p_u)] * prod over present blockers b of (1 - q_b)
No distractors. Participants (UPPERCASE) are roots that carry out one event; a participant is present exactly
when it acts (arrow strength 1), so every participant in a story acted and is told.

Sampling, one person at a time, nothing selected or filtered:
  1. every root happens with its rate;
  2. every event, in causal order: each present enabler u fires with chance p_u, each present blocker b
     fires with chance q_b; the event happens if some enabler fired and no blocker fired.
The person's story then tells:
  - every event that happened, and every participant who acted;
  - every enabling link that fired (u happened, v happened, u's coin for v fired);
  - every block that stopped an event that would otherwise have happened (v is told as avoided), with
    what put the person at risk of it; and every blocker that was there but did not stop an event.
So the story's true table is the person's true values, and 200 people are 200 independent draws
from the world: counting P(v | parents) over the stories estimates the world's own probabilities.
Usage: python benchmark/worlds_random.py      (checks both worlds)"""
import random, sys

WORLDS = {
    "homelessness": {
        "outcome": "homelessness",
        "root": {"EMPLOYER": 0.78, "FAMILY MEMBER": 0.32, "divorce": 0.40, "house fire": 0.30, "savings": 0.20},
        "mech": {   # event: ({enabler: p}, {blocker: q}), in causal order
            "job loss": ({"EMPLOYER": 1.0}, {}),
            "providing a place to stay": ({"FAMILY MEMBER": 1.0}, {}),
            "can't pay rent": ({"job loss": 0.95, "divorce": 0.6}, {"savings": 0.80}),
            "eviction": ({"can't pay rent": 0.65}, {}),
            "homelessness": ({"eviction": 0.9, "house fire": 0.4}, {"providing a place to stay": 0.95}),
        },
    },
    "mental_health": {
        "outcome": "depression",
        "root": {"EMPLOYER": 0.45, "THERAPIST": 0.29, "CLOSE FRIENDS": 0.36, "losing a loved one": 0.80},
        "mech": {
            "job loss": ({"EMPLOYER": 1.0}, {}),
            "therapy": ({"THERAPIST": 1.0}, {}),
            "social support": ({"CLOSE FRIENDS": 1.0}, {}),
            "loneliness": ({"losing a loved one": 0.9}, {"social support": 0.6}),
            "chronic stress": ({"job loss": 0.72, "loneliness": 0.7}, {}),
            "depression": ({"chronic stress": 0.72}, {"therapy": 0.7}),
        },
    },
}


def person(w, rng):
    """One person: true values of every node, and what the story tells (events, fired links, blocks)."""
    W = WORLDS[w]
    x = {r: int(rng.random() < p) for r, p in W["root"].items()}
    links, blocks, risk, failed = [], [], [], []
    for v, (en, bl) in W["mech"].items():
        fired = [u for u, p in en.items() if x[u] and rng.random() < p]
        stop = [b for b, q in bl.items() if x[b] and rng.random() < q]
        x[v] = int(bool(fired) and not stop)
        if x[v]:
            links += [(u, v) for u in fired]
            failed += [(b, v) for b in bl if x[b]]  # a blocker that was there but did not stop v
        elif fired and stop:                       # would have happened: the block is told, v as avoided
            blocks += [(b, v) for b in stop]
            risk += [(u, v) for u in fired]        # what put the person at risk of v
    return {"values": x, "links": links, "blocks": blocks, "risk": risk, "failed": failed}


def sample(w, n=200, seed=0):
    rng = random.Random(seed)
    return [person(w, rng) for _ in range(n)]


def told(p):
    """Nodes a story names: events that happened, participants who acted, and avoided events with their blockers."""
    named = {k for k, v in p["values"].items() if v and not k.isupper()}
    named |= {u for u, _ in p["links"]} | {x for b in p["blocks"] for x in b}
    return named


ROLE = {("EMPLOYER", "job loss"): "my employer let me go",
        ("FAMILY MEMBER", "providing a place to stay"): "a family member offered me a place to stay",
        ("THERAPIST", "therapy"): "I started seeing a therapist",
        ("CLOSE FRIENDS", "social support"): "my close friends stood by me"}


def item(w, p, sid):
    """One story's record, in the format of stories_v3 (truth arrows with kinds, facts for the writer),
    plus truth["nodes"]: every node the story names (an event can happen without any arrow)."""
    arrows = [(u, v, "carries out" if u.isupper() else "enables") for u, v in p["links"]]
    arrows += [(u, v, "risk") for u, v in p["risk"]] + [(b, v, "stopped") for b, v in p["blocks"]]
    arrows += [(b, v, "failed") for b, v in p["failed"]]
    named = told(p)
    facts = []
    for u, v, k in arrows:
        if k == "carries out": facts.append(ROLE[(u, v)])
        elif k == "enables": facts.append(f"{u}, and because of it, {v}")
        elif k == "risk": facts.append(f"{u} put me at risk of {v}")
    for u, v, k in arrows:
        if k == "stopped": facts.append(f"{u} kept {v} from happening")
        elif k == "failed": facts.append(f"{u} was there, but {v} happened anyway")
    in_arrows = {n for a in arrows for n in a[:2]}
    facts += [f"{n} (it happened; nothing else in this list came of it)" for n in sorted(named - in_arrows)]
    return {"id": sid, "world": w, "group": "random", "truth": {"arrows": arrows, "nodes": sorted(named), "distractors": []},
            "values": p["values"], "facts": facts}


def draw_stories(w, n=200, seed=0):
    """n people drawn at random from those to whom at least one event happened (a person with no event is drawn again)."""
    rng, out = random.Random(seed), []
    while len(out) < n:
        p = person(w, rng)
        if told(p):
            out.append(item(w, p, f"{w[:2]}{len(out):03d}"))
    return out


if __name__ == "__main__" and sys.argv[1:2] == ["facts"]:      # the people and facts of the 200 stories per world
    import json, pathlib
    d = pathlib.Path(__file__).parent / "stories_v4"; d.mkdir(exist_ok=True)
    for i, w in enumerate(WORLDS):
        items = draw_stories(w, 200, seed=800 + i)
        (d / f"{w}_facts.json").write_text(json.dumps(items, indent=1))
        print(w, len(items), "people ->", d / f"{w}_facts.json")
    sys.exit()

if __name__ == "__main__":
    for w, W in WORLDS.items():
        big = sample(w, 400_000, seed=1)
        nodes = list(W["root"]) + list(W["mech"])
        out = sum(p["values"][W["outcome"]] for p in big) / len(big)
        empty = sum(not told(p) for p in big) / len(big)
        share = {n: sum(n in told(p) for p in big) / len(big) for n in nodes}
        low = {n: 0 for n in nodes}
        for s in range(1000):                       # 1,000 benchmarks of 200 random people
            ppl = sample(w, 200, seed=100 + s)
            for n in nodes:
                low[n] += sum(n in told(p) for p in ppl) / 200 < 0.10
        print(f"\n{w}: outcome {out:.1%}, stories with nothing to tell {empty:.1%}")
        for n in sorted(nodes, key=lambda n: share[n]):
            print(f"   {n:28} named in {share[n]:5.1%} of stories; below 10% in {low[n] / 10:4.1f}% of 200-story samples")
