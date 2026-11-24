#!/usr/bin/env python3
"""Benchmark worlds and stories (spec: benchmark/world_spec.md).

Each world is a causal Bayesian network over binary nodes: roots happen with a rate; every other event follows a
noisy-OR with blockers, in causal order: each present cause u fires with chance p_u, each present blocker b fires
with chance q_b, and the event happens if some cause fired and no blocker fired. Participants (UPPERCASE) carry
out one event.

Who tells a story: a person to whom at least one starting event happened (a selection on parentless events only).

What a story tells (the story's truth: every told node with its value, and the arrows):
  - every starting event that happened, and the participant who carried it out;
  - for every event with at least one cause that happened (the event is "at stake"):
      happened      -> every cause that happened brought it on;
      did not happen, its protection was there  -> the protection kept it from happening (the causes put the
                                                    person at risk of it);
      did not happen, no protection there       -> it did not follow from its causes;
    and its protection, if it has one and it could happen (a root, or its own cause happened): there (with
    the participant behind it) or not there;
    and, for financial struggle (loneliness): if it did not happen and job loss (losing a family member) did not
    happen, that I still had my job (my family)
    (a cause that did not happen, no arrow);
  - nothing about events that were not at stake (an event none of whose causes happened, or a protection whose
    target was not at stake).
Usage: python benchmark/worlds_random.py           checks both worlds (rates, coverage of every node)
       python benchmark/worlds_random.py facts     writes stories_v7/<world>_facts.json (200 people each)"""
import random, sys

WORLDS = {
    "homelessness": {
        "outcome": "homelessness",
        "start": ["job loss", "divorce"],
        "tell_absent": {"financial struggle": ["job loss"]},   # divorce without a struggle, job kept: "I still had my job"
        "root": {"EMPLOYER": 0.55, "divorce": 0.45, "savings": 0.30, "FAMILY MEMBER": 0.80},
        "mech": {   # event: ({cause: p}, {blocker: q}), in causal order
            "job loss": ({"EMPLOYER": 1.0}, {}),
            "providing a place to stay": ({"FAMILY MEMBER": 0.5}, {}),
            "financial struggle": ({"job loss": 1.0, "divorce": 0.6}, {}),
            "can't pay rent": ({"financial struggle": 1.0}, {"savings": 0.80}),
            "eviction": ({"can't pay rent": 1.0}, {}),
            "homelessness": ({"eviction": 1.0}, {"providing a place to stay": 1.0}),
        },
    },
    "mental_health": {
        "outcome": "depression",
        "start": ["losing a family member", "breakup"],
        "tell_absent": {"loneliness": ["losing a family member"]},   # breakup without loneliness, family kept: "I still had my family"
        "root": {"losing a family member": 0.70, "breakup": 0.45, "CLOSE FRIENDS": 0.80, "THERAPIST": 0.40},
        "mech": {
            "social support": ({"CLOSE FRIENDS": 0.5}, {}),
            "therapy": ({"THERAPIST": 1.0}, {}),
            "loneliness": ({"losing a family member": 1.0, "breakup": 0.6}, {"social support": 0.6}),
            "depression": ({"loneliness": 0.7}, {"therapy": 0.7}),
        },
    },
}
ACTOR = {"job loss": "EMPLOYER", "providing a place to stay": "FAMILY MEMBER", "social support": "CLOSE FRIENDS",
         "therapy": "THERAPIST"}
SAY = {"job loss": "my employer let me go", "providing a place to stay": "a family member offered me a place to stay",
       "social support": "my close friends stood by me", "therapy": "I was seeing a therapist", "savings": "I had savings",
       "divorce": "I went through a divorce", "losing a family member": "a family member died", "breakup": "my partner and I broke up"}
SAY_NOT = {"job loss": "I still had my job", "losing a family member": "I still had my family", "providing a place to stay": "no family member offered me a place to stay", "social support": "no close friends stood by me",
           "therapy": "I was not seeing a therapist", "savings": "I had no savings"}


def person(w, rng):
    """True values of every node."""
    W = WORLDS[w]
    x = {r: int(rng.random() < p) for r, p in W["root"].items()}
    for v, (en, bl) in W["mech"].items():
        fired = [u for u, p in en.items() if x[u] and rng.random() < p]
        stop = [b for b, q in bl.items() if x[b] and rng.random() < q]
        x[v] = int(bool(fired) and not stop)
    return x


def possible(W, x, n):
    """A node could happen: a root, or an event with at least one cause that happened (not grey)."""
    return n in W["root"] or any(x[u] for u in W["mech"][n][0])


def tells(w, x):
    """What the story tells: {node: value}, the arrows (u, v, kind) and the facts for the writer."""
    W = WORLDS[w]
    told, arrows, facts = {}, [], []
    for s in W["start"]:
        if x[s]:
            told[s] = 1
            if s in ACTOR: told[ACTOR[s]] = 1; arrows.append((ACTOR[s], s, "carries out"))
            facts.append(SAY[s])
    for v, (en, bl) in W["mech"].items():
        causes = [u for u in en if x[u] and not u.isupper()]
        if not causes:
            continue                                           # not at stake: nothing told
        for u in W.get("tell_absent", {}).get(v, []):          # a cause that did not happen, told as absent
            if not x[u] and not x[v]:                          # only when the event did not happen either
                told[u] = 0; facts.append(SAY_NOT[u])
        prot = [b for b in bl if possible(W, x, b)]            # a protection that could not happen is not told
        for b in prot:                                         # the protection of an event at stake
            told[b] = x[b]
            if x[b] and b in ACTOR:
                told[ACTOR[b]] = 1; arrows.append((ACTOR[b], b, "carries out"))
            facts.append(SAY[b] if x[b] else SAY_NOT[b])
        told[v] = x[v]
        there = [b for b in prot if x[b]]
        if x[v]:
            arrows += [(u, v, "enables") for u in causes] + [(b, v, "failed") for b in there]
            facts += [f"{u}, and because of it, {v}" for u in causes]
            facts += [f"{b} was there, but {v} happened anyway" for b in there]
        elif there:
            arrows += [(u, v, "risk") for u in causes] + [(b, v, "stopped") for b in there]
            facts += [f"{' and '.join(causes)} put me at risk of {v}"] + [f"{b} kept {v} from happening" for b in there]
        else:
            facts.append(f"{' and '.join(causes)} happened, but {v} did not follow (did not happen)")
    return told, arrows, facts


def draw_stories(w, n=200, seed=0):
    """n people drawn at random among those to whom a starting event happened."""
    rng, out, W = random.Random(seed), [], WORLDS[w]
    while len(out) < n:
        x = person(w, rng)
        if not any(x[s] for s in W["start"]):
            continue
        told, arrows, facts = tells(w, x)
        out.append({"id": f"{w[:2]}{len(out):03d}", "world": w, "group": "random",
                    "truth": {"arrows": arrows, "nodes": told, "distractors": []}, "values": x, "facts": facts})
    return out


if __name__ == "__main__" and sys.argv[1:2] == ["facts"]:
    import json, pathlib
    d = pathlib.Path(__file__).parent / "stories_v7"; d.mkdir(exist_ok=True)
    for i, w in enumerate(WORLDS):
        items = draw_stories(w, 200, seed=900 + i)
        (d / f"{w}_facts.json").write_text(json.dumps(items, indent=1))
        print(w, len(items), "people ->", d / f"{w}_facts.json")
    sys.exit()

if __name__ == "__main__":
    for w, W in WORLDS.items():
        nodes = list(W["root"]) + list(W["mech"])
        big = draw_stories(w, 50_000, seed=1)
        out = sum(s["values"][W["outcome"]] for s in big) / len(big)
        print(f"\n{w}: outcome among storytellers {out:.1%}")
        low = {n: 0 for n in nodes}
        for k in range(300):
            ss = random.Random(k).sample(big, 200)
            for n in nodes:
                low[n] += sum(n in s["truth"]["nodes"] for s in ss) / 200 < 0.10
        for n in nodes:
            t = [s["truth"]["nodes"].get(n) for s in big]
            print(f"   {n:28} told {sum(v is not None for v in t) / len(big):5.1%} (as 1: {t.count(1) / len(big):5.1%}, as 0: {t.count(0) / len(big):5.1%}); below 10% in {low[n] / 3:.1f}% of 200-story samples")
