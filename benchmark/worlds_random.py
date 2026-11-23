#!/usr/bin/env python3
"""Benchmark worlds with plain random sampling (spec: benchmark/world_spec.md).

Each world is the original causal graph (draft_graphs/), with two removals: no house fire, and savings blocks
only can't pay rent. Every node is binary. Roots (participants in UPPERCASE, and root events) happen with a
rate; every other event follows a noisy-OR with blockers, in causal order:
    each present cause u fires with chance p_u, each present blocker b fires with chance q_b;
    the event happens if some cause fired and no blocker fired.
No hidden variables, no distractors.

A person is drawn at random; a person to whom nothing happened (every node 0) is drawn again.
The story tells, from the person's true values:
  1. every event that happened, and every participant who acted;
  2. every cause that brought on an event that happened;
  3. every block: a blocker that stopped a threatened event (with what threatened it), or a blocker that was
     there but did not stop it;
  4. what did not happen, where it matters:
     (a) a blocker of a happened or threatened event that was absent ("I had no savings");
     (b) an event that a happened event can bring on, but which did not follow ("I was never evicted").
  If a cause happened, the event did not follow, and a blocker of it was present, the story credits the
  blocker ("my sister took me in, so I never ended up on the street"): the truth lists it as a block.
The story's truth is every node it tells, with its value (1 or 0), and the arrows that fired.
Usage: python benchmark/worlds_random.py            checks both worlds
       python benchmark/worlds_random.py facts      writes stories_v5/<world>_facts.json (200 people each)"""
import random, sys

WORLDS = {
    "homelessness": {
        "outcome": "homelessness",
        "root": {"EMPLOYER": 0.78, "FAMILY MEMBER": 0.45, "divorce": 0.60, "savings": 0.30},
        "mech": {   # event: ({cause: p}, {blocker: q}), in causal order
            "job loss": ({"EMPLOYER": 0.98}, {}),
            "providing a place to stay": ({"FAMILY MEMBER": 0.9}, {}),
            "can't pay rent": ({"job loss": 0.95, "divorce": 0.6}, {"savings": 0.80}),
            "eviction": ({"can't pay rent": 0.65}, {}),
            "homelessness": ({"eviction": 0.9}, {"providing a place to stay": 0.95}),
        },
    },
    "mental_health": {
        "outcome": "depression",
        "root": {"EMPLOYER": 0.50, "THERAPIST": 0.40, "CLOSE FRIENDS": 0.45, "losing a loved one": 0.80},
        "mech": {
            "job loss": ({"EMPLOYER": 0.98}, {}),
            "therapy": ({"THERAPIST": 1.0}, {}),
            "social support": ({"CLOSE FRIENDS": 0.9}, {}),
            "loneliness": ({"losing a loved one": 0.9}, {"social support": 0.6}),
            "chronic stress": ({"job loss": 0.72, "loneliness": 0.7}, {}),
            "depression": ({"chronic stress": 0.72}, {"therapy": 0.7}),
        },
    },
}
ROLE = {("EMPLOYER", "job loss"): "my employer let me go",
        ("FAMILY MEMBER", "providing a place to stay"): "a family member offered me a place to stay",
        ("THERAPIST", "therapy"): "I started seeing a therapist",
        ("CLOSE FRIENDS", "social support"): "my close friends stood by me"}
ABSENT = {"savings": "I had no savings", "providing a place to stay": "no family member offered me a place to stay",
          "social support": "no close friends stood by me", "therapy": "I was not seeing a therapist"}


def person(w, rng):
    """True values of every node, and the arrows that fired, were blocked or failed."""
    W = WORLDS[w]
    x = {r: int(rng.random() < p) for r, p in W["root"].items()}
    links, blocks, risk, failed, threatened = [], [], [], [], set()
    for v, (en, bl) in W["mech"].items():
        fired = [u for u, p in en.items() if x[u] and rng.random() < p]
        stop = [b for b, q in bl.items() if x[b] and rng.random() < q]
        x[v] = int(bool(fired) and not stop)
        if fired:
            threatened.add(v)
        if x[v]:
            links += [(u, v) for u in fired]
            failed += [(b, v) for b in bl if x[b]]
        elif fired and stop:
            blocks += [(b, v) for b in stop]
            risk += [(u, v) for u in fired]
    return {"values": x, "links": links, "blocks": blocks, "risk": risk, "failed": failed, "threatened": threatened}


def story_truth(w, p):
    """The nodes a story tells, with their values, and its arrows (rules 1-4 of the module docstring)."""
    W, x = WORLDS[w], p["values"]
    arrows = [(u, v, "carries out" if u.isupper() else "enables") for u, v in p["links"]]
    arrows += [(u, v, "risk") for u, v in p["risk"]] + [(b, v, "stopped") for b, v in p["blocks"]]
    arrows += [(b, v, "failed") for b, v in p["failed"]]
    told = {n: 1 for n, v in x.items() if v and not n.isupper()}
    told.update({n: x[n] for a in arrows for n in a[:2]})
    absent, missed = [], []
    for v, (en, bl) in W["mech"].items():                  # a helper present when a cause happened but v did not
        if not x[v] and v not in p["threatened"]:            # follow: the story credits the helper (as it reads)
            causes = [u for u in en if x[u] and not u.isupper()]
            helpers = [b for b in bl if x[b]]
            if causes and helpers:
                arrows += [(u, v, "risk") for u in causes] + [(b, v, "stopped") for b in helpers]
                told.update({n: x[n] for n in causes + helpers + [v]})
    for v, (en, bl) in W["mech"].items():
        if x[v] or v in p["threatened"]:
            absent += [b for b in bl if not x[b] and b not in told]              # rule 4a
        for u in en:
            if x[u] and not u.isupper() and not x[v] and v not in p["threatened"] and not any(x[b] for b in bl):
                missed.append((u, v))                                            # rule 4b
    for b in absent: told[b] = 0
    for u, v in missed: told[v] = 0
    return told, arrows, absent, missed


def facts_of(w, p):
    told, arrows, absent, missed = story_truth(w, p)
    facts = []
    for u, v, k in arrows:
        if k == "carries out": facts.append(ROLE[(u, v)])
        elif k == "enables": facts.append(f"{u}, and because of it, {v}")
        elif k == "risk": facts.append(f"{u} put me at risk of {v}")
    for u, v, k in arrows:
        if k == "stopped": facts.append(f"{u} kept {v} from happening")
        elif k == "failed": facts.append(f"{u} was there, but {v} happened anyway")
    facts += [f"{ABSENT[b]} (did not happen)" for b in absent]
    facts += [f"{u} happened, but {v} did not follow (did not happen)" for u, v in missed]
    in_arrows = {n for a in arrows for n in a[:2]} | set(absent) | {v for _, v in missed} | {u for u, _ in missed}
    facts += [f"{n} (it happened; tell it plainly)" for n, val in told.items() if val and n not in in_arrows]
    return told, arrows, facts


def draw_stories(w, n=200, seed=0):
    """n people drawn at random; a person to whom nothing happened is drawn again."""
    rng, out = random.Random(seed), []
    while len(out) < n:
        p = person(w, rng)
        if not any(p["values"].values()):
            continue
        told, arrows, facts = facts_of(w, p)
        out.append({"id": f"{w[:2]}{len(out):03d}", "world": w, "group": "random",
                    "truth": {"arrows": arrows, "nodes": told, "distractors": []}, "values": p["values"], "facts": facts})
    return out


if __name__ == "__main__" and sys.argv[1:2] == ["facts"]:
    import json, pathlib
    d = pathlib.Path(__file__).parent / "stories_v5"; d.mkdir(exist_ok=True)
    for i, w in enumerate(WORLDS):
        items = draw_stories(w, 200, seed=800 + i)
        (d / f"{w}_facts.json").write_text(json.dumps(items, indent=1))
        print(w, len(items), "people ->", d / f"{w}_facts.json")
    sys.exit()

if __name__ == "__main__":
    for w, W in WORLDS.items():
        rng = random.Random(1); n = 0; skipped = 0; out = 0; share = {}
        while n < 200_000:
            p = person(w, rng)
            if not any(p["values"].values()): skipped += 1; continue
            n += 1; out += p["values"][W["outcome"]]
            told, _, _ = facts_of(w, p)
            for k, v in told.items(): share[(k, v)] = share.get((k, v), 0) + 1
        print(f"\n{w}: outcome {out / n:.1%}; people with nothing happened {skipped / (n + skipped):.1%}")
        for k in list(W["root"]) + list(W["mech"]):
            print(f"   {k:28} told as happened {share.get((k, 1), 0) / n:5.1%}   told as not happened {share.get((k, 0), 0) / n:5.1%}")
