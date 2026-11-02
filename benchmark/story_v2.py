#!/usr/bin/env python3
"""Story writing, version 2 (pilot): only facts a person could tell, natural tone from examples.

Truth of one person (lived facts only):
- A -> B when both happened (including a participant carrying out an event);
- A -> B as lived risk: A happened and put me at risk of B, which a blocker then stopped (one step only);
- blocker C -| B when C happened and B was about to happen (B would happen with every blocker off);
  if B did not happen, C stopped it; if B happened anyway, C was there but did not stop it.
No "would have caused" links. People are kept only if every event connects to the outcome, and a
stopped person only if a blocker stopped the outcome itself.
Usage: python benchmark/story_v2.py pilot5"""
import json, random, sys
from pathlib import Path
from pydantic import BaseModel
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
import generate_stories as gs          # world loading, sampling helpers, participant wording

EXAMPLES = [
    "When my hours were cut at work I couldn't keep up with the rent. I fell behind, got evicted, and ended up "
    "sleeping rough. Months on the street have left me ill most of the time.",
    "After I was laid off I quickly fell behind on rent and thought I'd lose the flat. But my sister stepped in "
    "and covered a couple of months, so the eviction never happened and I stayed housed.",
]


def lived_graph(w, s, t):
    free = w.run(s, t, off=w.BLOCKER_EVENTS)
    blockers = {b for _, (_, bl) in w.MECH.items() for b in bl}
    useful = {b for b in blockers if any(b in bl and free[x] for x, (_, bl) in w.MECH.items())}
    arrows = []
    for v, (en, bl) in w.MECH.items():
        if v in blockers and v not in useful:
            continue
        for u in en:
            if s[u] and s[v]:
                arrows.append((u, v, "carries out" if u.isupper() else "enables"))
        for u in bl:
            if s[u] and free[v]:
                arrows.append((u, v, "stopped" if not s[v] else "failed"))
    stopped = {a[1] for a in arrows if a[2] == "stopped"}
    for v in stopped:                      # one step of lived risk: what put me at risk of the stopped thing
        for u in w.MECH[v][0]:
            if s[u] and not s[v] and t[v][0][u]:
                arrows.append((u, v, "risk"))
    if "infidelity" in s and s["infidelity"] and any("infidelity" in a[:2] for a in arrows):
        arrows.append(("PARTNER" if s["PARTNER"] else "PROTAGONIST", "infidelity", "carries out"))
    return {"arrows": arrows, "distractors": [d for d in w.DISTRACT if s[d]]}


def connected(g, outcome):
    nodes = {n for a in g["arrows"] for n in a[:2]}
    if outcome not in nodes:
        return False
    adj = {n: set() for n in nodes}
    for a in g["arrows"]:
        adj[a[0]].add(a[1]); adj[a[1]].add(a[0])
    seen, st = {outcome}, [outcome]
    while st:
        for k in adj[st.pop()]:
            if k not in seen:
                seen.add(k); st.append(k)
    return seen == nodes


def sample_v2(w, seed, n_out, n_stop):
    rng = random.Random(seed)
    out, stop = [], []
    while len(out) < n_out or len(stop) < n_stop:
        s0, t = w.draw(rng)
        s = w.run(s0, t)
        g = lived_graph(w, s, t)
        if not connected(g, w.OUTCOME):
            continue
        if s[w.OUTCOME] and len(out) < n_out:
            out.append(("outcome", g))
        elif (not s[w.OUTCOME] and len(stop) < n_stop and w.run(s0, t, off=w.BLOCKER_EVENTS)[w.OUTCOME]
              and any(a[1] == w.OUTCOME and a[2] == "stopped" for a in g["arrows"])):
            stop.append(("stopped", g))
    return out + stop


POSITIVE = {"career"}      # worlds whose outcome is good: risk reads as "on track for"


ORDER = {"carries out": 0, "enables": 0, "risk": 1, "failed": 2, "stopped": 3}   # what stopped it comes last


def facts(g, world=""):
    lines = []
    g = dict(g, arrows=sorted(g["arrows"], key=lambda a: ORDER[a[2]]))
    for u, v, k in g["arrows"]:
        if k == "carries out":
            lines.append(gs.ROLE.get((u, v), f"{gs.WHO.get(u, u)} brought about {v}"))
        elif k == "enables":
            lines.append(f"{u}, and because of it, {v}")
        elif k == "risk":
            lines.append(f"{u} put me on track for {v}" if world in POSITIVE else f"{u} put me at risk of {v}")
        elif k == "stopped":
            lines.append(f"{u} kept {v} from happening")
        else:
            lines.append(f"{u} was there, but {v} happened anyway")
    return lines


def prompt(g, world=""):
    people = sorted({a[0] for a in g["arrows"] if a[0].isupper() and a[0] != "PROTAGONIST"})
    p = ["Write a short true-to-life story, in the first person, as a real person would tell it to a friend.",
         "Here is how two real people told theirs, for the tone:",
         *[f'"{e}"' for e in EXAMPLES], "",
         "The things that happened in your life (tell them in your own words, in a natural order):",
         *[f"- {f}" for f in facts(g, world)]]
    if g["distractors"]:
        p.append("- in passing, unrelated to the rest: " + ", ".join(g["distractors"]))
    p += ["",
          "You can add feelings and natural texture (how hard it was, how long it took), but no other events, "
          "causes or people" + (f" besides {', '.join(gs.WHO[x] for x in people)}" if people else "") + ".",
          "Only say what you lived through; never say what would have happened.",
          "Do not give a reason or purpose for anything unless the list gives it (no \"to cope\", \"to get through it\").",
          "\"Kept X from happening\" means X never happened at all, not \"for a while\".",
          "Do not add in-between steps such as starting a job, being unable to work, or a second loss.",
          "Several facts about the same event can be told in one sentence (e.g. \"the stress got me drinking\"); never say the "
          "same thing twice, and never copy the wording of the list."]
    return "\n".join(p)


def check(world, mod, pool=3000, draws=1000, seed=7):
    """Share of the 200 stories (100 outcome + 100 stopped) in which each node has a true arrow, under the
    lived-facts rules; the world passes if every node is >= 10% in >= 95% of draws and no distractor reaches 10%."""
    w = gs.load(mod)
    people = sample_v2(w, seed, pool, pool)
    out = [g for grp, g in people if grp == "outcome"]; stop = [g for grp, g in people if grp == "stopped"]
    nodes = sorted({n for g in out + stop for a in g["arrows"] for n in a[:2]})
    expected = [n for n in list(w.ROOT) + list(w.MECH) if n not in w.DISTRACT]
    missing = [n for n in expected if n not in nodes]
    rng = random.Random(seed + 1)
    fails = {n: 0 for n in expected}; whole = 0; dfail = 0
    share = {n: 0.0 for n in expected}
    for _ in range(draws):
        mix = rng.sample(out, 100) + rng.sample(stop, 100)
        ok = True
        for n in expected:
            f = sum(any(n in a[:2] for a in g["arrows"]) for g in mix) / 200
            share[n] += f / draws
            if f < 0.10:
                fails[n] += 1; ok = False
        whole += ok
        dfail += any(sum(d in g["distractors"] for g in mix) / 200 >= 0.10 for d in w.DISTRACT)
    print(f"== {world}: whole graph survives the 10% cut in {whole / draws:.1%} of draws; distractor reaches 10% in {dfail / draws:.1%}"
          + (f"; never appears: {missing}" if missing else ""))
    for n in sorted(expected, key=lambda n: share[n])[:5]:
        print(f"   {n:28s} {share[n]:6.1%}   below 10% in {fails[n] / draws:6.1%} of draws")


class Story(BaseModel):
    text: str


if __name__ == "__main__" and sys.argv[1:2] == ["check"]:
    for world, mod in (("homelessness", "world_homelessness"), ("career", "world_career"), ("mental_health", "world_mental")):
        check(world, mod)
    sys.exit()
if __name__ == "__main__" and sys.argv[1:2] == ["all"]:      # 100 outcome + 100 stopped per world
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from pipeline.llm_client import OpenRouterClient
    cl = OpenRouterClient()
    outdir = HERE / "stories_v2"; outdir.mkdir(exist_ok=True)
    for i, (world, mod) in enumerate((("homelessness", "world_homelessness"), ("career", "world_career"), ("mental_health", "world_mental"))):
        w = gs.load(mod)
        people = sample_v2(w, 700 + i, 100, 100)
        items = [{"id": f"{world[:2]}{j:03d}", "world": world, "group": grp, "truth": g,
                  "facts": facts(g, world), "prompt": prompt(g, world)} for j, (grp, g) in enumerate(people)]
        def write(it):
            it["text"] = cl.complete(task="benchmark_story_v2", prompt=it["prompt"], schema=Story, temperature=0.8).text
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(write, items))
        (outdir / f"{world}.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
        print(world, len(items), "stories ->", outdir / f"{world}.json", flush=True)
    sys.exit()

if __name__ == "__main__":
    from pipeline.llm_client import OpenRouterClient
    cl = OpenRouterClient()
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot5"
    plan = {"pilot5": [("homelessness", "world_homelessness", 1, 1), ("career", "world_career", 1, 1), ("mental_health", "world_mental", 0, 1)],
            "pilot5b": [("homelessness", "world_homelessness", 1, 1), ("career", "world_career", 1, 1), ("mental_health", "world_mental", 1, 0)]}[mode]
    base = {"pilot5": 500, "pilot5b": 600}[mode]
    for i, (world, mod, no, ns) in enumerate(plan):
        w = gs.load(mod)
        for grp, g in sample_v2(w, base + i, no, ns):
            text = cl.complete(task="benchmark_story_v2", prompt=prompt(g, world), schema=Story, temperature=0.8).text
            print(f"===== {world} · {grp}\nfacts: " + " | ".join(facts(g, world))
                  + (f" | side: {', '.join(g['distractors'])}" if g["distractors"] else "") + f"\n>>> {text}\n")
