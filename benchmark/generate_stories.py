#!/usr/bin/env python3
"""Generate the benchmark stories: for each world, 100 people who reached the outcome and 100
stopped by a named blocker, each written as one short first-person story by Gemma.

Each person's true graph comes from the world (see three_worlds.md, section 4). Gemma receives
exactly that graph (events that happened, events that were prevented, causal links, who did what,
unrelated events) and only puts it into words.

Usage: python benchmark/generate_stories.py pilot   # 2 stories per world, printed
       python benchmark/generate_stories.py all     # 200 per world -> benchmark/stories/<world>.json"""
import importlib.util, json, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pydantic import BaseModel

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
WORLDS = {"homelessness": "world_homelessness", "career": "world_career", "mental_health": "world_mental"}
WHO = {"PROTAGONIST": "I", "EMPLOYER": "my employer", "GOVERNMENT": "the government", "FAMILY MEMBER": "a family member",
       "CLOSE FRIENDS": "my close friends", "MENTOR": "a mentor", "MANAGER": "my manager", "COMPANY": "the company",
       "THERAPIST": "a therapist", "PARTNER": "my partner"}
N_EACH = 100


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def person_graph(w, s, t):
    """The person's true graph: arrows (head, tail, kind), events that happened, events prevented."""
    free = w.run(s, t, off=w.BLOCKER_EVENTS)
    arrows = []
    blockers = {b for _, (_, bl) in w.MECH.items() for b in bl}
    useful = {b for b in blockers if any(b in bl and free[x] for x, (_, bl) in w.MECH.items())}
    for v, (en, bl) in w.MECH.items():
        if v in blockers and v not in useful:      # a blocker event appears only when it stood in the way
            continue
        for u in en:
            if s[u] and s[v]:
                arrows.append((u, v, "carries out" if u.isupper() else "enables"))
            elif free[u] and free[v] and not s[v] and t[v][0][u]:   # stopped: u would have brought on v
                arrows.append((u, v, "would have"))
        for u in bl:
            if s[u] and free[v]:
                arrows.append((u, v, "blocks"))
    if "infidelity" in s and s["infidelity"] and any("infidelity" in a[:2] for a in arrows):   # who, 70/30
        arrows.append(("PARTNER" if s["PARTNER"] else "PROTAGONIST", "infidelity", "carries out"))
    in_graph = {x for a in arrows for x in a[:2]}
    happened = [n for n in list(w.ROOT) + list(w.MECH) if n in in_graph and not n.isupper() and s[n]]
    prevented = [n for n in w.MECH if n in in_graph and not s[n]]
    distract = [d for d in w.DISTRACT if s[d]]
    return {"arrows": arrows, "happened": happened, "prevented": prevented,
            "participants": sorted({x for a in arrows for x in a[:2] if x.isupper()}), "distractors": distract}


def sample(w, seed):
    rng = random.Random(seed)
    out_g, stop_g = [], []
    while len(out_g) < N_EACH or len(stop_g) < N_EACH:
        s0, t = w.draw(rng)
        s = w.run(s0, t)
        if s[w.OUTCOME] and len(out_g) < N_EACH:
            out_g.append(("outcome", person_graph(w, s, t)))
        elif not s[w.OUTCOME] and w.run(s0, t, off=w.BLOCKER_EVENTS)[w.OUTCOME] and len(stop_g) < N_EACH:
            stop_g.append(("stopped", person_graph(w, s, t)))
    return out_g + stop_g


ROLE = {   # how a participant's part reads in plain language
    ("EMPLOYER", "job loss"): "my employer let me go",
    ("GOVERNMENT", "rental assistance"): "the government gave me rental assistance",
    ("FAMILY MEMBER", "providing a place to stay"): "a family member offered me a place to stay",
    ("CLOSE FRIENDS", "drug abuse"): "my close friends got me into drugs",
    ("PROTAGONIST", "drug abuse"): "I started using drugs myself",
    ("PROTAGONIST", "training"): "I signed up for training",
    ("PROTAGONIST", "networking"): "I went out and networked",
    ("PROTAGONIST", "working long hours"): "I chose to work long hours",
    ("CLOSE FRIENDS", "networking"): "my close friends helped me network by introducing me to people",
    ("MENTOR", "receiving guidance"): "a mentor guided me",
    ("MANAGER", "promotion"): "my manager promoted me",
    ("COMPANY", "layoffs"): "the company went through layoffs",
    ("THERAPIST", "therapy"): "I started seeing a therapist",
    ("CLOSE FRIENDS", "social support"): "my close friends stood by me",
    ("PROTAGONIST", "exercise"): "I started exercising",
    ("PROTAGONIST", "alcohol use"): "I started drinking",
    ("PROTAGONIST", "infidelity"): "I was the one who was unfaithful",
    ("PARTNER", "infidelity"): "my partner was unfaithful",
}


def prompt(g):
    facts = []
    for u, v, k in g["arrows"]:
        if k == "carries out":
            facts.append(f"- {ROLE.get((u, v), WHO.get(u, u) + ' brought about ' + v)}")
        elif k == "enables":
            facts.append(f"- {u} is what brought on {v}")
        elif k == "would have" and u.isupper():
            facts.append(f"- {WHO.get(u, u)} would have brought about {v}, but {v} did not happen")
        elif k == "would have":
            facts.append(f"- {u} would have brought on {v}, but {v} did not happen")
        elif v in g["prevented"]:
            facts.append(f"- {u} is what stopped {v} from happening")
        else:
            facts.append(f"- {u} was there, but {v} happened anyway")
    lines = ["Write a short story (4 to 8 sentences) in the first person, the way an ordinary person would tell",
             "someone what happened in their life. Sound like a real person talking, not a list: vary the wording,",
             "and do not keep repeating phrases such as \"led to\", \"caused\" or \"prevented\". Each connection below",
             "must still be clear from what happens in the story.",
             "Use ONLY these facts. Do not add any other event, cause, consequence, person or outcome, and no detail",
             "that suggests one (for example, do not add that you lost all your possessions). Things that were stopped",
             "must clearly NOT happen in the story.",
             "The only people in the story are: " + ", ".join(["me"] + [WHO[p] for p in g["participants"] if p != "PROTAGONIST"])
             + ". Do not bring in anyone else (no spouse, partner, boss, doctor or other person unless listed).", "",
             "What happened: " + ", ".join(g["happened"]) + "."]
    if g["prevented"]:
        lines.append("What was stopped and did NOT happen: " + ", ".join(g["prevented"]) + ".")
    lines += ["How it connects:", *facts]
    if g["distractors"]:
        lines.append("Also mention in passing, as unrelated details that affect nothing: " + ", ".join(g["distractors"]) + ".")
    return "\n".join(lines)


class Story(BaseModel):
    text: str


def main(mode):
    from pipeline.llm_client import OpenRouterClient
    cl = OpenRouterClient()
    outdir = HERE / "stories"
    outdir.mkdir(exist_ok=True)
    for i, (world, mod) in enumerate(WORLDS.items()):
        w = load(mod)
        people = sample(w, seed=100 + i)
        if mode == "pilot":
            people = [people[1], people[N_EACH + 1]]      # one outcome, one stopped (not the first pilot's people)
        if mode == "pilot12":                              # 12 new people, none used in earlier pilots
            people = [people[k] for k in (4, 5, N_EACH + 4, N_EACH + 5)]
        if mode == "pilot12b":                             # 12 more new people
            people = [people[k] for k in (6, 7, N_EACH + 6, N_EACH + 7)]
        if mode == "pilot10b":                             # 10 new people; homelessness includes both infidelity carriers
            def carrier(g, who):
                return any(x[0] == who and x[1] == "infidelity" for x in g["arrows"])
            if world == "homelessness":
                rest = people[8:]
                part = next(p for p in rest if carrier(p[1], "PARTNER"))
                prot = next(p for p in rest if carrier(p[1], "PROTAGONIST"))
                people = [people[8], people[N_EACH + 8], part, prot]
            else:
                people = [people[k] for k in ((8, N_EACH + 8, N_EACH + 9) if world == "career" else (8, 9, N_EACH + 8))]
        if mode == "pilot10":                              # 10 new people in total, none used in earlier pilots
            pick = {"homelessness": [2, 3, N_EACH + 2, N_EACH + 3], "career": [2, N_EACH + 2, N_EACH + 3],
                    "mental_health": [2, 3, N_EACH + 2]}[world]
            people = [people[k] for k in pick]
        items = [{"id": f"{world[:2]}{j:03d}", "world": world, "group": grp, "truth": g, "prompt": prompt(g)}
                 for j, (grp, g) in enumerate(people)]
        lock = threading.Lock()

        def write(it):
            it["text"] = cl.complete(task="benchmark_story", prompt=it["prompt"], schema=Story, temperature=0.7).text
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(write, items))
        if mode.startswith("pilot"):
            for it in items:
                print(f"===== {world} · {it['group']}\n--- facts given ---\n{it['prompt'].split(chr(10) + chr(10), 1)[1]}\n--- story ---\n{it['text']}\n")
        else:
            (outdir / f"{world}.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
            print(world, len(items), "stories ->", outdir / f"{world}.json", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pilot")
