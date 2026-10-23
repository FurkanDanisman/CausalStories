#!/usr/bin/env python3
"""Synthetic world stories, as on Sep 4 (session 7c87bea0): sample people from the true
world (world_model.json, seed 0) with the same coin-flip walk, then write one short
first-person story per person that mentions exactly the events that happened to them.
p0-p9 reuse the original 10 stories; the rest are written by Gemma (OpenRouter).
    python world_generate.py 100 world100.json"""
import json, random, sys
from pathlib import Path
from pydantic import BaseModel

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "world100.json")
CPD = json.loads(Path("world_model.json").read_text())["cpd"]


def sample_person(rng):                      # identical to the Sep 4 script
    crisis = 1
    support = int(rng.random() < CPD["support"])
    rent = int(crisis and rng.random() < CPD["rent_trouble | crisis"])
    pe = CPD["eviction | rent_trouble,support"] if support else CPD["eviction | rent_trouble,no_support"]
    evict = int(rent and rng.random() < pe)
    homeless = int(evict and rng.random() < CPD["homeless | eviction"])
    health = int(homeless and rng.random() < CPD["health_decline | homeless"])
    return dict(crisis=crisis, support=support, rent_trouble=rent, eviction=evict,
                homeless=homeless, health_decline=health)


class Story(BaseModel):
    text: str


def main():
    rng = random.Random(0)
    people = [sample_person(rng) for _ in range(N)]
    orig = json.loads(Path("world_homelessness.variants.json").read_text())["variants"]
    for i, v in enumerate(orig):              # the first 10 draws must be the original 10
        assert people[i] == v["true_events"], f"p{i} differs from the original sample"
    examples = "\n".join(f'events {json.dumps(v["true_events"])}\nstory: "{v["text"]}"' for v in orig)
    sys.path.insert(0, ".")
    from pipeline.llm_client import OpenRouterClient
    client = OpenRouterClient()

    def write(i):
        if i < len(orig):
            return orig[i]["text"]
        prompt = (
            "Write a short first-person story (2-3 sentences) for one person, to match their "
            "events exactly: the crisis always happens; every other event is mentioned only if "
            "it happened to this person (1). If support = 1, someone helped them. Events that are "
            "0 did not happen: do not describe them as happening. Use a different kind of crisis "
            "from the examples where you can.\n\nExamples:\n" + examples +
            f"\n\nNow this person:\nevents {json.dumps(people[i])}\n"
            'Return JSON {"text": "<the story>"}.')
        return client.complete(task="world_story", prompt=prompt, schema=Story, temperature=0.7).text

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        texts = list(pool.map(write, range(N)))
    items = [{"id": f"p{i}", "text": t, "true_events": p} for i, (t, p) in enumerate(zip(texts, people))]
    OUT.write_text(json.dumps(items, indent=1))
    cols = list(people[0])
    print(f"wrote {OUT}: {N} stories; true frequencies: " +
          ", ".join(f"{c} {sum(p[c] for p in people)}/{N}" for c in cols))


if __name__ == "__main__":
    main()
