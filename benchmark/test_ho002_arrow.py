"""One arrow question, asked to the generator LLM exactly as the chain walk asks it (KevClient, model llm:llm, local
vLLM server at LLM_URL): story ho002, eviction -> being homeless, with the same node list as the chain walk.
5 times with the current causal-edge definition (guidance_events.md), 5 times with Regan et al.'s two
"Without effect" rows added. Prints each answer's probabilities and the first answer token with its top alternatives.
Run by test_ho002_arrow.sbatch (the server is started there)."""
import json, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import prompts
from pipeline.kev_client import KevClient

TEXT = ("My company laid me off, and I got divorced. Money got really tight, and I couldn't pay my rent anymore. "
        "I had no savings to fall back on. I got evicted, but my brother took me in, so I never ended up on the street.")
IDS = ["laying off", "divorce", "financial hardship", "not paying rent", "eviction", "providing shelter",
       "being homeless", "company", "brother"]
ROW_ENABLES = ('''  * A removes a barrier so B can continue (free, maintain): "Oksana removed
    obstacles to the ball rolling."
''', '''  * A would make B happen, but B does not happen despite it, because something
    else stopped it (without effect; despite, even though): "Despite our efforts,
    we couldn't get the ball rolling."
''')
ROW_BLOCKS = ('''  * A is a barrier to B, preventing it (refrain, forbid, hold): "Oksana prevented
    the ball from rolling."
''', '''  * A would stop B, but B happens despite it (without effect; despite, even
    though): "We tried but could not stop the ball."
''')

raw = []                                         # every response of the server, to read the first answer token
_open = urllib.request.urlopen


class Keep:
    def __init__(self, r): self.r = r
    def __enter__(self): return self
    def __exit__(self, *a): self.r.close()
    def read(self):
        b = self.r.read(); raw.append(json.loads(b)); return b


urllib.request.urlopen = lambda *a, **k: Keep(_open(*a, **k))

current = prompts.kev_arrow_question("eviction", "being homeless", IDS)
regan = dict(current)
for old, add in (ROW_ENABLES, ROW_BLOCKS):
    assert regan["instructions"].count(old) == 1, "definition text not found"
    regan["instructions"] = regan["instructions"].replace(old, old + add)

import os
MODEL = os.environ.get("TEST_MODEL", "llm:llm")     # llm:llm -> LLM_URL (local vLLM), as in the benchmark;
kev = KevClient(base_url="http://127.0.0.1:1", model=MODEL)   # llm:google/gemma-4-31b-it -> OpenRouter (LLM_URL unset)
for name, q in [("current definition", current), ("with Regan's two rows", regan)]:
    print(f"===== {name}", flush=True)
    for i in range(5):
        raw.clear()
        a = kev.ask(TEXT, {"q": q})["q"]
        ch = raw[-1]["choices"][0]
        first = ((ch.get("logprobs") or {}).get("content") or [{}])[0]
        top = [(t["token"], round(t["logprob"], 2)) for t in first.get("top_logprobs", [])[:5]]
        print(f"  {i + 1}: " + " ".join(f"{k}={v:.3f}" for k, v in a["probabilities"].items())
              + f" | first token {first.get('token')!r} | top {top}"
              + (f" | provider {raw[-1].get('provider')}" if raw[-1].get("provider") else ""), flush=True)
