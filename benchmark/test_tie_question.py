"""Grouping tie question, tested through OpenRouter before it goes into merge_kev.py.

A tie: a node answered "same" (or "opposite") to several groups with probabilities within 0.01 of its best one. The
new question asks which of the tied groups the node belongs to. Cases are rebuilt from finished runs; each group is
shown by its members (stated the same way as the group) that had joined before the story. Every case is asked twice,
with the group order swapped, and the two answers are averaged.
Usage: python benchmark/test_tie_question.py <openrouter model> [provider]"""
import collections, json, math, os, random, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import prompts

MODEL = sys.argv[1]
PROVIDER = sys.argv[2] if len(sys.argv) > 2 else None
TIE = 0.01


def ties(run):
    """[(story, node, [tied group labels])] from the run's recorded grouping decisions, in story order."""
    D = json.load(open(run + "/abstraction.json"))["decisions"]
    by = collections.defaultdict(list)
    for d in D:
        if d["choice"] in ("same", "opposite"):
            by[(d["narrative"], d["node"])].append((d["probs"][d["choice"]], d["group"]))
    out = []
    for (j, v), c in sorted(by.items()):
        best = max(p for p, _ in c)
        g = [lab for p, lab in c if best - p <= TIE]
        if len(g) > 1:
            out.append((j, v, g))
    return out


def members_before(run, story):
    """group label at grouping time -> the names of its 'same' members that joined before this story."""
    ab = json.load(open(run + "/abstraction.json"))
    lab = ab.get("labels", {})                        # final name -> label at grouping time
    return {lab.get(c, c): [m[1] for m in d["members"] if m[0] < story and m[2] > 0]
            for c, d in ab["concepts"].items()}


def show(names, k=6):
    top = [n for n, _ in collections.Counter(names).most_common(k)]
    return ", ".join(f'"{n}"' for n in top)


def question(node, groups):
    """groups: [(label, member names)] -> a choice question with one option per group."""
    opts = {f"group {i + 1}": f"the group of events like {show(ms)}" for i, (_, ms) in enumerate(groups)}
    instr = (f'Events from other people\'s stories have been grouped together. This story contains the event "{node}", '
             f"and it fits more than one of the groups below.\n\n"
             f'Judge only what the words "{node}" mean, not whether the event happened in this story.\n\n'
             "Group events only if they are the same kind of event at a high level and work the same way: they lead "
             "to the same kinds of outcomes. An event and its opposite describe one thing with opposite values (e.g. "
             "\"paying rent\" and \"falling behind on rent\").\n\n"
             f'Which group is "{node}" the same event as (or the same event stated the other way round)?')
    return {"type": "choice", "criteria": opts, "instructions": instr}


def ask(text, q):
    opts = list(q["criteria"].items()); L = [chr(65 + i) for i in range(len(opts))]
    prompt = (f"Text:\n{text}\n\nQuestion: {q['instructions']}\nOptions:\n" +
              "\n".join(f"{l}) {k}: {d}" for l, (k, d) in zip(L, opts)) +
              "\n\nAnswer with the single letter of the best option and nothing else.")
    body = {"model": MODEL, "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
            "messages": [{"role": "user", "content": prompt}]}
    if PROVIDER: body["provider"] = {"order": [PROVIDER], "allow_fallbacks": False}
    body["reasoning"] = {"enabled": False}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                                          "content-type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=120).read()); ch = r["choices"][0]
    lp = {}
    for t in ((ch.get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs", []):
        tok = t["token"].strip().rstrip(")").upper()
        if tok in L and tok not in lp: lp[tok] = t["logprob"]
    if not lp:
        lp = {(ch["message"].get("content") or "?").strip()[:1].upper(): 0.0}
    z = sum(math.exp(v) for v in lp.values())
    return [math.exp(lp[l]) / z if l in lp else 0.0 for l in L], r.get("provider")


def run_cases(run, world, pick, fixed_story=None, n=6):
    S = {s["id"]: s for s in json.load(open(f"benchmark/stories_v8/{world}.json"))}
    cases = [c for c in ties(run) if pick(c)][:n]
    for j, v, tied in cases:
        state = members_before(run, fixed_story or j)
        groups = [(g, state.get(g, [])) for g in tied]
        p1, prov1 = ask(S[j]["text"], question(v, groups))
        p2, prov2 = ask(S[j]["text"], question(v, groups[::-1]))
        avg = [(a + b) / 2 for a, b in zip(p1, p2[::-1])]
        best = max(range(len(groups)), key=lambda i: avg[i])
        print(f"  {j} {v!r}: picks {groups[best][0]!r} ({avg[best]:.2f})   [{prov1}, {prov2}]")
        for (g, ms), p in zip(groups, avg):
            print(f"      {p:.2f}  {g!r}: {show(ms, 4)}")


if __name__ == "__main__":
    print(f"### {MODEL} ({PROVIDER or 'any provider'})")
    print("== Gemma v10, mental health: depression nodes tied between the loneliness and the depression group "
          "(lists as at the first tie, me009)")
    run_cases("grid_v10/mental_health__gemma__llm", "mental_health",
              lambda c: "depress" in c[1] and len(c[2]) == 2, fixed_story="me009")
    print("== Llama-70B v8, homelessness: homelessness nodes tied between the two homelessness groups")
    run_cases("grid_v8/homelessness__llama70__llm", "homelessness",
              lambda c: ("homeless" in c[1] or "street" in c[1]) and any("ho060" in g for g in c[2]))
    print("== Gemma v10, homelessness: other ties")
    run_cases("grid_v10/homelessness__gemma__llm", "homelessness", lambda c: True, n=5)
