"""Shortcut questions (A -> C where the story has A -> B -> C) where our cluster Llama-3.3-70B answered enables:
asked again through OpenRouter, current closing sentence vs the stronger one, same provider for both.
Usage: python benchmark/test_shortcut_wording.py [model] [provider]"""
import json, math, os, sys, urllib.request, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import prompts
MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/llama-3.3-70b-instruct"
PROVIDER = sys.argv[2] if len(sys.argv) > 2 else None
R = "grid_v8/homelessness__llama70__llm"
S = {s["id"]: s for s in json.load(open("benchmark/stories_v8/homelessness.json"))}
e = json.load(open(R + "/evaluation_gemma-4-31b-it_flip_cut10.json"))["individual"]
OLD = 'If "{h}" affects\n"{t}" only through another listed node, answer none.'
NEW = 'If "{h}" leads to\n"{t}" by way of another listed node, for example "{h}" -> X -> "{t}", answer none.'

cases = []
for A, C in [("EMPLOYER", "financial struggle"), ("job loss", "can't pay rent")]:
    n = 0
    for sid in sorted(e):
        r = e[sid]
        if not isinstance(r, dict) or r.get("score") is None: continue
        G = json.load(open(f"{R}/{sid}__gemma-kev.chain.json")); m = r["map"]
        for x in G.get("asked", []):
            if m.get(x["head"]) == A and m.get(x["tail"]) == C and max(x["probs"], key=x["probs"].get) == "enables":
                cases.append((sid, x["head"], x["tail"], G["nodes"], x["probs"])); n += 1; break
        if n == 5: break

def ask(text, q):
    opts = list(q["criteria"].items()); L = [chr(65 + i) for i in range(len(opts))]
    prompt = (f"Text:\n{text}\n\nQuestion: {q['instructions']}\nOptions:\n" +
              "\n".join(f"{l}) {k}: {d}" for l, (k, d) in zip(L, opts)) +
              "\n\nAnswer with the single letter of the best option and nothing else.")
    body = {"model": MODEL, "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
            "messages": [{"role": "user", "content": prompt}]}
    if PROVIDER: body["provider"] = {"order": [PROVIDER], "allow_fallbacks": False}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "content-type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=120).read()); ch = r["choices"][0]
    lp = {}
    for t in ((ch.get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs", []):
        tok = t["token"].strip().rstrip(")").upper()
        if tok in L and tok not in lp: lp[tok] = t["logprob"]
    if not lp:
        lp = {(ch["message"].get("content") or "?").strip()[:1].upper(): 0.0}
    z = sum(math.exp(v) for v in lp.values())
    return {k: round(math.exp(lp[l]) / z, 3) if l in lp else 0.0 for l, (k, _) in zip(L, opts)}, r.get("provider")

tally = collections.Counter()
for sid, h, t, nodes, cl in cases:
    q = prompts.kev_arrow_question(h, t, nodes)
    old_s = OLD.format(h=h, t=t)
    assert q["instructions"].count(old_s) == 1, (sid, h, t)
    qn = dict(q); qn["instructions"] = q["instructions"].replace(old_s, NEW.format(h=h, t=t))
    a, pa = ask(S[sid]["text"], q); b, pb = ask(S[sid]["text"], qn)
    ca, cb = max(a, key=a.get), max(b, key=b.get); tally[("current", ca)] += 1; tally[("new", cb)] += 1
    print(f"{sid} {h!r} -> {t!r}\n   cluster v8: {max(cl, key=cl.get)} | current: {ca} {a} ({pa}) | new: {cb} {b} ({pb})")
print("\ntally:", dict(tally))
