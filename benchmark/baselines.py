#!/usr/bin/env python3
"""Competing approaches (benchmark/workplan.md §2). Each writes one graph per story in the pipeline's
format (<out>/<id>__<tag>.chain.json: nodes, kinds, edges, asked), so OUR aggregation (run.py merge-kev,
name-groups, node-happened, implied, connect, then Steps 5-6) and the scorer apply to it unchanged. C6 writes the world graph directly.

All LLM calls go to one OpenAI-compatible endpoint (a vLLM server on Killarney):
  LLM_URL   default http://127.0.0.1:8000/v1/chat/completions
  LLM_MODEL the served model name (e.g. the Gemma 4 31B weights path)

  C1 naive    one prompt per story -> nodes + arrows                 (ReCITE prompt, Saklad et al., ACL 2026)
  C2 whole    story + our extracted nodes -> all arrows in one answer (ReCITE node-names prompt; full-graph prompting)
  C3 pairwise our nodes, every ordered pair asked separately         (Kiciman et al., 2023)
  C4 bfs      our nodes, roots first, then "what does X cause?"     (Jiralerspong et al., 2024)
  C5 cmap     every causal claim in the story (cause, effect, sign)  (Causal Map; Powell & Cabral, 2025)
  C6 corpus   all 200 stories in one prompt -> world graph; story x node table by the same LLM

Usage: python benchmark/baselines.py <C1..C6> <texts.json> <out_dir> [--nodes chain_nodes.json] [--workers 16]"""
import argparse, json, math, os, re, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = os.environ.get("LLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = os.environ.get("LLM_MODEL", "gemma")
RELS = ("enables", "blocks")
K = int(os.environ.get("MAX_EVENTS", "8"))   # the same event budget as our node extraction (Prompt 1a, K = 8)
# Our Prompt 1a sentence, added to every competing approach whose LLM picks its own nodes (C1, C5, C6);
# participants are not counted, as in our method (they come from a separate call there).
TOP_K = ("Use at most {k} events: the {k} most important for the causal chain. People and organisations "
         "who act are not counted in this limit. " + 'Each event must be a different thing that happened. Never list two events for one happening, even when the story describes it twice or in two steps: "the car broke down" and "the car stopped working" are one event; "a flood warning" and "the flood" are one event. List it once, with the most general name. Never list an event together with its opposite: when the story says something was avoided, name the event that was avoided ("the flood"), not its opposite ("staying dry").')

# ReCITE's generation prompts (src/recite/prompts/causal_graph_generation*.md), verbatim except:
# a "relation" field (enables / blocks) is added because the benchmark scores the sign, C1 does not get
# "Your graph will contain exactly NUM_NODES nodes" (that would give away the true node count), and C1 gets
# our event budget (TOP_K, at most K = 8 events, the same for every method).
RECITE = """You are an expert causal reasoner and economist. Your task is to generate a causal graph for the provided markdown text. First, use extremely long chain-of-thought reasoning in <think> tags. Then, provide your final answer in a JSON code block, strictly following the following format:
```json
{
    "relationships": [
        {"source": causal_variable0, "sink": affected_variable0, "relation": "enables" or "blocks"},
        {"source": causal_variable1, "sink": affected_variable1, "relation": "enables" or "blocks"},
        ...
    ]
}
```

"relation" is "enables" if the source brings about or increases the sink, and "blocks" if the source prevents or reduces it. When answering, do not provide any additional reasoning, commentary, or other information - only provide the JSON code block, with each dictionary representing one relationship in the graph."""

RECITE_NODES = """You are an expert causal reasoner and economist. Your task is to generate a causal graph for the provided markdown text. First, use extremely long chain-of-thought reasoning in <think> tags. Then, provide your final answer in a JSON code block, strictly following the following format:
```json
{
    "relationships": [
        {"source": id_of_source_node, "sink": id_of_sink_node, "relation": "enables" or "blocks"},
        {"source": id_of_source_node, "sink": id_of_sink_node, "relation": "enables" or "blocks"},
        ...
    ]
}
```

"relation" is "enables" if the source brings about or increases the sink, and "blocks" if the source prevents or reduces it.

You will be provided with the source markdown text and the name of each node in the graph. Ensure that each node is included at least once in the generated causal graph. Do not use the node's name in the graph; instead, use the id corresponding to the node. For the example nodes below (not the same as the ones you will be provided), whenever you want to include the node named "demand" in your graph, you would use the integer 2 rather than the word demand.
```json
{
    "nodes": [
        {"name": "supply", "id": 1},
        {"name": "demand", "id": 2},
        ...
    ]
}
```

When answering, do not provide any additional reasoning, commentary, or other information - only provide the JSON code block, with each dictionary representing one relationship in the graph.

Here are the nodes for your graph:
```json
NODE_JSON
```"""


def log_inference(model, prompt_tokens=None, completion_tokens=None, questions=None, chars=None):
    """One model call to $INFER_LOG (JSON lines), as pipeline/kev_client.log_inference."""
    path = os.environ.get("INFER_LOG")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"model": model, "step": os.environ.get("INFER_STEP"), "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens, "questions": questions, "chars": chars}) + "\n")


def chat(messages, max_tokens=16384, temperature=0.0, logprobs=False):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}}    # model thinking off (Qwen); ReCITE's <think> is in the text
    if logprobs:
        body.update({"logprobs": True, "top_logprobs": 10})
    data = json.dumps(body).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request(URL, data=data, headers={"content-type": "application/json",
                                                                  "authorization": "Bearer EMPTY"})
            with urllib.request.urlopen(req, timeout=1800) as r:   # ReCITE's long reasoning (up to 16,384 tokens) can pass 10 min when many stories run at once
                resp = json.loads(r.read())
                u = resp.get("usage") or {}
                log_inference("generator", u.get("prompt_tokens"), u.get("completion_tokens"), chars=len(data))
                return resp["choices"][0]
        except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError) as e:
            print(f"[retry] request failed ({type(e).__name__}), attempt {attempt + 1}/6", file=sys.stderr, flush=True)
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def parse_json(text):
    """The last JSON object in the answer (after any <think> block or code fence)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S) or re.findall(r"(\{.*\})", text, flags=re.S)
    for b in reversed(blocks):
        try:
            return json.loads(b)
        except json.JSONDecodeError:
            continue
    return {}


class NoAnswer(RuntimeError):
    """Every try returned no usable JSON: the story (or step) is marked failed, never kept as an empty graph."""


def ask_json(system, user, tries=3, what=""):
    for t in range(tries):
        t0 = time.time()
        ch = chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                  temperature=0.0 if t == 0 else 0.7)
        obj = parse_json(ch["message"].get("content"))
        if obj:
            return obj
        why = "cut at the token limit" if ch.get("finish_reason") == "length" else "no valid JSON"
        print(f"[retry] {what or 'answer'}: {why} after {time.time() - t0:.0f} s, try {t + 1}/{tries}",
              file=sys.stderr, flush=True)
    raise NoAnswer(f"{what or 'answer'}: no valid JSON after {tries} tries")


def graph(sid, text, nodes, kinds, edges, tag, extra=None):
    """Pipeline graph format. 'asked' holds the probability of each ordered pair's relation, which OUR
    aggregation averages: an arrow the method gives = 1 for its relation; no arrow = none."""
    have = {(h, t): (r, p) for h, t, r, p in edges}
    asked = []
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            r, p = have.get((a, b), ("none", 1.0))
            probs = {"enables": 0.0, "blocks": 0.0, "none": 0.0}
            if r == "none":
                probs["none"] = 1.0
            else:
                probs[r] = p; probs["none"] = round(1 - p, 4)
            asked.append({"head": a, "tail": b, "probs": probs, "reps": [probs]})
    g = {"tag": tag, "id": sid, "text": text, "nodes": nodes, "kinds": kinds, "chains": [], "connected": True,
         "edges": [{"head": h, "tail": t, "rel": r, "prob": p} for h, t, r, p in edges], "asked": asked}
    g.update(extra or {})
    return g


def rel_of(x):
    x = str(x or "").lower()
    return "blocks" if ("block" in x or "prevent" in x or "reduc" in x or "decreas" in x) else "enables"


# ------------------------------------------------------------------ C1 naive
def c1(it, _nodes):
    obj = ask_json(RECITE + "\n\n" + TOP_K.format(k=K), it["text"], what=f"C1 {it['id']}")
    edges, nodes = [], []
    for r in obj.get("relationships", []):
        s, t = str(r.get("source", "")).strip(), str(r.get("sink", "")).strip()
        if s and t and s != t:
            edges.append((s, t, rel_of(r.get("relation")), 1.0))
            nodes += [s, t]
    nodes = list(dict.fromkeys(nodes))
    return graph(it["id"], it["text"], nodes, {n: "event" for n in nodes}, edges, "C1")


# ------------------------------------------------------------------ C2 whole graph on our nodes
def c2(it, nodes_of):
    nodes, kinds = nodes_of[it["id"]]
    ids = {i + 1: n for i, n in enumerate(nodes)}
    prompt = RECITE_NODES.replace("NODE_JSON", json.dumps({"nodes": [{"name": n, "id": i} for i, n in ids.items()]}, indent=1))
    obj = ask_json(prompt, it["text"], what=f"C2 {it['id']}")
    edges = []
    for r in obj.get("relationships", []):
        try:
            s, t = ids[int(r.get("source"))], ids[int(r.get("sink"))]
        except (KeyError, TypeError, ValueError):
            continue
        if s != t:
            edges.append((s, t, rel_of(r.get("relation")), 1.0))
    return graph(it["id"], it["text"], nodes, kinds, edges, "C2")


# ------------------------------------------------------------------ C3 pairwise (Kiciman et al.)
PAIR = """Text:
{text}

Considering only what the text says happened, which statement is true?
A) "{a}" causes or enables "{b}".
B) "{a}" prevents or blocks "{b}".
C) "{a}" has no direct causal effect on "{b}".
Answer with the single letter A, B or C."""


def pair_probs(text, a, b):
    ch = chat([{"role": "user", "content": PAIR.format(text=text, a=a, b=b)}], max_tokens=1, logprobs=True)
    lp = {}
    for t in ((ch.get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs", []):
        k = t["token"].strip().upper()[:1]
        if k in "ABC" and k not in lp:
            lp[k] = t["logprob"]
    if not lp:
        k = (ch["message"].get("content") or "C").strip()[:1].upper()
        lp = {k if k in "ABC" else "C": 0.0}
    z = sum(math.exp(v) for v in lp.values())
    p = {k: math.exp(lp[k]) / z if k in lp else 0.0 for k in "ABC"}
    return {"enables": p["A"], "blocks": p["B"], "none": p["C"]}


def c3(it, nodes_of):
    nodes, kinds = nodes_of[it["id"]]
    edges, asked = [], []
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            pr = pair_probs(it["text"], a, b)
            asked.append({"head": a, "tail": b, "probs": pr, "reps": [pr]})
            best = max(pr, key=pr.get)
            if best != "none":
                edges.append((a, b, best, pr[best]))
    g = graph(it["id"], it["text"], nodes, kinds, edges, "C3")
    g["asked"] = asked                                     # the real pairwise probabilities
    return g


# ------------------------------------------------------------------ C4 BFS (Jiralerspong et al.)
BFS_ROOTS = """Text:
{text}

Variables: {vars}

Which of these variables are not caused by any other variable in the list, according to the text?
Answer only with JSON: {{"roots": ["<variable>", ...]}}"""
BFS_EXPAND = """Text:
{text}

Variables: {vars}

According to the text, which of the variables does "{x}" directly cause or enable, and which does it directly prevent or block?
Answer only with JSON: {{"enables": ["<variable>", ...], "blocks": ["<variable>", ...]}}"""


def c4(it, nodes_of):
    nodes, kinds = nodes_of[it["id"]]
    vs = json.dumps(nodes)
    roots = [r for r in ask_json("You are an expert causal reasoner.", BFS_ROOTS.format(text=it["text"], vars=vs), what=f"C4 roots {it['id']}").get("roots", [])
             if r in nodes] or nodes[:1]
    queue, seen, edges = list(roots), set(roots), []
    while queue:
        x = queue.pop(0)
        obj = ask_json("You are an expert causal reasoner.", BFS_EXPAND.format(text=it["text"], vars=vs, x=x), what=f"C4 expand {it['id']}")
        for rel in RELS:
            for y in obj.get(rel, []) or []:
                if y in nodes and y != x:
                    edges.append((x, y, rel, 1.0))
                    if y not in seen:
                        seen.add(y); queue.append(y)
    return graph(it["id"], it["text"], nodes, kinds, edges, "C4")


# ------------------------------------------------------------------ C5 Causal Map claim coding
CMAP = """You are helping with causal mapping of qualitative data. Read the text and identify every causal claim it makes: each place where the narrator says that one thing influenced another. For each claim give a short label for the influence factor (cause), a short label for the consequence factor (effect), and whether the influence made the consequence happen or more likely ("increases") or stopped it or made it less likely ("decreases"). Use short, general labels that could be reused across many texts.
Answer only with JSON: {"claims": [{"cause": "<label>", "effect": "<label>", "direction": "increases" or "decreases"}, ...]}"""


def c5(it, _nodes):
    obj = ask_json(CMAP + "\n" + TOP_K.format(k=K), it["text"], what=f"C5 {it['id']}")
    edges, nodes = [], []
    for c in obj.get("claims", []):
        s, t = str(c.get("cause", "")).strip(), str(c.get("effect", "")).strip()
        if s and t and s != t:
            edges.append((s, t, rel_of(c.get("direction")), 1.0))
            nodes += [s, t]
    nodes = list(dict.fromkeys(nodes))
    return graph(it["id"], it["text"], nodes, {n: "event" for n in nodes}, edges, "C5")


# ------------------------------------------------------------------ C6 whole corpus
CORPUS = """You are an expert causal reasoner. Below are {n} first-person stories from different people about the same topic. Build ONE causal graph of the world these stories come from: the events, conditions and actors (people or organisations who act) that recur across the stories, and the direct causal arrows between them. An arrow "enables" if the source brings about the target, and "blocks" if the source prevents it.
{top_k}
Answer only with JSON: {{"nodes": [{{"name": "<short label>", "kind": "event" or "participant"}}, ...], "edges": [{{"source": "<name>", "target": "<name>", "relation": "enables" or "blocks"}}, ...]}}

STORIES:
{stories}"""
PRESENT = """Text:
{text}

Which of these appear in the text (as something that happened, or as an actor who acted)? {names}
Answer only with JSON: {{"present": ["<name>", ...]}}"""


def c6(items, out, workers):
    stories = "\n\n".join(f"[{it['id']}] {it['text']}" for it in items)
    obj = ask_json("You are an expert causal reasoner.", CORPUS.format(n=len(items), stories=stories, top_k=TOP_K.format(k=K)), what="C6 world graph")
    names = [str(n.get("name", "")).strip() for n in obj.get("nodes", []) if str(n.get("name", "")).strip()]
    kind = {str(n.get("name", "")).strip(): ("participant" if str(n.get("kind", "")).startswith("p") else "event")
            for n in obj.get("nodes", [])}
    arrows = [{"head": e.get("source"), "tail": e.get("target"), "rel": rel_of(e.get("relation")), "estimate": 1.0}
              for e in obj.get("edges", []) if e.get("source") in names and e.get("target") in names]

    def pres(it):                                          # which world nodes each story contains (same LLM)
        try:
            got = ask_json("You are a careful reader.", PRESENT.format(text=it["text"], names=json.dumps(names)),
                           what=f"C6 present {it['id']}").get("present", [])
        except NoAnswer:
            failed.append(it["id"]); got = []                 # marked failed below, not silently "absent"
        return it["id"], {n: int(n in got) for n in names}
    failed = []
    with ThreadPoolExecutor(workers) as ex:
        table = dict(ex.map(pres, items))
    concepts = {n: {"kind": kind.get(n, "event"), "members": [[sid, n, 1] for sid, row in table.items() if row[n]]}
                for n in names}
    json.dump({"concepts": concepts, "source": "C6", "failed_stories": sorted(failed)}, open(out / "abstraction.json", "w"), indent=1)
    if failed:
        print(f"[failed] C6 present: {len(failed)} stories without an answer: {sorted(failed)}", file=sys.stderr, flush=True)
    json.dump({"table": table, "network": {"nodes": [{"id": n, "kind": kind.get(n, "event")} for n in names],
                                           "arrows": arrows}}, open(out / "network.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method"); ap.add_argument("texts"); ap.add_argument("out")
    ap.add_argument("--nodes", help="chain_nodes.json from our node extraction (C2-C4)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items = json.load(open(a.texts))
    t0 = time.time()
    if a.method == "C6":
        c6(items, out, a.workers)
    else:
        nodes_of = {}
        if a.method in ("C2", "C3", "C4"):
            for it in json.load(open(a.nodes)):
                nodes_of[it["id"]] = (list(it["nodes"]), it.get("kinds") or {n: "event" for n in it["nodes"]})
        fn = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}[a.method]

        def one(it):
            try:
                g = fn(it, nodes_of)
            except Exception as e:                         # a story that fails gets an empty graph (scored as a miss)
                g = graph(it["id"], it["text"], [], {}, [], a.method, {"failed": f"{type(e).__name__}: {e}"[:300]})
            json.dump(g, open(out / f"{it['id']}__gemma-kev.chain.json", "w"), indent=1)
            return g
        with ThreadPoolExecutor(a.workers) as ex:
            gs = list(ex.map(one, items))
        print(f"{a.method}: {len(gs)} graphs, {sum(1 for g in gs if g.get('failed'))} failed")
    json.dump({"method": a.method, "llm": MODEL, "seconds_method": round(time.time() - t0, 1)},
              open(out / "baseline_info.json", "w"), indent=1)


if __name__ == "__main__":
    main()
