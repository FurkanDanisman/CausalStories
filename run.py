#!/usr/bin/env python3
"""Causal Stories pipeline: the agreed method (method_Oct.tex; results/api_world100_matrix).

Section 1 (one story -> one graph):
  chain-nodes    generator LLM: the K most important events + the participants   -> chain_nodes.json
  chain-kev      decision model: chain walk (event->event, participant->event) and the
                 separate influence walk (event->participant)                     -> <id>__<tag>.chain.json
Section 2 (story graphs -> one graph):
  merge-kev      Step 1: decision model groups nodes, one story at a time         -> abstraction.json
  merge-llm      Step 1: LLM groups nodes in batches, merged in a tournament       -> abstraction.json
  name-groups    Step 1: generator names the groups (and their opposites)         -> abstraction.json
  node-happened  Step 3: did each named event node happen?                         -> <id>__happened.json
  implied        Step 3: did each group a story does not name happen? (1/0/cannot tell; no imputation)
                                                                                    -> implied.json
  connect        Step 4: inferred groups connected on top of each story graph      -> <id>__connect.json
  matrix         Steps 5-6 (no model): usage, matrix of averaged answers, graph with intervals
                                                                                    -> network.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import chain, combine, matrix, merge_kev, prompts
from pipeline.kev_client import KevClient
from pipeline.llm_client import (AnthropicClient, HFClient, OpenRouterClient,
                                 OllamaClient, OpenAIClient, VLLMClient)


def build_client(backend: str, model: str | None, base_url: str | None = None):
    if backend == "anthropic":
        return AnthropicClient(model=model or "REPLACE_WITH_MODEL_ID")
    if backend == "openai":                      # also: any vLLM/TGI/llama.cpp server via base_url
        return OpenAIClient(model=model or "gpt-4o", base_url=base_url)
    if backend == "openrouter":                  # hosted Gemma etc., key in OPENROUTER_API_KEY
        return OpenRouterClient(model=model or "google/gemma-4-31b-it")
    if backend == "ollama":
        return OllamaClient(model=model or "qwen2.5:14b")
    if backend == "hf":                          # in-process transformers
        return HFClient(model=model or "Qwen/Qwen2.5-7B-Instruct")
    if backend == "vllm":                        # in-process vLLM, schema-guided JSON
        return VLLMClient(model=model)
    raise ValueError(f"unknown backend: {backend}")


def _tag(args) -> str:
    return args.tag or (args.model or args.backend).replace("/", "_")


def do_chain_nodes(args, outdir: Path) -> None:
    """Chain walk, LLM part (e.g. Gemma): the K most important events of each narrative
    in --texts-file, and its participants. Writes <outdir>/chain_nodes.json, the input
    for --mode chain-kev."""
    items = json.loads(Path(args.texts_file).read_text())
    client = build_client(args.backend, args.model, args.base_url)
    print(f"=== CHAIN-NODES · {_tag(args)} · {len(items)} narrative(s) · K={args.max_nodes} ===")
    out = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as pool:          # stories are independent
        found = list(pool.map(lambda it: chain.llm_nodes(client, it["text"], args.max_nodes), items))
    for it, (kinds, raw) in zip(items, found):
        ids = list(kinds)
        print(f"\n[{it['id']}] {it['text']}")
        print("  Gemma, event call:       " + ", ".join(f"{x['id']} ({x['kind']})" for x in raw["events_call"]))
        print("  Gemma, participant call: " + ", ".join(f"{x['id']} ({x['kind']})" for x in raw["participants_call"]))
        for n in ids:
            print(f"  {n!r} ({kinds[n]})")
        out.append({"id": it["id"], "text": it["text"], "nodes": ids, "kinds": kinds, "raw": raw})
    (outdir / "chain_nodes.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outdir}/chain_nodes.json")


def do_chain_kev(args, outdir: Path) -> None:
    """Chain walk, Kev part: walk and connectivity for each narrative in --nodes-file
    ({id, text, nodes, kinds}). Needs a running Kev server at --kev-url."""
    items = json.loads(Path(args.nodes_file).read_text())
    tag = args.tag or "kev"
    print(f"=== CHAIN-KEV · {tag} · {len(items)} narrative(s) · {args.kev_url} ===")

    def one(it):                                   # own client per story: request counts stay per story
        buf = []
        g = chain.run_chain(KevClient(args.kev_url), it["text"], it["nodes"], seed=args.seed,
                            kinds=it.get("kinds"), log=buf.append)
        return g, buf

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        done = list(pool.map(one, items))
    for it, (g, buf) in zip(items, done):
        print(f"\n[{it['id']}] {it['text']}")
        print("\n".join(buf))
        rel = {(h, t): r for h, t, r, _ in g.edges}
        print("SUBGRAPHS (every chain the walk produced)")
        for k, c in enumerate(g.chains, 1):
            path = c[0] + "".join(f" {'-|' if rel[(h, t)] == 'blocks' else '->'} {t}" for h, t in zip(c, c[1:]))
            print(f"  {k}. {path}")
        print("FULL GRAPH")
        for h, t, r, p in g.edges:
            print(f"  {h} {'-|' if r == 'blocks' else '->'} {t}   p={p:.2f}")
        print(f"  Kev requests: {g.kev_requests} · connected: {g.connected}")
        data = {"tag": tag, "id": it["id"], "text": it["text"], **g.to_json()}
        (outdir / f"{it['id']}__{tag}.chain.json").write_text(json.dumps(data, indent=2))
    print(f"\nsaved *__{tag}.chain.json in {outdir}")


def do_merge_kev(args, outdir: Path) -> None:
    """Section 2, Steps 1-2 (Kev): group the nodes of all narrative graphs (--graphs-glob)
    one narrative at a time, with opposite members flagged. Writes <outdir>/abstraction.json."""
    graphs = _graphs(args)
    print(f"=== MERGE (Kev) · {len(graphs)} graph(s) · {args.kev_url} ===")
    res = merge_kev.kev_merge(KevClient(args.kev_url), graphs)
    print("GROUPS")
    for c, d in res["concepts"].items():
        print(f"  [{d['kind']}] {c}: " + "; ".join(f"{m[0]}: {m[1]}{' (opposite)' if m[2] < 0 else ''}"
                                                   for m in d["members"]))
    (outdir / "abstraction.json").write_text(json.dumps(res, indent=2))
    print(f"\nsaved {outdir}/abstraction.json")


def do_merge_llm(args, outdir: Path) -> None:
    """Section 2, Step 1 with an LLM (pipeline/llm_group.py): batches of --batch-size stories, merged in a
    tournament, with Step 1's rules checked after every call. Writes <outdir>/abstraction.json."""
    from pipeline import llm_group
    graphs = _graphs(args)
    print(f"=== MERGE (LLM, batches of {args.batch_size}) · {len(graphs)} graph(s) ===")
    import os
    os.environ.setdefault("LLM_GROUP_LOG", str(outdir / "llm_group_answers.jsonl"))   # every prompt and raw answer
    res = llm_group.llm_group(graphs, args.batch_size)
    for c, d in res["concepts"].items():
        print(f"  [{d['kind']}] {c}: " + "; ".join(f"{m[0]}: {m[1]}{' (opposite)' if m[2] < 0 else ''}" for m in d["members"]))
    (outdir / "abstraction.json").write_text(json.dumps(res, indent=2))
    print(f"\nsaved {outdir}/abstraction.json")


def do_name_groups(args, outdir: Path) -> None:
    """Section 2, Step 1 naming (LLM): high-level names for the groups in
    <outdir>/abstraction.json, written back in place."""
    abst = json.loads((outdir / "abstraction.json").read_text())
    client = build_client(args.backend, args.model, args.base_url)
    print(f"=== NAME GROUPS · {_tag(args)} · {len(abst['concepts'])} group(s) ===")
    abst = merge_kev.name_groups(client, abst)
    (outdir / "abstraction.json").write_text(json.dumps(abst, indent=2))
    print(f"\nsaved {outdir}/abstraction.json")


def do_node_happened(args, outdir: Path) -> None:
    """For each story graph (--graphs-glob): Kev/Jev answers "did this happen in this story?"
    (kev_happened.md) for every event node. Writes <outdir>/<id>__happened.json."""
    graphs = _graphs(args)
    print(f"=== NODE HAPPENED · {len(graphs)} graph(s) · {args.kev_url} ===")

    def one(g):
        ev = [n for n in g["nodes"] if (g.get("kinds") or {}).get(n, "event") == "event"]
        if not ev:                                  # no event nodes: nothing to ask
            return {}
        ans = KevClient(args.kev_url).ask(g["text"], {f"n{i}": prompts.kev_happened_question(v) for i, v in enumerate(ev)})
        return {v: {"choice": ans[f"n{i}"]["choice"], "probabilities": ans[f"n{i}"]["probabilities"]} for i, v in enumerate(ev)}

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        res = list(pool.map(one, graphs))
    for g, r in zip(graphs, res):
        print(f"[{g['id']}] " + "; ".join(f"{v}: {a['choice']} ({a['probabilities'].get(a['choice'], 0):.2f})" for v, a in r.items()))
        (outdir / f"{g['id']}__happened.json").write_text(json.dumps(r, indent=2))


def _graphs(args) -> list[dict]:
    import glob
    files = sorted(glob.glob(args.graphs_glob))
    if not files:
        raise SystemExit(f"no graphs match {args.graphs_glob}")
    return [json.loads(Path(f).read_text()) for f in files]


def do_implied(args, outdir: Path) -> None:
    """Section 2, Step 4 (Kev): missing event concepts -> happened / not happened /
    cannot tell. Writes <outdir>/implied.json."""
    graphs = _graphs(args)
    abst = json.loads((outdir / "abstraction.json").read_text())
    print(f"=== IMPLIED · {len(graphs)} graph(s) · {args.kev_url} ===")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as pool:          # stories are independent
        parts = list(pool.map(lambda g: combine.implied(KevClient(args.kev_url), [g], abst, seed=args.seed), graphs))
    res = {k: v for part in parts for k, v in part.items()}
    for j, d in res.items():
        print(f"  {j}: " + "  ".join(f"{c}={'.' if v is None else v}" for c, v in d["values"].items()))
    (outdir / "implied.json").write_text(json.dumps(res, indent=2))
    print(f"saved {outdir}/implied.json")


def do_connect(args, outdir: Path) -> None:
    """Connect the inferred groups to each story graph, on top of the existing graph. New
    nodes = the event groups a story does not name but Jev inferred as happened / not
    happened (implied.json; "cannot tell" is left out), shown by their group name.
    Walk (the chain walk's rule): start from a random node (existing or new); when a node x is
    first reached it is asked only about nodes NOT on the current path that cannot already reach
    x in the story's current graph (a new node about every such node, an existing node only about
    the new ones; its existing arrows are reused, not asked again); follow every child; restart
    from a random node no path has reached, until every node is on a path. The story's graph
    stays acyclic (method_Oct.tex, Theorem 3). Existing arrows are never changed. New event -> participant arrows
    come from the separate influence question, as in the influence walk.
    Writes <outdir>/<id>__connect.json."""
    import random
    graphs = _graphs(args)
    abst = json.loads((outdir / "abstraction.json").read_text())
    imp = json.loads((outdir / "implied.json").read_text())
    print(f"=== CONNECT (walk on top of the existing graphs) · {len(graphs)} graph(s) · {args.kev_url} ===")

    def one(g):
        j, kinds = g["id"], dict(g.get("kinds") or {})
        present = {c for c, d in abst["concepts"].items() if any(m[0] == j for m in d["members"])}
        new = {}
        for c, d in abst["concepts"].items():
            v = imp.get(j, {}).get("values", {}).get(c)
            if d["kind"] == "event" and c not in present and v in (0, 1):
                name = c if c not in g["nodes"] else f"{c} (group)"
                new[name] = {"group": c, "value": v}
        if not new:
            return {"new_nodes": {}, "edges": [], "asked": [], "chains": []}
        ids = list(g["nodes"]) + list(new)
        kinds.update({n: "event" for n in new})
        kev = KevClient(args.kev_url)
        old = {}
        for e in g["edges"]:
            if kinds.get(e["tail"], "event") != "participant":          # existing chain arrows, reused
                old.setdefault(e["head"], []).append((e["tail"], e["rel"], e["prob"]))
        children, asked, added, log = {}, [], [], []
        out = {n: {t for t, _, _ in old.get(n, [])} for n in ids}      # the story's current graph (no influence arrows)

        def reaches(y, x):                        # can y already reach x in the story's current graph?
            seen, stack = set(), [y]
            while stack:
                u = stack.pop()
                if u == x:
                    return True
                if u not in seen:
                    seen.add(u); stack += list(out.get(u, ()))
            return False

        def ask(path):
            x = path[-1]
            if x not in children:
                # the chain walk's rule, written so it also covers the existing arrows: x is asked only about
                # nodes off its path that cannot already reach x (method_Oct.tex, Step 4); every question
                # involves a new node (an existing node is asked only about new nodes)
                cand = [y for y in ids if y not in path and (x in new or y in new) and not reaches(y, x)]
                res = chain.kev_arrows(kev, g["text"], ids, [(x, y) for y in cand], kinds) if cand else []
                got = [(t, c, p[c]) for _, t, c, p, _ in res if c != "none"]
                asked.extend({"head": x, "tail": t, "probs": {k: round(v, 4) for k, v in p.items()}}
                             for _, t, _, p, _ in res)
                added.extend({"head": x, "tail": t, "rel": c, "prob": round(pr, 3)} for t, c, pr in got)
                children[x] = old.get(x, []) + got
                out[x].update(t for t, _, _ in got)
                log.append(f"  ask {x!r} (path {' > '.join(path)}): " +
                           (", ".join(f"{t} {c} {pr:.2f}" for t, c, pr in got) or "no new arrows"))
            return children[x]

        chains, reached, closed = [], set(), set()

        def is_closed(t):
            # every node t can reach (through answers already given) has been asked: following t
            # cannot ask anything new, it would only list more paths (their number grows exponentially)
            if t in closed:
                return True
            seen, stack = set(), [t]
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                if u not in children:
                    return False
                seen.add(u)
                stack += [w for w, _, _ in children[u]]
            closed.update(seen)
            return True

        def expand(path):
            reached.update(path)
            kids = [t for t, _, _ in ask(path) if t not in path]
            if not kids:
                chains.append(path)
                return
            for t in kids:
                if is_closed(t):                  # same questions and answers as following it; path not listed
                    reached.add(t)
                    continue
                expand(path + [t])

        rng = random.Random(args.seed)
        unvisited = list(ids)                                         # start from a random node, existing or new
        while unvisited:
            start = rng.choice(unvisited)
            log.append(f"  start at {start!r}")
            expand([start])
            unvisited = [n for n in ids if n not in reached]
        # new event -> participant: the influence question, as in the influence walk
        parts = [n for n in g["nodes"] if kinds.get(n) == "participant"]
        pairs = [(e, p) for e in new for p in parts]
        if pairs:
            for h, t, c, p, _ in chain.kev_arrows(kev, g["text"], ids, pairs, kinds, into_participants=True):
                asked.append({"head": h, "tail": t, "probs": {k: round(v, 4) for k, v in p.items()}})
                if c != "none":
                    added.append({"head": h, "tail": t, "rel": c, "prob": round(p[c], 3)})
        return {"new_nodes": new, "edges": added, "asked": asked, "chains": chains, "log": log}

    from concurrent.futures import ThreadPoolExecutor
    import threading
    lock = threading.Lock()
    todo = [g for g in graphs if not (outdir / f"{g['id']}__connect.json").exists()]
    if len(todo) < len(graphs):                   # resumed after a time limit: finished stories are kept
        print(f"resuming: {len(graphs) - len(todo)} of {len(graphs)} stories already connected")

    def run_one(g):                               # each story is saved as soon as it is done
        r = one(g)
        tmp = outdir / f"{g['id']}__connect.json.tmp"
        tmp.write_text(json.dumps(r, indent=2)); tmp.replace(outdir / f"{g['id']}__connect.json")
        with lock:
            print(f"[{g['id']}] new: " + ", ".join(f"{n}={d['value']}" for n, d in r["new_nodes"].items()))
            print("\n".join(r.get("log", [])))
            for e in r["edges"]:
                print(f"    + {e['head']} {'-|' if e['rel'] == 'blocks' else '->'} {e['tail']}   p={e['prob']:.2f}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_one, todo))


def do_matrix(args, outdir: Path) -> None:
    """Section 2, Steps 5-6 (no model): usage, the averaged matrix and the graph (pipeline/matrix.py).
    Writes <outdir>/network.json."""
    import time
    t0 = time.time()
    graphs = _graphs(args)
    res = matrix.aggregate(outdir, graphs)
    print(f"=== MATRIX · {len(graphs)} graph(s) · {len(res['network']['arrows'])} arrow(s) without influence ===")
    for a in res["network"]["arrows"]:
        ci = f"[{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}]" if a["ci95"] else "(n=1, no CI)"
        print(f"  {a['head']} {'-|' if a['rel'] == 'blocks' else '->'} {a['tail']}   avg {a['average']:.2f}  "
              f"est {a['estimate']:.2f} {ci}  n={a['n']}")
    print("USAGE " + ", ".join(f"{c} {u:.0%}" for c, u in sorted(res["usage"].items(), key=lambda x: -x[1])))
    (outdir / "network.json").write_text(json.dumps(res, indent=2))
    with open(outdir / "timing.tsv", "a") as f:                  # the method's time: its own steps, start to end
        f.write(f"matrix\t{time.time() - t0:.0f}\n")
    print(f"saved {outdir}/network.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=10, help="merge-llm: stories per LLM grouping call")
    ap.add_argument("--mode", required=True, choices=['chain-nodes', 'chain-kev', 'merge-kev', 'merge-llm', 'name-groups', 'node-happened', 'implied', 'connect', 'matrix'])
    ap.add_argument("--backend", default="openrouter",
                    choices=["anthropic", "openai", "openrouter", "ollama", "hf", "vllm"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--tag", default=None, help="label for output files (defaults to model name)")
    ap.add_argument("--texts-file", default=None, help="JSON [{id, text}] (chain-nodes)")
    ap.add_argument("--max-nodes", type=int, default=5, help="K: most important events (chain-nodes)")
    ap.add_argument("--seed", type=int, default=0, help="random start nodes of the walks")
    ap.add_argument("--nodes-file", default=None, help="chain_nodes.json (chain-kev)")
    ap.add_argument("--kev-url", default="http://127.0.0.1:8009", help="decision model server; Jev: https://api.typesafe.ai")
    ap.add_argument("--workers", type=int, default=1, help="stories processed in parallel")
    ap.add_argument("--graphs-glob", default=None, help="story graphs, e.g. out/*__gemma-kev.chain.json")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    {
        "chain-nodes": do_chain_nodes, "chain-kev": do_chain_kev, "merge-kev": do_merge_kev, "merge-llm": do_merge_llm,
        "name-groups": do_name_groups, "node-happened": do_node_happened, "implied": do_implied,
        "connect": do_connect, "matrix": do_matrix,
    }[args.mode](args, outdir)


if __name__ == "__main__":
    main()
