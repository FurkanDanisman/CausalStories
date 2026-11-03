#!/usr/bin/env python3
"""Aggregation alternatives (benchmark/workplan.md §3), plus the aggregation steps of C5 (Causal Map)
and C7 (PC / GES). Each reads story graphs and writes a world graph (abstraction.json + network.json)
that evaluate_run.py scores with Table 2.

Same input for A1-A5: OUR story graphs and OUR groups from one run (<run>/abstraction.json:
concepts with their member nodes, concept_edges per story). A concept is present in a story when the
story has one of its member nodes.

  A1 freq     frequency map (Causal Map overview): keep a link seen in >= MIN_LINK stories; sign = majority
  A2 fcm      fuzzy cognitive map averaging (Ozesmi & Ozesmi 2004): mean signed weight (+p enables, -p blocks)
              over all stories; keep |mean| >= FCM_MIN; sign = sign of the mean
  A3 median   consensus graph: keep a link present in more than half of the stories that contain both ends
  A4 llmmerge an LLM merges all story graphs (concept names) into one graph  (needs LLM_URL / LLM_MODEL)
  A5 pc / ges classical causal discovery on the story x concept table (causal-learn); sign = sign of the
              association; undirected CPDAG edges are oriented by concept name order (stated, not tuned)
  C5agg       Causal Map aggregation of C5's claim labels: sentence-embedding clustering of the labels, then A1
Usage: python benchmark/alt_aggregate.py <A1|A2|A3|A4|pc|ges|C5agg> <run_dir> <out_dir>"""
import json, math, os, sys
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from collections import Counter, defaultdict
from pathlib import Path

MIN_LINK = 10          # A1 / C5agg: a link must appear in at least 10 of the 200 stories (5%)
FCM_MIN = 0.05         # A2: |mean weight| over the 200 stories (~ 10 stories at full weight)


def load_run(run):
    abst = json.load(open(run / "abstraction.json"))
    concepts = abst["concepts"]
    stories = sorted({m[0] for c in concepts.values() for m in c["members"]} |
                     {json.load(open(f))["id"] for f in run.glob("*__gemma-kev.chain.json") if not f.name.startswith("._")})
    table = {s: {c: 0 for c in concepts} for s in stories}
    for c, v in concepts.items():
        for m in v["members"]:
            table[m[0]][c] = 1
    edges = defaultdict(list)            # story -> [(head concept, tail concept, rel, prob)]
    for s, es in abst.get("concept_edges", {}).items():
        for e in es:
            if e["head"] != e["tail"] and e["head"] in concepts and e["tail"] in concepts:
                edges[s].append((e["head"], e["tail"], e["rel"], float(e.get("prob", 1.0))))
    return concepts, table, edges


def write(out, concepts, table, arrows, source):
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"concepts": concepts, "source": source}, open(out / "abstraction.json", "w"), indent=1)
    json.dump({"table": table, "network": {"nodes": [{"id": c, "kind": v["kind"]} for c, v in concepts.items()],
                                           "arrows": arrows}}, open(out / "network.json", "w"), indent=1)
    print(f"{source}: {len(concepts)} concepts, {len(arrows)} arrows -> {out}")


def freq(edges, n_min=MIN_LINK):
    cnt = defaultdict(Counter)
    for es in edges.values():
        for h, t, r, p in set((h, t, r, 1.0) for h, t, r, _ in es):
            cnt[(h, t)][r] += 1
    return [{"head": h, "tail": t, "rel": c.most_common(1)[0][0], "estimate": sum(c.values()) / len(edges)}
            for (h, t), c in cnt.items() if sum(c.values()) >= n_min]


def fcm(edges, n_stories):
    w = defaultdict(float)
    for es in edges.values():
        seen = {}                                              # one weight per link per story: the strongest
        for h, t, r, p in es:
            v = p if r == "enables" else -p
            if abs(v) > abs(seen.get((h, t), 0.0)):
                seen[(h, t)] = v
        for k, v in seen.items():
            w[k] += v
    return [{"head": h, "tail": t, "rel": "enables" if v > 0 else "blocks", "estimate": abs(v) / n_stories}
            for (h, t), v in w.items() if abs(v) / n_stories >= FCM_MIN]


def median(edges, table):
    cnt = defaultdict(Counter)
    for es in edges.values():
        for h, t, r, _ in set(es):
            cnt[(h, t)][r] += 1
    out = []
    for (h, t), c in cnt.items():
        both = sum(1 for row in table.values() if row[h] and row[t])
        if both and sum(c.values()) > both / 2:
            out.append({"head": h, "tail": t, "rel": c.most_common(1)[0][0], "estimate": sum(c.values()) / both})
    return out


def causal_discovery(table, concepts, method):
    import numpy as np
    keep = [c for c in concepts if 0.05 <= sum(r[c] for r in table.values()) / len(table) < 1.0]   # used, not constant
    X = np.array([[r[c] for c in keep] for r in table.values()], dtype=float)
    if method == "pc":
        from causallearn.search.ConstraintBased.PC import pc
        G = pc(X, 0.05, "chisq", show_progress=False).G.graph
    else:
        from causallearn.search.ScoreBased.GES import ges
        G = ges(X, score_func="local_score_BIC")["G"].graph
    arrows = []
    for i in range(len(keep)):
        for j in range(len(keep)):
            if i == j:
                continue
            directed = G[j, i] == 1 and G[i, j] == -1                       # i -> j
            undirected = G[i, j] == -1 and G[j, i] == -1 and keep[i] < keep[j]
            if directed or undirected:
                corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
                arrows.append({"head": keep[i], "tail": keep[j], "rel": "enables" if corr >= 0 else "blocks",
                               "estimate": abs(float(np.nan_to_num(corr)))})
    return arrows


def llm_merge(concepts, edges):
    sys.path.insert(0, str(Path(__file__).parent))
    from baselines import ask_json, rel_of
    lines = []
    for s, es in sorted(edges.items()):
        if es:
            lines.append(f"[{s}] " + "; ".join(f"{h} {'->' if r == 'enables' else '-|'} {t}" for h, t, r, _ in es))
    prompt = (f"Below are {len(lines)} causal graphs, one per person, over these factors: {json.dumps(sorted(concepts))}. "
              "'A -> B' means A enables B; 'A -| B' means A blocks B. Merge them into ONE causal graph of the "
              "world they come from: keep the factors and arrows that the graphs support, drop noise.\n"
              'Answer only with JSON: {"edges": [{"source": "<factor>", "target": "<factor>", "relation": "enables" or "blocks"}, ...]}\n\n'
              + "\n".join(lines))
    obj = ask_json("You are an expert causal reasoner.", prompt)
    return [{"head": e.get("source"), "tail": e.get("target"), "rel": rel_of(e.get("relation")), "estimate": 1.0}
            for e in obj.get("edges", []) if e.get("source") in concepts and e.get("target") in concepts]


def c5agg(run, out):
    """Causal Map aggregation: cluster C5's free-text claim labels by meaning, then count links."""
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    gs = [json.load(open(f)) for f in sorted(run.glob("*__gemma-kev.chain.json")) if not f.name.startswith("._")]
    labels = sorted({n for g in gs for n in g["nodes"]})
    emb = SentenceTransformer(os.environ.get("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2"), device="cpu").encode(labels, normalize_embeddings=True)
    cl = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, metric="cosine", linkage="average").fit(emb).labels_
    groups = defaultdict(list)
    for l, k in zip(labels, cl):
        groups[k].append(l)
    count = Counter(n for g in gs for n in set(g["nodes"]))
    name = {k: max(v, key=lambda l: count[l]) for k, v in groups.items()}       # cluster name = its most used label
    of = {l: name[k] for l, k in zip(labels, cl)}
    concepts = {nm: {"kind": "event", "members": []} for nm in name.values()}
    table = {g["id"]: {c: 0 for c in concepts} for g in gs}
    edges = defaultdict(list)
    for g in gs:
        for n in g["nodes"]:
            concepts[of[n]]["members"].append([g["id"], n, 1]); table[g["id"]][of[n]] = 1
        for e in g["edges"]:
            if of[e["head"]] != of[e["tail"]]:
                edges[g["id"]].append((of[e["head"]], of[e["tail"]], e["rel"], 1.0))
    write(out, concepts, table, freq(edges), "C5 Causal Map")


def main():
    method, run, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if method == "C5agg":
        return c5agg(run, out)
    concepts, table, edges = load_run(run)
    n = len(table)
    arrows = {"A1": lambda: freq(edges), "A2": lambda: fcm(edges, n), "A3": lambda: median(edges, table),
              "A4": lambda: llm_merge(concepts, edges), "pc": lambda: causal_discovery(table, concepts, "pc"),
              "ges": lambda: causal_discovery(table, concepts, "ges")}[method]()
    write(out, concepts, table, arrows, method)


if __name__ == "__main__":
    main()
