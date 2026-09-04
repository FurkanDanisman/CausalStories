"""Aggregate N per-variant causal graphs into one probability-weighted graph.

Steps (a first, LLM-driven version of the methodology.tex pipeline):
  1. canonicalize: an LLM clusters nodes from all retellings so the same event/
     participant worded differently becomes one canonical node.
  2. aggregate: relabel every edge to canonical nodes, count how many variants
     drew each directed edge, and set its probability = count / n_variants. Keep
     edges seen in at least `min_count` variants (single-variant edges are noise).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from . import prompts
from .llm_client import LLMClient
from .schema import CausalGraph, Edge, Node, NodeClustering, NodeKind


def canonicalize(client: LLMClient, base_text: str, graphs: list[CausalGraph]) -> dict[str, str]:
    nodes: dict[str, str] = {}
    for g in graphs:
        for nd in g.nodes:
            nodes.setdefault(nd.id, nd.kind.value)
    if not nodes:
        return {}
    listing = "\n".join(f"  - {nid}  ({kind})" for nid, kind in nodes.items())
    out = client.complete(task="canonicalize", schema=NodeClustering, temperature=0.0,
                          prompt=prompts.canonicalize_prompt(base_text, listing))
    mapping = dict(out.mapping)
    for nid in nodes:                 # any node the model skipped maps to itself
        mapping.setdefault(nid, nid)
    return mapping


def relabel_graph(graph: CausalGraph, node2canon: dict[str, str]) -> CausalGraph:
    """Apply the canonical-name mapping to one graph (naming standardization)."""
    def c(x: str) -> str:
        return node2canon.get(x, x)

    kinds: dict[str, Counter] = defaultdict(Counter)
    types: dict[str, set] = defaultdict(set)
    for nd in graph.nodes:
        cid = c(nd.id)
        kinds[cid][nd.kind.value] += 1
        types[cid].update(nd.event_types)
    edges = set()
    for e in graph.edges:
        h, t = c(e.head), c(e.tail)
        if h != t:
            edges.add((h, t))
    nodes = [Node(id=k, kind=NodeKind(v.most_common(1)[0][0]), event_types=sorted(types[k]))
             for k, v in kinds.items()]
    return CausalGraph(nodes=nodes, edges=[Edge(head=h, tail=t, prob=1.0) for h, t in edges])


def aggregate(graphs: list[CausalGraph], node2canon: dict[str, str],
              n_variants: int, min_count: int = 2) -> CausalGraph:
    def canon(nid: str) -> str:
        return node2canon.get(nid, nid)

    kind_votes: dict[str, Counter] = defaultdict(Counter)
    types: dict[str, set] = defaultdict(set)
    for g in graphs:
        for nd in g.nodes:
            c = canon(nd.id)
            kind_votes[c][nd.kind.value] += 1
            types[c].update(nd.event_types)

    edge_count: Counter = Counter()
    for g in graphs:
        seen = set()
        for e in g.edges:
            h, t = canon(e.head), canon(e.tail)
            if h != t and (h, t) not in seen:   # count each edge once per variant
                edge_count[(h, t)] += 1
                seen.add((h, t))

    edges = [Edge(head=h, tail=t, prob=round(c / n_variants, 3))
             for (h, t), c in edge_count.items() if c >= min_count]
    edges.sort(key=lambda e: (-e.prob, e.head, e.tail))

    used = {n for e in edges for n in (e.head, e.tail)}
    nodes = [Node(id=c,
                  kind=NodeKind(kind_votes[c].most_common(1)[0][0]) if kind_votes[c] else NodeKind.EVENT,
                  event_types=sorted(types[c]))
             for c in used]
    return CausalGraph(nodes=nodes, edges=edges)
