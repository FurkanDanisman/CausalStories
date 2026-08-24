"""Extraction stages: nodes once, edges K times, aggregate to arrow probabilities.

Design choice for self-consistency: we extract the node set ONCE, then run edge
extraction K times against that fixed node vocabulary. Because every run shares
the same node ids, we can aggregate edges by exact (head, tail) match -- no
cross-run node alignment needed. An arrow's probability is the fraction of runs
that drew it. This is a single-text preview of the cross-narrative edge-
probability aggregation described in methodology.tex.
"""

from __future__ import annotations

from collections import Counter

from .llm_client import LLMClient
from .prompts import extract_edges_prompt, extract_nodes_prompt
from .schema import CausalGraph, Edge, EdgeExtraction, Node, NodeExtraction


def extract_nodes(client: LLMClient, text: str) -> list[Node]:
    out = client.complete(
        task="extract_nodes", prompt=extract_nodes_prompt(text),
        schema=NodeExtraction, temperature=0.0,
    )
    # de-dup by id, keep first
    seen, nodes = set(), []
    for n in out.nodes:
        if n.id not in seen:
            seen.add(n.id)
            nodes.append(n)
    return nodes


def extract_edges_once(client: LLMClient, text: str, node_ids: list[str],
                       temperature: float) -> list[tuple[str, str]]:
    out = client.complete(
        task="extract_edges", prompt=extract_edges_prompt(text, node_ids),
        schema=EdgeExtraction, temperature=temperature,
    )
    valid = set(node_ids)
    return [(e.head, e.tail) for e in out.edges
            if e.head in valid and e.tail in valid and e.head != e.tail]


def extract_graph(client: LLMClient, text: str, k: int = 1) -> CausalGraph:
    """Full single-example extraction with K-sample self-consistency."""
    nodes = extract_nodes(client, text)
    node_ids = [n.id for n in nodes]

    temp = 0.0 if k == 1 else 0.7
    counter: Counter[tuple[str, str]] = Counter()
    for _ in range(k):
        counter.update(set(extract_edges_once(client, text, node_ids, temp)))

    edges = [Edge(head=h, tail=t, prob=c / k) for (h, t), c in counter.items()]
    edges.sort(key=lambda e: (-e.prob, e.head, e.tail))
    return CausalGraph(nodes=nodes, edges=edges)
