"""Chain-walk extraction: one narrative in, one connected causal graph out.

See chain_extraction_proposal.md. Who does what:
  nodes      LLM (Gemma)  the K most important events/participants   (or given by hand)
  opposites  LLM (Gemma)  the opposite name of each node              (or given by hand)
  walk       Kev (choice) when a node X is reached, one request with one question per node Y
                          NOT on the current chain: direct relation X -> Y is enables / blocks /
                          none. Each node is asked once (the first time it is reached); later
                          chains reuse that answer. Chains start from random unvisited nodes.
  connect    algorithm    the graph must be connected; if not, the story is flagged
  polarity   algorithm    rename nodes to their opposite so every arrow is "enables"
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

from . import prompts
from .kev_client import KevClient
from .llm_client import LLMClient
from .schema import NegatedNames, NodeExtraction

Edge = tuple[str, str, str, float]              # (head, tail, "enables"|"blocks", Kev probability)


@dataclass
class ChainGraph:
    nodes: list[str]
    edges: list[Edge]                           # after polarity
    raw_edges: list[Edge]                       # Kev's arrows before polarity
    chains: list[list[str]] = field(default_factory=list)
    kev_requests: int = 0
    connected: bool = True
    balanced: bool = True                       # every arrow could be made "enables"
    renamed: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict:
        e = lambda es: [{"head": h, "tail": t, "rel": r, "prob": round(p, 3)} for h, t, r, p in es]
        return {"nodes": self.nodes,
                "edges": e(self.edges), "raw_edges": e(self.raw_edges), "chains": self.chains,
                "kev_requests": self.kev_requests, "connected": self.connected,
                "balanced": self.balanced, "renamed": self.renamed}


# ------------------------------------------------------------ LLM (Gemma) steps

def llm_nodes(client: LLMClient, text: str, max_nodes: int | None) -> list[str]:
    out = client.complete(task="extract_nodes", schema=NodeExtraction, temperature=0.0,
                          prompt=prompts.extract_nodes_prompt(text, max_nodes))
    ids = list(dict.fromkeys(n.id for n in out.nodes))
    return ids[:max_nodes] if max_nodes else ids


def llm_opposites(client: LLMClient, text: str, ids: list[str]) -> dict[str, str]:
    out = client.complete(task="negate_nodes", schema=NegatedNames, temperature=0.0,
                          prompt=prompts.negate_nodes_prompt(text, ids))
    return {x.id: x.opposite for x in out.names if x.id in ids}


# ------------------------------------------------------------------ Kev steps

def kev_arrows(kev: KevClient, text: str, ids: list[str],
               pairs: list[tuple[str, str]]) -> list[tuple[str, str, str, dict]]:
    """One Kev request, one choice question per (head, tail) pair. Returns
    (head, tail, most likely option, all option probabilities)."""
    qs = {f"q{i}": prompts.kev_arrow_question(h, t, ids) for i, (h, t) in enumerate(pairs)}
    ans = kev.ask(text, qs)
    out = []
    for i, (h, t) in enumerate(pairs):
        a = ans[f"q{i}"]
        out.append((h, t, a["choice"], a["probabilities"]))
    return out


def walk(kev: KevClient, text: str, ids: list[str], seed: int = 0, log=print):
    children: dict[str, list[tuple[str, str, float]]] = {}
    chains: list[list[str]] = []

    def ask(path: list[str]) -> list[tuple[str, str, float]]:
        x = path[-1]
        if x not in children:                     # candidates: every node not on the current chain
            res = kev_arrows(kev, text, ids, [(x, y) for y in ids if y not in path])
            children[x] = [(t, c, p[c]) for _, t, c, p in res if c != "none"]
            log(f"  ask #{len(children)} {x!r}")
            for _, t, c, p in res:
                probs = " ".join(f"{k}={v:.2f}" for k, v in p.items())
                log(f"      -> {t!r}: {c.upper() if c != 'none' else 'none'}   ({probs})")
        return children[x]

    def expand(path: list[str]) -> None:
        kids = [t for t, _, _ in ask(path) if t not in path]     # a reused answer may name a node on this chain
        if not kids:
            chains.append(path)
            return
        for t in kids:
            expand(path + [t])

    rng = random.Random(seed)
    unvisited = list(ids)
    while unvisited:
        start = rng.choice(unvisited)
        log(f"  start at {start!r}")
        expand([start])
        on_chain = {n for c in chains for n in c}
        unvisited = [i for i in unvisited if i not in on_chain]
    edges = [(h, t, r, p) for h, kids in children.items() for t, r, p in kids]
    return edges, chains


def components(ids: list[str], edges: list[Edge]) -> list[list[str]]:
    adj = {i: set() for i in ids}
    for h, t, *_ in edges:
        adj[h].add(t)
        adj[t].add(h)
    comps, seen = [], set()
    for i in ids:
        if i in seen:
            continue
        comp, q = [], deque([i])
        seen.add(i)
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u] - seen:
                seen.add(v)
                q.append(v)
        comps.append(comp)
    return comps


# ------------------------------------------------------------------ polarity

def polarity_signs(ids: list[str], edges: list[Edge]) -> tuple[dict[str, int], bool]:
    """Sign s(n) in {+1, -1} per node (-1 = rename to its opposite) such that every
    arrow becomes "enables": s(head) * s(tail) * sign(rel) = +1. Possible iff every
    undirected loop has an even number of "blocks" arrows (Harary's balance theorem).
    Per connected part, flip the smaller side."""
    adj: dict[str, list[tuple[str, int]]] = {i: [] for i in ids}
    for h, t, r, _ in edges:
        s = -1 if r == "blocks" else 1
        adj[h].append((t, s))
        adj[t].append((h, s))
    sign, balanced = {}, True
    for comp in components(ids, edges):
        sign[comp[0]] = 1
        q = deque([comp[0]])
        while q:
            u = q.popleft()
            for v, s in adj[u]:
                want = sign[u] * s
                if v not in sign:
                    sign[v] = want
                    q.append(v)
                elif sign[v] != want:
                    balanced = False
        if sum(sign[n] == -1 for n in comp) * 2 > len(comp):
            for n in comp:
                sign[n] = -sign[n]
    return sign, balanced


def apply_polarity(g: ChainGraph, opposites: dict[str, str], log=print) -> ChainGraph:
    sign, g.balanced = polarity_signs(g.nodes, g.edges)
    flip = [n for n in g.nodes if sign[n] == -1]
    name = {n: (opposites.get(n) or f"not: {n}") if n in flip else n for n in g.nodes}
    for n in flip:
        log(f"  rename {n!r} -> {name[n]!r}")
    g.renamed = {n: name[n] for n in flip}
    g.edges = [(name[h], name[t],
                "enables" if sign[h] * sign[t] * (-1 if r == "blocks" else 1) == 1 else "blocks", p)
               for h, t, r, p in g.edges]
    g.chains = [[name[n] for n in c] for c in g.chains]
    g.nodes = [name[n] for n in g.nodes]
    if not g.balanced:
        log("  a loop has an odd number of blocks arrows: at least one arrow stays 'blocks'")
    return g


# ------------------------------------------------------------------ pipeline

def run_chain(kev: KevClient, text: str, ids: list[str], opposites: dict[str, str],
              seed: int = 0, log=print) -> ChainGraph:
    start = kev.requests
    log("WALK (Kev choice)")
    edges, chains = walk(kev, text, ids, seed=seed, log=log)
    log("CONNECT")
    parts = components(ids, edges)
    ok = len(parts) == 1
    log(f"  connected: {ok}" + ("" if ok else f"  FLAGGED: {len(parts)} parts {parts}"))
    g = ChainGraph(nodes=list(ids), edges=list(edges), raw_edges=list(edges), chains=chains,
                   kev_requests=kev.requests - start, connected=ok)
    log("POLARITY")
    return apply_polarity(g, opposites, log=log)
