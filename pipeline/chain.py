"""Chain-walk extraction: one narrative in, one connected causal graph out.

See method_Oct.tex, Section 1. Who does what:
  nodes      LLM (Gemma)  the K most important events, then the participants
  walk       Kev (choice) when a node X is reached, one request with one question per node Y
                          NOT on the current chain. Each node is asked once (the first time it
                          is reached); later chains reuse that answer. Chains start from random
                          unvisited nodes. Questions by pair type:
                            event -> event              enables / blocks / none
                            participant -> event        does X carry out or initiate Y? yes / no
                            event -> participant        not asked
                            participant -> participant  not asked
  connect    algorithm    the graph must be connected; if not, the story is flagged
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

from . import prompts
from .kev_client import KevClient
from .llm_client import LLMClient
from .schema import NodeExtraction

Edge = tuple[str, str, str, float]              # (head, tail, "enables"|"blocks", Kev probability)


@dataclass
class ChainGraph:
    nodes: list[str]
    edges: list[Edge]
    chains: list[list[str]] = field(default_factory=list)
    kev_requests: int = 0
    connected: bool = True
    kinds: dict[str, str] = field(default_factory=dict)   # node id -> "event" | "participant"
    asked: list[tuple[str, str, dict]] = field(default_factory=list)  # every pair asked + probabilities

    def to_json(self) -> dict:
        return {"nodes": self.nodes,
                "edges": [{"head": h, "tail": t, "rel": r, "prob": round(p, 3)} for h, t, r, p in self.edges],
                "chains": self.chains, "kev_requests": self.kev_requests,
                "connected": self.connected, "kinds": self.kinds,
                "asked": [{"head": h, "tail": t, "probs": {k: round(v, 4) for k, v in pr.items()}}
                          for h, t, pr in self.asked]}


# ------------------------------------------------------------ LLM (Gemma) step

def llm_nodes(client: LLMClient, text: str, max_nodes: int | None) -> dict[str, str]:
    """Node id -> kind. Two calls: the K most important events, then the participants."""
    ev = client.complete(task="extract_events", schema=NodeExtraction, temperature=0.0,
                         prompt=prompts.extract_events_prompt(text, max_nodes))
    events = list(dict.fromkeys(n.id for n in ev.nodes))
    events = events[:max_nodes] if max_nodes else events
    pa = client.complete(task="extract_participants", schema=NodeExtraction, temperature=0.0,
                         prompt=prompts.extract_participants_prompt(text))
    kinds = {e: "event" for e in events}
    for n in pa.nodes:
        kinds.setdefault(n.id, "participant")
    return kinds


# ------------------------------------------------------------------ Kev steps

def kev_arrows(kev: KevClient, text: str, ids: list[str], pairs: list[tuple[str, str]],
               kinds: dict[str, str] | None = None) -> list[tuple[str, str, str, dict]]:
    """One Kev request, one choice question per (head, tail) pair. Returns
    (head, tail, most likely option, probabilities over enables / blocks / none).
    participant -> event: "does head carry out or initiate tail?" (yes = enables arrow);
    event -> participant and participant -> participant: not asked;
    event -> event: enables / blocks / none."""
    kinds = kinds or {}
    asked, qs = [], {}
    for h, t in pairs:
        kh, kt = kinds.get(h, "event"), kinds.get(t, "event")
        if kt == "participant":
            continue
        q = (prompts.kev_agent_question(h, t, ids) if kh == "participant"
             else prompts.kev_arrow_question(h, t, ids))
        qs[f"q{len(asked)}"] = q
        asked.append((h, t, kh == "participant"))
    if not qs:
        return []
    ans = kev.ask(text, qs)
    out = []
    for i, (h, t, agent) in enumerate(asked):
        a = ans[f"q{i}"]
        if agent:                                    # yes -> enables arrow, no -> none
            pr = a["probabilities"]
            out.append((h, t, "enables" if a["choice"] == "yes" else "none",
                        {"enables": pr["yes"], "blocks": 0.0, "none": pr["no"]}))
        else:
            out.append((h, t, a["choice"], a["probabilities"]))
    return out


def walk(kev: KevClient, text: str, ids: list[str], seed: int = 0, log=print,
         kinds: dict[str, str] | None = None):
    children: dict[str, list[tuple[str, str, float]]] = {}
    chains: list[list[str]] = []
    asked: list[tuple[str, str, dict]] = []

    def ask(path: list[str]) -> list[tuple[str, str, float]]:
        x = path[-1]
        if x not in children:                     # candidates: every node not on the current chain
            res = kev_arrows(kev, text, ids, [(x, y) for y in ids if y not in path], kinds)
            children[x] = [(t, c, p[c]) for _, t, c, p in res if c != "none"]
            asked.extend((h, t, p) for h, t, _, p in res)
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
    return edges, chains, asked


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


# ------------------------------------------------------------------ pipeline

def run_chain(kev: KevClient, text: str, ids: list[str], seed: int = 0, log=print,
              kinds: dict[str, str] | None = None) -> ChainGraph:
    start = kev.requests
    log("WALK (Kev choice)")
    edges, chains, asked = walk(kev, text, ids, seed=seed, log=log, kinds=kinds)
    log("CONNECT")
    parts = components(ids, edges)
    ok = len(parts) == 1
    log(f"  connected: {ok}" + ("" if ok else f"  FLAGGED: {len(parts)} parts {parts}"))
    return ChainGraph(nodes=list(ids), edges=list(edges), chains=chains,
                      kev_requests=kev.requests - start, connected=ok,
                      kinds={n: (kinds or {}).get(n, "event") for n in ids}, asked=asked)
