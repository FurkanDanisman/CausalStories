"""Chain-walk extraction: one narrative in, one connected causal graph out.

See method_Oct.tex, Section 1. Who does what:
  nodes      LLM (Gemma)  the K most important events, then the participants
  walk       Kev (choice) when a node X is reached, one request with one question per node Y
                          NOT on the current chain. Each node is asked once (the first time it
                          is reached); later chains reuse that answer. Chains start from random
                          unvisited nodes. Questions by pair type:
                            event -> event              enables / blocks / none
                            participant -> event        enables / blocks / none
                            event -> participant        not in this walk: a separate influence walk
                                                        asks every event about every participant,
                                                        positive / negative / none (kev_influence.md;
                                                        drawn like enables / blocks)
                            participant -> participant  not asked
                          PARTICIPANT_ARROWS=agent switches back to the yes/no "carries out" question.
  connect    algorithm    the graph must be connected; if not, the story is flagged
"""

from __future__ import annotations

import os
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
    asked: list[tuple] = field(default_factory=list)  # every pair asked: (head, tail, mean probs, per-repeat probs)

    def to_json(self) -> dict:
        return {"nodes": self.nodes,
                "edges": [{"head": h, "tail": t, "rel": r, "prob": round(p, 3)} for h, t, r, p in self.edges],
                "chains": self.chains, "kev_requests": self.kev_requests,
                "connected": self.connected, "kinds": self.kinds,
                "asked": [{"head": h, "tail": t, "probs": {k: round(v, 4) for k, v in pr.items()},
                           "reps": [{k: round(v, 4) for k, v in r.items()} for r in reps]}
                          for h, t, pr, reps in self.asked]}


# ------------------------------------------------------------ LLM (Gemma) step

def llm_nodes(client: LLMClient, text: str, max_nodes: int | None) -> tuple[dict[str, str], dict]:
    """(node id -> kind, Gemma's raw answers). Two calls: the K most important events, then
    the participants. The kind Gemma gives each item is kept: only items labelled "event"
    in the event call are events (and count toward K); items labelled "participant" in
    either call are participants."""
    ev = client.complete(task="extract_events", schema=NodeExtraction, temperature=0.0,
                         prompt=prompts.extract_events_prompt(text, max_nodes))
    pa = client.complete(task="extract_participants", schema=NodeExtraction, temperature=0.0,
                         prompt=prompts.extract_participants_prompt(text))
    raw = {"events_call": [{"id": n.id, "kind": n.kind.value} for n in ev.nodes],
           "participants_call": [{"id": n.id, "kind": n.kind.value} for n in pa.nodes]}
    events = list(dict.fromkeys(n.id for n in ev.nodes if n.kind.value == "event"))
    events = events[:max_nodes] if max_nodes else events
    kinds = {e: "event" for e in events}
    for n in list(ev.nodes) + list(pa.nodes):
        if n.kind.value == "participant":
            kinds.setdefault(n.id, "participant")
    return kinds, raw


# ------------------------------------------------------------------ Kev steps

REPEATS = int(os.environ.get("KEV_REPEATS", "1"))   # each question asked R times, options reordered (1 = once)


def _reordered(q: dict, k: int) -> dict:
    """Repeat k of question q: options rotated by k, and reversed on odd k."""
    items = list(q["criteria"].items())
    items = items[k % len(items):] + items[:k % len(items)]
    if k % 2:
        items = items[::-1]
    return {**q, "criteria": dict(items)}


def kev_arrows(kev: KevClient, text: str, ids: list[str], pairs: list[tuple[str, str]],
               kinds: dict[str, str] | None = None, into_participants: bool = False) -> list[tuple[str, str, str, dict]]:
    """One Kev request, one choice question per (head, tail) pair, each asked REPEATS times
    with the options reordered. Returns (head, tail, most likely option by mean probability,
    mean probabilities over enables / blocks / none, per-repeat probabilities).
    participant -> event: "does head carry out or initiate tail?" (yes = enables arrow);
    event -> participant and participant -> participant: not asked;
    event -> event: enables / blocks / none."""
    kinds = kinds or {}
    paper = os.environ.get("PARTICIPANT_ARROWS", "paper") == "paper"   # Regan et al. participants (default)
    asked, qs = [], {}
    for h, t in pairs:
        kh, kt = kinds.get(h, "event"), kinds.get(t, "event")
        if kh == kt == "participant" or (kt == "participant" and not (paper and into_participants)):
            continue                                   # event -> participant only in the influence walk
        agent = kh == "participant" and not paper
        influence = kt == "participant" and paper              # event -> participant: positive / negative
        q = (prompts.kev_agent_question(h, t, ids) if agent else
             prompts.kev_influence_question(h, t, ids) if influence else
             prompts.kev_arrow_question(h, t, ids))
        qs[f"q{len(asked)}"] = q
        asked.append((h, t, "agent" if agent else "influence" if influence else "arrow"))
    if not qs:
        return []
    ans = kev.ask(text, {f"{qid}_{k}": _reordered(q, k) for qid, q in qs.items() for k in range(REPEATS)})
    out = []
    for i, (h, t, how) in enumerate(asked):
        reps = []
        for k in range(REPEATS):
            pr = ans[f"q{i}_{k}"]["probabilities"]
            if how == "agent":                       # yes -> enables arrow
                pr = {"enables": pr["yes"], "blocks": 0.0, "none": pr["no"]}
            elif how == "influence":                 # positive -> enables (green), negative -> blocks (red)
                pr = {"enables": pr["positive"], "blocks": pr["negative"], "none": pr["none"]}
            reps.append(dict(pr))
        mean = {r: sum(x[r] for x in reps) / REPEATS for r in reps[0]}
        out.append((h, t, max(mean, key=mean.get), mean, reps))   # the walk uses the mean
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
            children[x] = [(t, c, p[c]) for _, t, c, p, _ in res if c != "none"]
            asked.extend((h, t, p, reps) for h, t, _, p, reps in res)
            log(f"  ask #{len(children)} {x!r}")
            for _, t, c, p, _ in res:
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

def influence_walk(kev: KevClient, text: str, ids: list[str], kinds: dict[str, str] | None = None,
                   log=print):
    """Walk 2, separate from the chain walk: every event is asked once about every participant,
    positive / negative / no influence (kev_influence.md). Nothing leaves a participant here,
    so each path is one step. Returns (edges, asked)."""
    kinds = kinds or {}
    events = [n for n in ids if kinds.get(n, "event") == "event"]
    parts = [n for n in ids if kinds.get(n, "event") == "participant"]
    pairs = [(e, p) for e in events for p in parts]
    if not pairs:
        return [], []
    res = kev_arrows(kev, text, ids, pairs, kinds, into_participants=True)
    for h, t, c, p, _ in res:
        probs = " ".join(f"{k}={v:.2f}" for k, v in p.items())
        log(f"  {h!r} -> {t!r}: {({'enables': 'POSITIVE', 'blocks': 'NEGATIVE'}).get(c, 'none')}   ({probs})")
    edges = [(h, t, c, p[c]) for h, t, c, p, _ in res if c != "none"]
    return edges, [(h, t, p, reps) for h, t, _, p, reps in res]


def run_chain(kev: KevClient, text: str, ids: list[str], seed: int = 0, log=print,
              kinds: dict[str, str] | None = None) -> ChainGraph:
    start = kev.requests
    log("WALK 1 (chain walk: event -> event, participant -> event)")
    edges, chains, asked = walk(kev, text, ids, seed=seed, log=log, kinds=kinds)
    if os.environ.get("PARTICIPANT_ARROWS", "paper") == "paper":
        log("WALK 2 (influence: event -> participant)")
        e2, a2 = influence_walk(kev, text, ids, kinds, log=log)
        edges, asked = list(edges) + e2, list(asked) + a2
    log("CONNECT (both walks)")
    parts = components(ids, edges)
    ok = len(parts) == 1
    log(f"  connected: {ok}" + ("" if ok else f"  FLAGGED: {len(parts)} parts {parts}"))
    return ChainGraph(nodes=list(ids), edges=list(edges), chains=chains,
                      kev_requests=kev.requests - start, connected=ok,
                      kinds={n: (kinds or {}).get(n, "event") for n in ids}, asked=asked)
