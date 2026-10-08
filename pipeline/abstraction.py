"""Section 2, Steps 2-3: low-level vocabulary -> shared concepts.

Merging is a constructive abstraction (Beckers & Halpern 2019):
  Step 2  the low-level vocabulary L is the union of all narrative-level nodes, each
          tagged with its narrative; events and participants are kept apart.
  Step 3  an LLM partitions L into concepts (plus DROP), in rounds: the narratives are
          split into M subgroups, each subgroup is merged, and the results are merged
          M at a time until one remains. The LLM sees each item's causes and effects.
          Rules (merge_concepts.md): R1 partition, R2 no invented concepts, R3 same
          causal role, R4 dependent concepts merged, R5 cause and effect never merged,
          R6 events and participants never merged with each other.
  Code enforces R1 (unassigned items keep their own concept), R2 (concepts come only
  from items), R5 (a concept with an arrow inside it is split) and R6 (one kind per
  call). A concept's value in narrative x_j is 1 if any of x_j's nodes are in it
  (OR), else missing. The check flags concepts whose members have no concept-level
  cause or effect in common.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from . import prompts
from .llm_client import LLMClient
from .schema import NodeClustering

Key = tuple[str, str]                 # (narrative id, node id): one low-level node


@dataclass
class Item:
    name: str
    kind: str
    members: set[Key] = field(default_factory=set)


def vocabulary(graphs: list[dict]) -> tuple[dict[Key, str], dict[str, list[tuple[str, str, str, float]]]]:
    """Step 2. Returns kind of every low-level node and each narrative's edges."""
    kind: dict[Key, str] = {}
    edges: dict[str, list] = {}
    for g in graphs:
        j = g["id"]
        kinds = g.get("kinds") or {}
        for v in g["nodes"]:
            kind[(j, v)] = kinds.get(v, "event")
        edges[j] = [(e["head"], e["tail"], e.get("rel", "enables"), e.get("prob", 1.0)) for e in g["edges"]]
    return kind, edges


def _neighbours(members: set[Key], edges: dict[str, list]) -> tuple[list[str], list[str]]:
    causes, effects = [], []
    for j, v in members:
        for h, t, _, _ in edges[j]:
            if t == v and h not in causes:
                causes.append(h)
            if h == v and t not in effects:
                effects.append(t)
    return causes[:6], effects[:6]


def _merge_once(client: LLMClient, items: list[Item], kind: str, edges: dict[str, list],
                log=print) -> tuple[list[Item], set[Key]]:
    """One LLM call: merge a list of items of one kind into concepts."""
    lines = []
    for i, it in enumerate(items):
        narr = ", ".join(sorted({j for j, _ in it.members}))
        causes, effects = _neighbours(it.members, edges)
        lines.append(f'  [n{i}] "{it.name}" (narrative {narr}) | causes: '
                     f'{", ".join(effects) or "-"} | caused by: {", ".join(causes) or "-"}')
    try:
        out = client.complete(task="merge_concepts", schema=NodeClustering, temperature=0.0,
                              prompt=prompts.merge_concepts_prompt(kind, "\n".join(lines)))
        mapping = {k.strip(): v.strip() for k, v in out.mapping.items()}
    except Exception as e:                                     # keep every item as its own concept
        log(f"    merge failed ({type(e).__name__}: {e}); items kept as they are")
        mapping = {}
    merged: dict[str, Item] = {}
    dropped: set[Key] = set()
    for i, it in enumerate(items):
        name = mapping.get(f"n{i}") or it.name                 # R1: unassigned keeps its own concept
        if name.upper() == "DROP":
            dropped |= it.members
            log(f"    DROP  {it.name!r}")
            continue
        merged.setdefault(name, Item(name=name, kind=kind)).members |= it.members
    return list(merged.values()), dropped


def merge_kind(client: LLMClient, kind: str, low: list[Key], narratives: list[str],
               edges: dict[str, list], M: int, log=print) -> tuple[list[Item], set[Key]]:
    """Step 3 for one kind (R6): M subgroups of narratives, then merge M results at a time."""
    if not low:
        return [], set()
    groups = [narratives[g::M] for g in range(M)]
    batches = [[Item(name=v, kind=kind, members={(j, v)}) for j, v in low if j in grp] for grp in groups]
    batches = [b for b in batches if b]
    dropped: set[Key] = set()
    rnd = 1
    while True:
        log(f"  {kind}s, round {rnd}: {len(batches)} subgroup(s)")
        results = []
        for b in batches:
            items, d = _merge_once(client, b, kind, edges, log)
            results.append(items)
            dropped |= d
        if len(results) == 1:
            return results[0], dropped
        step = max(M, 2)
        batches = [[it for r in results[k:k + step] for it in r] for k in range(0, len(results), step)]
        rnd += 1


def abstract(client: LLMClient, graphs: list[dict], M: int = 1, log=print) -> dict:
    kind, edges = vocabulary(graphs)
    narratives = [g["id"] for g in graphs]
    log(f"STEP 2  low-level vocabulary: {len(kind)} nodes from {len(narratives)} narratives "
        f"({sum(k == 'event' for k in kind.values())} events, "
        f"{sum(k == 'participant' for k in kind.values())} participants)")

    log(f"STEP 3  merge into concepts (M = {M})")
    concept_of: dict[Key, str | None] = {}
    concept_kind: dict[str, str] = {}
    for k in ("event", "participant"):
        low = [key for key, kk in kind.items() if kk == k]
        items, dropped = merge_kind(client, k, low, narratives, edges, M, log)
        for it in items:
            name = it.name if it.name not in concept_kind else f"{it.name} ({k})"
            concept_kind[name] = k
            for key in it.members:
                concept_of[key] = name
        for key in dropped:
            concept_of[key] = None

    # R5: an arrow inside a concept means cause and effect were merged -> split the effect off
    flags: list[str] = []
    changed = True
    while changed:
        changed = False
        for j, es in edges.items():
            for h, t, _, _ in es:
                ch, ct = concept_of.get((j, h)), concept_of.get((j, t))
                if ch is not None and ch == ct:
                    new = t if t not in concept_kind else f"{t} ({j})"
                    concept_kind[new] = kind[(j, t)]
                    concept_of[(j, t)] = new
                    flags.append(f"R5 split: {h!r} -> {t!r} in {j} were both in {ch!r}; "
                                 f"{t!r} is now its own concept")
                    changed = True

    members: dict[str, list[Key]] = defaultdict(list)
    for key, c in concept_of.items():
        if c is not None:
            members[c].append(key)

    # values: 1 if any of x_j's nodes is in the concept (OR), else missing
    table = {j: {c: (1 if any(m[0] == j for m in ms) else None) for c, ms in members.items()}
             for j in narratives}

    # concept-level arrows per narrative
    concept_edges: dict[str, list[dict]] = {}
    for j, es in edges.items():
        best: dict[tuple[str, str], tuple[str, float]] = {}
        for h, t, r, p in es:
            ch, ct = concept_of.get((j, h)), concept_of.get((j, t))
            if ch is None or ct is None or ch == ct:
                continue
            if (ch, ct) not in best or p > best[(ch, ct)][1]:
                best[(ch, ct)] = (r, p)
        concept_edges[j] = [{"head": a, "tail": b, "rel": r, "prob": p} for (a, b), (r, p) in best.items()]

    # check: members of a concept should share some concept-level cause or effect
    def neigh(key: Key) -> set[tuple[str, str]]:
        j, v = key
        out = set()
        for h, t, _, _ in edges[j]:
            if h == v and concept_of.get((j, t)):
                out.add(("effect", concept_of[(j, t)]))
            if t == v and concept_of.get((j, h)):
                out.add(("cause", concept_of[(j, h)]))
        return out
    for c, ms in members.items():
        for a in range(len(ms)):
            for b in range(a + 1, len(ms)):
                na, nb = neigh(ms[a]), neigh(ms[b])
                if na and nb and not (na & nb):
                    flags.append(f"check: {c!r} members {ms[a]} and {ms[b]} share no cause or effect")

    return {
        "concepts": {c: {"kind": concept_kind[c], "members": [f"{j}: {v}" for j, v in ms]}
                     for c, ms in members.items()},
        "dropped": [f"{j}: {v}" for (j, v), c in concept_of.items() if c is None],
        "table": table,
        "concept_edges": concept_edges,
        "flags": flags,
    }
