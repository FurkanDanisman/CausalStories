"""Section 2, Steps 1-2 (see method_Oct.tex): group nodes across narratives with Kev,
then align their direction. Replaces the Gemma merge (abstraction.py).

  Step 1  narratives are taken one at a time. Each node of the first narrative starts a
          group. For each later narrative, every node is compared with every existing
          group of the same kind in one Kev request: events get "same / opposite /
          different", participants "same / different". Assignment takes the most
          probable (node, group) pair first; a group takes at most one node per
          narrative and a node joins at most one group. A node with no "same" or
          "opposite" answer starts its own group. No node is dropped.
  Step 2  a node that joined as "opposite" gets sign -1: its value is flipped (Step 3)
          and its arrows flip enables <-> blocks (Step 5).
Naming (LLM, after Kev has grouped): each group gets a high-level concept name, events
by meaning and participants by role (name_groups.md); until then a group is labelled by
its first member.
"""

from __future__ import annotations

from . import prompts
from .kev_client import KevClient
from .llm_client import LLMClient
from .schema import NodeClustering


def kev_merge(kev: KevClient, graphs: list[dict], log=print) -> dict:
    groups: list[dict] = []            # {"kind", "name", "members": [[j, v, sign]]}
    decisions: list[dict] = []
    for g in graphs:
        j, kinds = g["id"], g.get("kinds") or {}
        nodes = list(g["nodes"])
        qs, keys = {}, []
        for v in nodes:
            k = kinds.get(v, "event")
            for gi, G in enumerate(groups):
                if G["kind"] != k:
                    continue
                shown = [m[1] for m in G["members"] if m[2] > 0]
                qs[f"q{len(keys)}"] = prompts.kev_group_question(v, shown, k)
                keys.append((v, gi))
        cands = []
        if qs:
            ans = kev.ask(g["text"], qs)
            for i, (v, gi) in enumerate(keys):
                pr = ans[f"q{i}"]["probabilities"]
                choice = max(pr, key=pr.get)
                decisions.append({"narrative": j, "node": v, "group": groups[gi]["name"],
                                  "probs": {r: round(x, 4) for r, x in pr.items()}, "choice": choice})
                if choice in ("same", "opposite"):
                    cands.append((pr[choice], v, gi, 1 if choice == "same" else -1))
        placed, taken = {}, set()
        for p, v, gi, sign in sorted(cands, key=lambda c: -c[0]):
            if v in placed or gi in taken:
                continue
            placed[v] = (gi, sign)
            taken.add(gi)
        log(f"[{j}]")
        for v in nodes:
            if v in placed:
                gi, sign = placed[v]
                groups[gi]["members"].append([j, v, sign])
                log(f"  {v!r} -> {groups[gi]['name']!r}" + ("  (opposite)" if sign < 0 else ""))
            else:
                name = v if all(G["name"] != v for G in groups) else f"{v} ({j})"
                groups.append({"kind": kinds.get(v, "event"), "name": name, "members": [[j, v, 1]]})
                log(f"  {v!r} -> new group")

    concepts = {G["name"]: {"kind": G["kind"], "members": G["members"]} for G in groups}
    narratives = [g["id"] for g in graphs]
    table = {j: {c: (1 if any(m[0] == j for m in d["members"]) else None) for c, d in concepts.items()}
             for j in narratives}
    of = {(m[0], m[1]): (c, m[2]) for c, d in concepts.items() for m in d["members"]}
    flip = {"enables": "blocks", "blocks": "enables"}
    concept_edges = {}
    for g in graphs:
        j, es = g["id"], []
        for e in g["edges"]:
            (ch, sh), (ct, st) = of[(j, e["head"])], of[(j, e["tail"])]
            rel = e["rel"] if sh * st > 0 else flip.get(e["rel"], e["rel"])
            es.append({"head": ch, "tail": ct, "rel": rel, "prob": e["prob"]})
        concept_edges[j] = es
    return {"concepts": concepts, "dropped": [], "table": table,
            "concept_edges": concept_edges, "decisions": decisions}


def name_groups(client: LLMClient, abst: dict, log=print) -> dict:
    """High-level names for Kev's groups, one LLM call per kind. Renames the concepts in
    abst (concepts, table, concept_edges); the grouping itself is unchanged."""
    rename: dict[str, str] = {}
    for kind in ("event", "participant"):
        names = [c for c, d in abst["concepts"].items() if d["kind"] == kind]
        if not names:
            continue
        lines = []
        for i, c in enumerate(names):
            ms = "; ".join(f'"{m[1]}"' + (" (opposite)" if m[2] < 0 else "") for m in abst["concepts"][c]["members"])
            lines.append(f"  g{i}: {ms}")
        try:
            out = client.complete(task="name_groups", schema=NodeClustering, temperature=0.0,
                                  prompt=prompts.name_groups_prompt(kind, "\n".join(lines)))
            mapping = {k.strip(): v.strip() for k, v in out.mapping.items()}
        except Exception as e:
            log(f"  naming failed ({type(e).__name__}: {e}); groups keep their labels")
            mapping = {}
        used = set(rename.values())
        for i, c in enumerate(names):
            new = mapping.get(f"g{i}") or c
            base, n = new, 2
            while new in used:                      # different groups keep different names
                new, n = f"{base} ({n})", n + 1
            used.add(new)
            rename[c] = new
            log(f"  [{kind}] {new}  <-  " + "; ".join(m[1] for m in abst["concepts"][c]["members"]))
    abst["concepts"] = {rename[c]: d for c, d in abst["concepts"].items()}
    abst["table"] = {j: {rename[c]: v for c, v in row.items()} for j, row in abst["table"].items()}
    abst["concept_edges"] = {j: [{**e, "head": rename[e["head"]], "tail": rename[e["tail"]]} for e in es]
                             for j, es in abst["concept_edges"].items()}
    abst["labels"] = {new: old for old, new in rename.items()}
    return abst
