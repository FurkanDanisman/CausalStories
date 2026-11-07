#!/usr/bin/env python3
"""Re-asks the decision model grouping questions from a finished run, to tell a model error from a serving
fault. For one story, it rebuilds the groups as they stood when that story was grouped (Section 2, Step 1:
groups in creation order, each shown by its members so far, opposite members left out) and asks:
  1. the full request exactly as the pipeline sent it (all nodes x all groups), compared with the recorded answers;
  2. each question about the target node alone, three times;
  3. the full request with the questions in reverse order.
Usage: python benchmark/kev_test.py <run_dir> <story id> "<node>" [kev url]"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import prompts
from pipeline.kev_client import KevClient


def groups_before(run, story):
    """The groups (name, kind, members shown) that existed when <story> was grouped."""
    graphs = [json.loads(Path(f).read_text()) for f in sorted(run.glob("*__gemma-kev.chain.json")) if not f.name.startswith("._")]
    order = [g["id"] for g in graphs]
    pos = {j: i for i, j in enumerate(order)}
    nodes_of = {g["id"]: list(g["nodes"]) for g in graphs}
    abst = json.loads((run / "abstraction.json").read_text())
    gs = []
    for c, d in abst["concepts"].items():
        first = d["members"][0]
        if pos[first[0]] >= pos[story]:
            continue
        gs.append((pos[first[0]], nodes_of[first[0]].index(first[1]), d["kind"],
                   [m[1] for m in d["members"] if pos[m[0]] < pos[story] and m[2] > 0], first[1]))
    gs.sort()
    names, out = set(), []
    for _, _, kind, shown, label in gs:              # names as kev_merge gave them before naming
        out.append({"kind": kind, "shown": shown, "name": label if label not in names else None})
        names.add(label)
    graph = next(g for g in graphs if g["id"] == story)
    rec = [d for d in abst["decisions"] if d["narrative"] == story]
    return graph, out, rec


def ask(kev, text, qs):
    ans = kev.ask(text, qs)
    return {k: {r: round(p, 4) for r, p in ans[k]["probabilities"].items()} for k in qs}


def main():
    run, story, node = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    kev = KevClient(sys.argv[4] if len(sys.argv) > 4 else "http://127.0.0.1:8009")
    g, groups, rec = groups_before(run, story)
    kinds = g.get("kinds") or {}
    keys, qs = [], {}
    for v in g["nodes"]:
        for gi, G in enumerate(groups):
            if G["kind"] != kinds.get(v, "event"):
                continue
            qs[f"q{len(keys)}"] = prompts.kev_group_question(v, G["shown"], G["kind"])
            keys.append((v, gi))
    print(f"story {story}: {g['text']}\n{len(groups)} groups existed; {len(qs)} questions in the full request\n")
    recorded = {(d["node"], d["group"]): d["probs"] for d in rec}
    full = ask(kev, g["text"], qs)
    rev = ask(kev, g["text"], dict(reversed(list(qs.items()))))
    print(f"{'group (members shown)':<60} {'recorded same':>13} {'full now':>9} {'reversed':>9} {'alone x3':>22}")
    for i, (v, gi) in enumerate(keys):
        if v != node:
            continue
        G = groups[gi]; q = qs[f"q{i}"]
        alone = [ask(kev, g["text"], {"q": q})["q"]["same"] for _ in range(3)]
        r = recorded.get((v, G["name"]), {}).get("same")
        label = (G["name"] or "?") + " " + json.dumps(G["shown"][:3])[:40]
        print(f"{label:<60} {r if r is not None else '-':>13} {full[f'q{i}']['same']:>9} {rev[f'q{i}']['same']:>9} {str(alone):>22}")


if __name__ == "__main__":
    main()
