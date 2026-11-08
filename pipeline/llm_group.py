"""Section 2, Step 1 with an LLM instead of the decision model (method_Oct.tex): the same rules, applied by a
generator LLM that sees many nodes at once.

  1. The stories are split into batches of M (default 10, in story order). For each batch and each kind
     (events, participants), one LLM call groups the batch's nodes.
  2. The batch groupings are merged pairwise in a tournament (20 -> 10 -> 5 -> 3 -> 2 -> 1 for 200 stories).
     A merge shows each group by up to 5 example members and asks, for every group of the second list, whether
     it is the same as a group of the first list, its opposite, or neither.
  3. The rules of Step 1 are checked after every call: every node is in exactly one group; a group takes at
     most one node per story (a second node of the same story starts its own group); opposite members are
     marked, as with the decision model. A node (or, in a merge, a group) the answer leaves out is asked about
     again in a follow-up call, up to two rounds; only a node still left out then gets its own group, and is
     counted. Every prompt and raw answer is saved ($LLM_GROUP_LOG).
The result has the same format as merge_kev.kev_merge (abstraction.json); naming (Prompt 4c) follows as before.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter

from .merge_kev import abstraction

RULES = """Group events only if they are the same kind of event at a high level and work the same way: they lead
to the same kinds of outcomes (Beckers and Halpern, "Abstracting Causal Models"). Judge what the words mean, not
whether the event happened in its story: an event that is mentioned but prevented is still the same event. An
event and its opposite describe one thing with opposite values: one happening means the other did not (e.g.
"paying rent" and "falling behind on rent"; "keeping the flat" and "eviction"); put them in one group and mark the
opposite ones. An event that prevents, stops, causes or follows another is not its opposite: it is a different
event (e.g. "a relative paying the rent" prevents "eviction" but is a different event)."""
RULES_P = """Group participants only if they play the same role in the stories (e.g. "my boss" and "the employer";
"my sister" and "a family member"). The person the events happen to ("I", "my brother", "she") is one role."""

BATCH = """Below are the {what} of {n} different people's stories, listed by story. Put them into groups:
{rules}
A group may contain at most one {item} from each story, and every {item} must be in exactly one group (a group may
have a single member).

{listing}

Answer only with JSON: {{"groups": [{{"members": [{{"id": "<id>", "opposite": false}}, ...]}}, ...]}}
Mark "opposite": true for a member that is the opposite of the group's other members (events only)."""

MERGE = """Two lists of groups of {what} from people's stories were made separately. Each group is shown by example
members. {rules}

List A:
{a}

List B:
{b}

For every group of list B, say whether it is the same as a group of list A, the opposite of one, or neither.
Answer only with JSON: {{"matches": [{{"b": "<B id>", "a": "<A id or null>", "relation": "same" or "opposite" or "none"}}, ...]}}"""


FOLLOW = """Groups of {what} from people's stories, each shown by example members:
{groups}

These {what} are not in a group yet:
{items}

{rules}
For each of them, say which group it belongs to (the same {item}, or for events its opposite), or "new" if it fits
none. Answer only with JSON: {{"placements": [{{"id": "<id>", "group": "<group id or new>", "opposite": false}}, ...]}}"""

FOLLOW_MERGE = """Two lists of groups of {what} from people's stories were made separately. Each group is shown by
example members. {rules}

List A:
{a}

These groups of list B still need an answer:
{b}

For each of them, say whether it is the same as a group of list A, the opposite of one, or neither.
Answer only with JSON: {{"matches": [{{"b": "<B id>", "a": "<A id or null>", "relation": "same" or "opposite" or "none"}}, ...]}}"""

STATS = {"calls": 0, "followups": 0, "forced_own_group": 0}


def chat_json(prompt: str, max_tokens: int = 8192) -> dict:
    """One call to the OpenAI-compatible LLM server ($LLM_URL, model $LLM_MODEL); the last JSON object of the answer."""
    url = os.environ.get("LLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
    body = {"model": os.environ.get("LLM_MODEL", "llm"), "temperature": 0.0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}], "chat_template_kwargs": {"enable_thinking": False}}
    data = json.dumps(body).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=data, headers={"content-type": "application/json",
                                                                  "authorization": "Bearer EMPTY"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                resp = json.loads(r.read())
            u = resp.get("usage") or {}
            from .kev_client import log_inference
            log_inference("generator", u.get("prompt_tokens"), u.get("completion_tokens"), chars=len(prompt))
            text = resp["choices"][0]["message"]["content"] or ""
            STATS["calls"] += 1
            if os.environ.get("LLM_GROUP_LOG"):                 # every prompt and raw answer, to trace any group
                with open(os.environ["LLM_GROUP_LOG"], "a") as f:
                    f.write(json.dumps({"prompt": prompt, "answer": text}) + "\n")
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
            m = re.findall(r"\{.*\}", text, flags=re.S)
            return json.loads(m[-1]) if m else {}
        except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, json.JSONDecodeError) as e:
            print(f"[retry] LLM grouping call failed ({type(e).__name__}), attempt {attempt + 1}/4", flush=True)
            time.sleep(2 ** attempt)
    return {}


def _fix(groups: list[list], allowed: set) -> list[list]:
    """Apply Step 1's rules to an answer: members are [story, node, sign]; drop unknown or repeated nodes; a second
    node of the same story leaves the group and starts its own; every allowed node not placed gets its own group;
    a group whose members are all opposites is turned round."""
    seen, out = set(), []
    for g in groups:
        keep, stories = [], set()
        for m in g:
            key = (m[0], m[1])
            if key not in allowed or key in seen:
                continue
            seen.add(key)
            if m[0] in stories:
                out.append([[m[0], m[1], 1]])                  # one node per story per group
            else:
                stories.add(m[0]); keep.append(m)
        if keep:
            if all(m[2] < 0 for m in keep):
                keep = [[m[0], m[1], 1] for m in keep]
            out.append(keep)
    left = sorted(allowed - seen)
    STATS["forced_own_group"] += len(left)                      # still left out after the follow-up rounds
    out += [[[j, v, 1]] for (j, v) in left]
    return out


def _batch(stories: list[tuple[str, list[str]]], kind: str) -> list[list]:
    ids, lines = {}, []
    for j, nodes in stories:
        lines.append(f"Story {j}:")
        for v in nodes:
            i = f"n{len(ids)}"; ids[i] = (j, v); lines.append(f'  [{i}] "{v}"')
    if not ids:
        return []
    what, item = ("events", "event") if kind == "event" else ("participants", "participant")
    ans = chat_json(BATCH.format(what=what, n=len(stories), rules=RULES if kind == "event" else RULES_P,
                                 item=item, listing="\n".join(lines)))
    groups, placed = [], set()
    for g in ans.get("groups", []) if isinstance(ans, dict) else []:
        ms = []
        for m in g.get("members", []) if isinstance(g, dict) else []:
            i = m.get("id") if isinstance(m, dict) else m
            if i in ids and i not in placed:
                placed.add(i)
                ms.append([ids[i][0], ids[i][1], -1 if isinstance(m, dict) and m.get("opposite") and kind == "event" else 1])
        if ms:
            groups.append(ms)
    for _ in range(2):                                           # follow-up: ask again about nodes left out
        missing = [i for i in ids if i not in placed]
        if not missing:
            break
        STATS["followups"] += 1
        ans = chat_json(FOLLOW.format(what=what, item=item, rules=RULES if kind == "event" else RULES_P,
                                      groups="\n".join(f"  [G{k}] {_show(g)}" for k, g in enumerate(groups)) or "  (none)",
                                      items="\n".join(f'  [{i}] "{ids[i][1]}" (story {ids[i][0]})' for i in missing)))
        for m in ans.get("placements", []) if isinstance(ans, dict) else []:
            i = m.get("id") if isinstance(m, dict) else None
            if i not in ids or i in placed:
                continue
            sign = -1 if m.get("opposite") and kind == "event" else 1
            gid = str(m.get("group", "new"))
            k = int(gid.lstrip("Gg")) if gid.lstrip("Gg").isdigit() else None
            if k is not None and 0 <= k < len(groups):
                groups[k].append([ids[i][0], ids[i][1], sign])
            else:
                groups.append([[ids[i][0], ids[i][1], 1]])
            placed.add(i)
    return _fix(groups, set(ids.values()))


def _show(g: list, k: int = 5) -> str:
    pos = Counter(m[1] for m in g if m[2] > 0).most_common(k)
    neg = Counter(m[1] for m in g if m[2] < 0).most_common(2)
    s = "; ".join(f'"{v}"' for v, _ in pos)
    return s + (("; opposite: " + "; ".join(f'"{v}"' for v, _ in neg)) if neg else "")


def _merge(A: list[list], B: list[list], kind: str) -> list[list]:
    if not A or not B:
        return A + B
    what = "events" if kind == "event" else "participants"
    ans = chat_json(MERGE.format(what=what, rules=RULES if kind == "event" else RULES_P,
                                 a="\n".join(f"  [A{i}] {_show(g)}" for i, g in enumerate(A)),
                                 b="\n".join(f"  [B{i}] {_show(g)}" for i, g in enumerate(B))))
    out = [list(g) for g in A]
    joined, answered = set(), set()
    answers = list(ans.get("matches", []) if isinstance(ans, dict) else [])
    for m in answers:
        try: answered.add(int(str(m.get("b", "")).lstrip("Bb")))
        except ValueError: pass
    for _ in range(2):                                           # follow-up: ask again about groups left out
        missing = [i for i in range(len(B)) if i not in answered]
        if not missing:
            break
        STATS["followups"] += 1
        more = chat_json(FOLLOW_MERGE.format(what=what, rules=RULES if kind == "event" else RULES_P,
                                             a="\n".join(f"  [A{i}] {_show(g)}" for i, g in enumerate(A)),
                                             b="\n".join(f"  [B{i}] {_show(B[i])}" for i in missing)))
        for m in more.get("matches", []) if isinstance(more, dict) else []:
            try: bi = int(str(m.get("b", "")).lstrip("Bb"))
            except ValueError: continue
            if bi in missing and bi not in answered:
                answered.add(bi); answers.append(m)
    for m in answers:
        try:
            b = int(str(m.get("b", "")).lstrip("Bb")); a = m.get("a")
            a = None if a in (None, "null", "") else int(str(a).lstrip("Aa"))
        except ValueError:
            continue
        rel = m.get("relation")
        if b in joined or not (0 <= b < len(B)) or a is None or not (0 <= a < len(A)) or rel not in ("same", "opposite"):
            continue
        if rel == "opposite" and kind != "event":
            continue
        flip = -1 if rel == "opposite" else 1
        out[a] += [[j, v, s * flip] for j, v, s in B[b]]
        joined.add(b)
    out += [list(g) for i, g in enumerate(B) if i not in joined]
    allowed = {(m[0], m[1]) for g in A + B for m in g}
    return _fix(out, allowed)


def llm_group(graphs: list[dict], batch_size: int = 10, log=print) -> dict:
    """Groups of all story graphs (Section 2, Step 1, LLM setting) in abstraction.json format."""
    groups = []
    for kind in ("event", "participant"):
        level = []
        for i in range(0, len(graphs), batch_size):
            part = [(g["id"], [v for v in g["nodes"] if (g.get("kinds") or {}).get(v, "event") == kind])
                    for g in graphs[i:i + batch_size]]
            level.append(_batch(part, kind))
            log(f"  [{kind}] batch {i // batch_size + 1}: {len(level[-1])} groups")
        while len(level) > 1:                                    # tournament: merge neighbours, carry an odd one
            level = [_merge(level[k], level[k + 1], kind) if k + 1 < len(level) else level[k]
                     for k in range(0, len(level), 2)]
            log(f"  [{kind}] merged to {len(level)} list(s): {[len(x) for x in level]} groups")
        names = set()
        for g in (level[0] if level else []):
            first = next((m for m in g if m[2] > 0), g[0])
            name = first[1] if first[1] not in names else f"{first[1]} ({first[0]})"
            names.add(name)
            groups.append({"kind": kind, "name": name, "members": g})
    res = abstraction(groups, graphs, decisions=[])
    res["llm_group"] = dict(STATS, batch_size=batch_size)          # calls, follow-up calls, nodes forced into own group
    log(f"  LLM grouping: {STATS}")
    return res
