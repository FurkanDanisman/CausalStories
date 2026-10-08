"""Prompt builders — thin renderers over the .md files in prompt_templates/.

Every prompt lives in an editable markdown file. Editing a .md file changes the
prompt the pipeline sends on the very next run: templates are read from disk on
each call (no caching), and HTML comments (<!-- ... -->) are stripped so you can
annotate the files freely.

Template syntax:
  {{VAR}}            runtime value passed by the pipeline (e.g. {{TEXT}})
  {{GUIDANCE}}       auto-injected partial  <- guidance.md
  {{WORKED_EXAMPLE}} auto-injected partial  <- worked_example.md
"""

from __future__ import annotations

import os
import re
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent / "prompt_templates"
_PARTIALS = {"GUIDANCE": "guidance.md", "GUIDANCE_EVENTS": "guidance_events.md",
             "WORKED_EXAMPLE": "worked_example.md"}
_COMMENT = re.compile(r"<!--.*?-->\s*", re.DOTALL)


def _load(name: str) -> str:
    # A/B experiments: PROMPT_TEMPLATE_DIR overrides individual templates; any file
    # absent there falls back to the baseline dir (so a variant need only ship the
    # files it changes, and eval prompts align/judge stay on baseline).
    override = os.environ.get("PROMPT_TEMPLATE_DIR")
    path = Path(override) / name if override and (Path(override) / name).exists() else TEMPLATE_DIR / name
    return _COMMENT.sub("", path.read_text()).strip()


def render(template_file: str, **vars: str) -> str:
    text = _load(template_file)
    for key, fname in _PARTIALS.items():
        token = "{{" + key + "}}"
        if token in text:
            text = text.replace(token, _load(fname))
    for k, v in vars.items():
        text = text.replace("{{" + k + "}}", v)
    return text


def extract_nodes_prompt(text: str, max_nodes: int | None = None) -> str:
    limit = (f"Return at most {max_nodes} nodes: the {max_nodes} most important for "
             f"the causal chain." if max_nodes else "")
    return render("extract_nodes.md", TEXT=text, NODE_LIMIT=limit)


def _listing(node_ids: list[str]) -> str:
    return "\n".join(f"  - {nid}" for nid in node_ids)


# Kev choice options for kev_arrow.md; descriptions follow extract_edges.md.
KEV_ARROW_OPTIONS = {
    "enables": 'the first node makes the second happen, start, continue, or become more likely',
    "blocks": 'the first node makes the second stop, or prevents it / makes it less likely',
    "none": 'the text does not justify a direct causal arrow from the first node to the second',
}


def extract_events_prompt(text: str, max_nodes: int | None = None) -> str:
    limit = (f"Return at most {max_nodes} events: the {max_nodes} most important for "
             f"the causal chain." if max_nodes else "")
    return render("extract_events.md", TEXT=text, NODE_LIMIT=limit)


def extract_participants_prompt(text: str) -> str:
    return render("extract_participants.md", TEXT=text)


# Kev choice options for kev_agent.md (participant -> event); "yes" is drawn as an arrow.
KEV_AGENT_OPTIONS = {
    "yes": "the participant carries out or initiates the event in the text",
    "no": "the text does not say the participant carries out or initiates the event",
}


def kev_agent_question(head: str, tail: str, node_ids: list[str]) -> dict:
    return {"type": "choice", "criteria": KEV_AGENT_OPTIONS,
            "instructions": render("kev_agent.md", HEAD=head, TAIL=tail, NODE_LIST=_listing(node_ids))}


# Kev choice options for kev_implied.md (Section 2, Step 4).
KEV_IMPLIED_OPTIONS = {
    "happened": "the text states or implies that it happened",
    "not happened": "the text states or implies that it did not happen",
    "cannot tell": "the text does not say whether it happened",
}


def kev_implied_question(concept: str) -> dict:
    return {"type": "choice", "criteria": KEV_IMPLIED_OPTIONS,
            "instructions": render("kev_implied.md", CONCEPT=concept)}


def kev_arrow_question(head: str, tail: str, node_ids: list[str]) -> dict:
    return {"type": "choice", "criteria": KEV_ARROW_OPTIONS,
            "instructions": render("kev_arrow.md", HEAD=head, TAIL=tail, NODE_LIST=_listing(node_ids))}


def extract_edges_prompt(text: str, node_ids: list[str]) -> str:
    listing = "\n".join(f"  - {nid}" for nid in node_ids)
    return render("extract_edges.md", TEXT=text, NODE_LIST=listing)


def align_nodes_prompt(extracted_ids: list[str], gold_ids: list[str]) -> str:
    ex = "\n".join(f"  - {i}" for i in extracted_ids)
    go = "\n".join(f"  - {i}" for i in gold_ids)
    return render("align_nodes.md", EXTRACTED=ex, GOLD=go)


def judge_prompt(text: str, extracted_edges: list[dict], gold_edges: list[dict]) -> str:
    return render("judge.md", TEXT=text,
                  EXTRACTED_EDGES=str(extracted_edges), GOLD_EDGES=str(gold_edges))


def judge_edge_prompt(text: str, edges: list[tuple[str, str]]) -> str:
    listing = "\n".join(f"  [{i}] {h} -> {t}" for i, (h, t) in enumerate(edges))
    return render("judge_edge.md", TEXT=text, EDGES=listing)


# Angles for synthetic retellings — each induces a different aggregation challenge
# (omission, granularity, added detail, re-wording, reordering).
PERSPECTIVES = [
    "a brief summary that keeps only the two or three most central events",
    "a detailed eyewitness account that foregrounds one participant and describes "
    "their actions step by step, breaking events into finer sub-steps",
    "a secondhand retelling by someone unsure of some details, using vaguer, more "
    "general wording and omitting at least one event",
    "a fuller account that adds one plausible related event (a cause or a "
    "consequence) that fits the story but was not stated in the original",
    "a version told from a different participant's point of view, mentioning only "
    "what that person would plausibly know and omitting the rest",
    "a neutral rephrasing that uses different words for the same events and reorders "
    "the sentences",
]


def generate_variant_prompt(base_text: str, perspective: str) -> str:
    return render("generate_variant.md", TEXT=base_text, PERSPECTIVE=perspective)


def canonicalize_prompt(base_text: str, nodes_listing: str) -> str:
    return render("canonicalize.md", TEXT=base_text, NODES=nodes_listing)


def refute_prompt(text: str, events: list[str]) -> str:
    listing = "\n".join(f"  [{i}] {e}" for i, e in enumerate(events))
    return render("refute.md", TEXT=text, EVENTS=listing)


# Naming guidance for merge_concepts.md, by node kind.
MERGE_NAMING = {
    "event": "Name each concept by its meaning, as a short general phrase "
             "(e.g. \"hours were cut\", \"lost my job\" and \"was laid off\" -> \"loss of income\").",
    "participant": "Name each concept by the participant's role in the story, not by who they are. "
                   "The person the events happen to (e.g. \"I\", \"my brother\", \"she\") is "
                   "\"PROTAGONIST\". Other participants get a role name in capitals (e.g. "
                   "\"FAMILY MEMBER\", \"LANDLORD\", \"EMPLOYER\"). A narrator who only tells the "
                   "story and does not act in it is \"DROP\".",
}


def merge_concepts_prompt(kind: str, items: str) -> str:
    return render("merge_concepts.md", KIND=kind, NAMING=MERGE_NAMING[kind], ITEMS=items)

