"""Prompt builders — thin renderers over the .md files in prompt_templates/.

Every prompt lives in an editable markdown file. Editing a .md file changes the
prompt the pipeline sends on the very next run: templates are read from disk on
each call (no caching), and HTML comments (<!-- ... -->) are stripped so you can
annotate the files freely.

Template syntax:
  {{VAR}}            runtime value passed by the pipeline (e.g. {{TEXT}})
  {{GUIDANCE_EVENTS}} auto-injected partial  <- guidance_events.md
"""

from __future__ import annotations

import os
import re
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent / "prompt_templates"
_PARTIALS = {"GUIDANCE_EVENTS": "guidance_events.md"}
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


# Kev choice options for kev_implied.md (Section 2, Step 4).
KEV_IMPLIED_OPTIONS = {
    "happened": "the text states or implies that it happened",
    "not happened": "the text states or implies that it did not happen",
    "cannot tell": "the text does not say whether it happened",
}


def kev_happened_question(node: str) -> dict:
    return {"type": "choice", "criteria": KEV_IMPLIED_OPTIONS,
            "instructions": render("kev_happened.md", NODE=node)}


def kev_implied_question(concept: str) -> dict:
    return {"type": "choice", "criteria": KEV_IMPLIED_OPTIONS,
            "instructions": render("kev_implied.md", CONCEPT=concept)}


# Kev choice options for kev_influence.md (event -> participant); positive is drawn like
# enables (green), negative like blocks (red).
KEV_INFLUENCE_OPTIONS = {
    "positive": "the event helps the participant or makes things better for them",
    "negative": "the event harms the participant or makes things worse for them",
    "none": "the text does not say the event influences the participant",
}


def kev_influence_question(head: str, tail: str, node_ids: list[str]) -> dict:
    return {"type": "choice", "criteria": KEV_INFLUENCE_OPTIONS,
            "instructions": render("kev_influence.md", HEAD=head, TAIL=tail, NODE_LIST=_listing(node_ids))}


def kev_arrow_question(head: str, tail: str, node_ids: list[str]) -> dict:
    return {"type": "choice", "criteria": KEV_ARROW_OPTIONS,
            "instructions": render("kev_arrow.md", HEAD=head, TAIL=tail, NODE_LIST=_listing(node_ids))}


# Kev choice options for kev_group.md / kev_group_participant.md (Section 2, Step 1).
KEV_GROUP_OPTIONS = {
    "same": "the event is the same kind of event as the group",
    "opposite": "the event is the opposite of the group: one happening means the other did not",
    "different": "the event is a different event from the group",
}
KEV_GROUP_P_OPTIONS = {
    "same": "the participant plays the same role as the group",
    "different": "the participant plays a different role from the group",
}


def kev_group_question(node: str, members: list[str], kind: str) -> dict:
    listing = "\n".join(f'  - "{m}"' for m in members)
    if kind == "participant":
        return {"type": "choice", "criteria": KEV_GROUP_P_OPTIONS,
                "instructions": render("kev_group_participant.md", NODE=node, MEMBERS=listing)}
    return {"type": "choice", "criteria": KEV_GROUP_OPTIONS,
            "instructions": render("kev_group.md", NODE=node, MEMBERS=listing)}


# Naming guidance for merge_concepts.md, by node kind.
MERGE_NAMING = {
    "event": "Name each concept by its meaning, as a short general phrase "
             "(e.g. \"hours were cut\", \"lost my job\" and \"was laid off\" -> \"loss of income\").",
    "participant": "Name each concept by the participant's role in the story, not by who they are. "
                   "The person the events happen to (e.g. \"I\", \"my brother\", \"she\") is "
                   "\"PROTAGONIST\". Other participants get a role name in capitals (e.g. "
                   "\"FAMILY MEMBER\", \"LANDLORD\", \"EMPLOYER\").",
}


def name_groups_prompt(kind: str, groups: str) -> str:
    return render("name_groups.md", KIND="events" if kind == "event" else "participants",
                  NAMING=MERGE_NAMING[kind], GROUPS=groups)


