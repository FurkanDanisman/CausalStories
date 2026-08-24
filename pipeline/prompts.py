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

import re
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent / "prompt_templates"
_PARTIALS = {"GUIDANCE": "guidance.md", "WORKED_EXAMPLE": "worked_example.md"}
_COMMENT = re.compile(r"<!--.*?-->\s*", re.DOTALL)


def _load(name: str) -> str:
    return _COMMENT.sub("", (TEMPLATE_DIR / name).read_text()).strip()


def render(template_file: str, **vars: str) -> str:
    text = _load(template_file)
    for key, fname in _PARTIALS.items():
        token = "{{" + key + "}}"
        if token in text:
            text = text.replace(token, _load(fname))
    for k, v in vars.items():
        text = text.replace("{{" + k + "}}", v)
    return text


def extract_nodes_prompt(text: str) -> str:
    return render("extract_nodes.md", TEXT=text)


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
