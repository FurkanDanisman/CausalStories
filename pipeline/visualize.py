"""Render the predicted and gold graphs to Graphviz DOT / SVG and a side-by-side
HTML comparison.

Colour code (same in both panels):
  * node fill: orange = participant, red = event (matches the paper's figures)
  * edge GREEN  = arrow present in BOTH graphs (a correct match)
  * edge ORANGE = predicted-only arrow (false positive) — predicted panel
  * edge GREY   = gold-only arrow the model missed (false negative) — gold panel
"""

from __future__ import annotations

import shutil
import subprocess

from .dataset import Example
from .evaluate import EvalReport
from .schema import CausalGraph, NodeKind

_PARTICIPANT_FILL = "#F6C177"   # orange
_EVENT_FILL = "#EA9999"         # red
_MATCH = "#2E8B57"              # green
_FP = "#E69138"                # orange
_FN = "#999999"                # grey


def _esc(s: str) -> str:
    return s.replace('"', "'")


def _dot(title: str, nodes: list[tuple[str, str]],
         edges: list[tuple[str, str, str, str]]) -> str:
    """nodes: (id, 'participant'|'event'); edges: (head, tail, color, label)."""
    out = ["digraph G {", f'  label="{_esc(title)}"; labelloc=t; rankdir=TB;',
           '  node [style=filled, fontname="Helvetica", fontsize=11];',
           '  edge [fontname="Helvetica", fontsize=9];']
    for nid, kind in nodes:
        fill = _PARTICIPANT_FILL if kind == "participant" else _EVENT_FILL
        shape = "ellipse" if kind == "participant" else "box"
        label = _esc(nid[len("Entity::"):] if nid.startswith("Entity::") else nid)
        out.append(f'  "{_esc(nid)}" [shape={shape}, fillcolor="{fill}", label="{label}"];')
    for head, tail, color, label in edges:
        lab = f', label="{label}"' if label else ""
        out.append(f'  "{_esc(head)}" -> "{_esc(tail)}" [color="{color}", '
                   f'penwidth=2.0, fontcolor="{color}"{lab}];')
    out.append("}")
    return "\n".join(out)


def predicted_dot(graph: CausalGraph, report: EvalReport) -> str:
    nodes = [(n.id, n.kind.value) for n in graph.nodes]
    edges = []
    for e in graph.edges:
        matched = (e.head, e.tail) in report.matched_pred
        edges.append((e.head, e.tail, _MATCH if matched else _FP, f"{e.prob:.2f}"))
    return _dot("PREDICTED (green=correct, orange=false positive)", nodes, edges)


def gold_dot(example: Example, report: EvalReport) -> str:
    nodes = [(nid, "participant" if nid.startswith("Entity::") else "event")
             for nid in example.gold_node_ids]
    # ensure any edge endpoints not in gold_node_ids still get drawn
    known = set(example.gold_node_ids)
    for e in example.gold_edges:
        for nid in (e["head"], e["tail"]):
            if nid not in known:
                known.add(nid)
                nodes.append((nid, "participant" if nid.startswith("Entity::") else "event"))
    edges = []
    for e in example.gold_edges:
        recovered = (e["head"], e["tail"]) in report.recovered_gold
        edges.append((e["head"], e["tail"], _MATCH if recovered else _FN, ""))
    return _dot("GOLD / TRUTH (green=recovered, grey=missed)", nodes, edges)


def render_svg(dot_str: str) -> str | None:
    """Return inline <svg>… via the `dot` binary, or None if graphviz is absent."""
    if not shutil.which("dot"):
        return None
    svg = subprocess.run(["dot", "-Tsvg"], input=dot_str, capture_output=True,
                         text=True, check=True).stdout
    return svg[svg.index("<svg"):]  # strip xml/doctype header so it inlines cleanly


def comparison_html(gold_svg: str, pred_svg: str, report: EvalReport,
                    meta: dict) -> str:
    rows = "".join(
        f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
        for k, v in {
            "structurally valid": report.valid,
            "edge precision": f"{report.precision:.3f}",
            "edge recall": f"{report.recall:.3f}",
            "edge F1": f"{report.f1:.3f}",
            "matched / pred / gold": f"{report.matched} / {report.n_pred} / {report.n_gold}",
            "LLM-judge": f"{report.judge_score}/5",
        }.items())
    banner = ("" if not meta.get("is_mock") else
              '<p class="warn">⚠ Backend = <b>mock</b>: the PREDICTED graph is '
              'hand-authored canned output, not a real model. Swap in a real backend '
              'for genuine accuracy.</p>')
    return f"""<div style="font-family:Helvetica,Arial,sans-serif;max-width:1200px;margin:auto">
<h2>Causal graph: predicted vs. truth</h2>
<p><b>{meta.get('torque_id','')}</b> · split=<b>{meta.get('split','')}</b> ·
backend=<b>{meta.get('backend','')}</b></p>
{banner}
<p style="color:#555">{_esc(meta.get('text',''))}</p>
<table style="border-collapse:collapse" border="1" cellpadding="5">{rows}</table>
<p style="color:#555"><i>Judge:</i> {_esc(report.judge_rationale)}</p>
<div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap">
  <div style="flex:1;min-width:340px">{gold_svg or '<i>graphviz not installed</i>'}</div>
  <div style="flex:1;min-width:340px">{pred_svg or '<i>graphviz not installed</i>'}</div>
</div>
<style>.warn{{background:#fff3cd;border:1px solid #ffe08a;padding:8px 12px;border-radius:6px}}</style>
</div>"""
