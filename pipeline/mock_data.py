"""Canned answers for MockLLMClient so the pipeline runs offline end-to-end.

The "keating" fixture mimics a plausible (imperfect) LLM extraction of the short
train text for torque_id docid_PRI19980115.2000.0186_sentid_6. Node names are
deliberately paraphrased away from the gold names so the semantic-alignment step
in evaluation is actually exercised, and one spurious participant->event edge is
included so precision is < 1.0 (a realistic result, not a rigged 100%).
"""

from __future__ import annotations

from typing import Optional, Type

from pydantic import BaseModel

_KEATING = {
    "extract_nodes": {
        "nodes": [
            {"id": "ninth US circuit court of appeals", "kind": "participant", "event_types": []},
            {"id": "Keating", "kind": "participant", "event_types": []},
            {"id": "Keating was released from prison", "kind": "event",
             "event_types": ["Releasing", "Legal_rulings"]},
            {"id": "Keating was eligible for parole", "kind": "event",
             "event_types": ["Legality"]},
            {"id": "court ruled the original appeal was flawed", "kind": "event",
             "event_types": ["Legal_rulings"]},
            {"id": "issues had not been raised before", "kind": "event",
             "event_types": ["Scenario"]},
        ]
    },
    "extract_edges": {
        "edges": [
            {"head": "court ruled the original appeal was flawed",
             "tail": "Keating was released from prison"},
            {"head": "Keating was eligible for parole",
             "tail": "Keating was released from prison"},
            {"head": "issues had not been raised before",
             "tail": "court ruled the original appeal was flawed"},
            {"head": "ninth US circuit court of appeals",
             "tail": "court ruled the original appeal was flawed"},
        ]
    },
    "align_nodes": {
        "mapping": {
            "court ruled the original appeal was flawed": "ruled that the original appeal was flawed",
            "Keating was released from prison": "Keating was released",
            "Keating was eligible for parole": "Keating not eligible for parole",
            "issues had not been raised before": "issues that had not been raised before",
            "ninth US circuit court of appeals": None,
            "Keating": None,
        }
    },
    "judge": {
        "score": 4,
        "rationale": "Recovers the core release/ruling causal chain; adds a defensible "
                     "court->ruling agent edge not present in gold. Polarity of the parole "
                     "node differs but the event referent matches.",
    },
    "judge_edges": {
        "verdicts": [
            {"index": 0, "valid": True},
            {"index": 1, "valid": True},
            {"index": 2, "valid": True},
            {"index": 3, "valid": True},
        ]
    },
}

_FIXTURES = {"keating": _KEATING}


def answer(fixture: str, task: str) -> Optional[dict]:
    return _FIXTURES.get(fixture, {}).get(task)


def empty_for(schema: Type[BaseModel]) -> dict:
    name = schema.__name__
    return {
        "NodeExtraction": {"nodes": []},
        "EdgeExtraction": {"edges": []},
        "NodeAlignment": {"mapping": {}},
        "JudgeScore": {"score": 1, "rationale": "n/a"},
        "EdgeValidityBatch": {"verdicts": []},
        "Variant": {"text": ""},
        "RefuteBatch": {"verdicts": []},
        "NodeClustering": {"mapping": {}},
        "CausalGraph": {"nodes": [], "edges": []},
    }.get(name, {})
