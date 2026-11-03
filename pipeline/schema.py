"""Schemas for the structured LLM outputs: node extraction (Step 1) and group naming."""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class NodeKind(str, Enum):
    EVENT = "event"
    PARTICIPANT = "participant"


class Node(BaseModel):
    id: str = Field(..., description="Canonical name / text mention of the node.")
    kind: NodeKind
    event_types: List[str] = Field(
        default_factory=list,
        description="FrameNet/MAVEN event types (events only); empty for participants.",
    )


class NodeExtraction(BaseModel):
    nodes: List[Node]


class NodeClustering(BaseModel):
    """Maps each node (verbatim, across all retellings) to a canonical name, so the
    same event/participant worded differently in different retellings is unified."""

    mapping: dict[str, str]
