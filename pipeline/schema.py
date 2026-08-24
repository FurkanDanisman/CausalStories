"""Data model for the extracted causal graph.

Per the current design decision, edges are UNLABELED directed causal arrows
(A -> B means "A causally leads to / influences B"). We deliberately drop the
ENABLES/BLOCKS labels and fine-grained sub-relations from the paper: an arrow is
enough. Node typing (event vs participant) and FrameNet/MAVEN event types are
kept so the output still aligns with the gold TORQUESTRA node structure.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class NodeKind(str, Enum):
    EVENT = "event"
    PARTICIPANT = "participant"  # gold graphs prefix these "Entity::"


class Node(BaseModel):
    id: str = Field(..., description="Canonical name / text mention of the node.")
    kind: NodeKind
    event_types: List[str] = Field(
        default_factory=list,
        description="FrameNet/MAVEN event types (events only); empty for participants.",
    )


class Edge(BaseModel):
    head: str = Field(..., description="Source node id (the cause).")
    tail: str = Field(..., description="Target node id (the effect).")
    prob: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence / self-consistency frequency of this arrow across K samples.",
    )


class CausalGraph(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

    def node_ids(self) -> List[str]:
        return [n.id for n in self.nodes]

    def to_gold_like(self) -> list[dict]:
        """Serialize to the TORQUESTRA edge-list shape (minus the rel labels)."""
        return [{"head": e.head, "tail": e.tail, "prob": round(e.prob, 3)} for e in self.edges]


# ---- schemas handed to the LLM for tool-forced structured output ----

class NodeExtraction(BaseModel):
    nodes: List[Node]


class EdgeSpec(BaseModel):
    head: str
    tail: str


class EdgeExtraction(BaseModel):
    edges: List[EdgeSpec]


class NodeAlignment(BaseModel):
    """Maps each extracted node id to a gold node id (or null if unmatched)."""

    mapping: dict[str, Optional[str]]


class JudgeScore(BaseModel):
    score: int = Field(..., ge=1, le=5)
    rationale: str
