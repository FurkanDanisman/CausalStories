"""Causal graph extraction pipeline (single-example validation).

Goal of this package: prove that, for ONE narrative from TORQUESTRA, an LLM can
extract a causal graph of (participants, events, directed causal arrows) that we
can score against the gold human graph.

The design is model-agnostic: every LLM call goes through the LLMClient protocol
(see llm_client.py), so the same pipeline runs against a mock, an API model, or a
local open-source model by swapping one object.
"""
