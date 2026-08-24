<!--
EVAL PROMPT: holistic LLM-judge score. Vars: {{TEXT}}, {{EXTRACTED_EDGES}}, {{GOLD_EDGES}}.
-->
Rate how well an extracted causal graph captures the causal story of a text,
compared to a gold reference graph. Ignore edge labels; judge whether the right
participants/events and the right directed causal arrows are present.

Score 1-5 (5 = captures the gold causal structure with no important errors).

Text:
"""{{TEXT}}"""

Extracted arrows (head -> tail): {{EXTRACTED_EDGES}}

Gold arrows (head -> tail): {{GOLD_EDGES}}
