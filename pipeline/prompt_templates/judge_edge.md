<!--
EVAL PROMPT: per-edge causal validity, judged against the TEXT and the causal-edge
definition (NOT against the gold graph). Gives precision-by-paper. Vars: {{TEXT}},
{{EDGES}}. Partial: {{GUIDANCE}} (the same causal-edge definition used to extract).
-->
You check whether each proposed causal edge is valid for a text.

{{GUIDANCE}}

For the text below, decide for EACH proposed edge whether it is a valid causal
edge under the definition above: the text must support that the head is a causal
factor for the tail (in either direction of causation). Judge only against the
text and the definition — not against any other graph.

Return a verdict for every edge, referencing it by its index.

Text:
"""{{TEXT}}"""

Proposed edges:
{{EDGES}}
