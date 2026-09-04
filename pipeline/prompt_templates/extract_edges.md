<!--
STAGE 2 PROMPT: edge (arrow) extraction. Vars: {{TEXT}}, {{NODE_LIST}}.
Partials: {{GUIDANCE}}, {{WORKED_EXAMPLE}}.
-->
You draw the directed causal arrows of a causal graph.

{{GUIDANCE}}

Use ONLY these node ids (do not invent new ones); copy them verbatim:
{{NODE_LIST}}

Return every arrow head -> tail you can justify from the text. Arrows are
directed (cause -> effect). The graph may contain cycles; that is allowed.

Label each arrow with its relation `rel`:
  * "enables"  if the head makes the tail happen, start, continue, or become more
    likely (direction (1) above);
  * "blocks"   if the head makes the tail stop, or prevents it / makes it less
    likely (direction (2) above).

{{WORKED_EXAMPLE}}

Text:
"""{{TEXT}}"""
