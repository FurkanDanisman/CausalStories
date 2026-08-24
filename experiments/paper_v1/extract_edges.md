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

{{WORKED_EXAMPLE}}

Text:
"""{{TEXT}}"""
