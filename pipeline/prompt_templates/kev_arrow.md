<!--
KEV CHOICE QUESTION (instructions): the direct causal relation from HEAD to TAIL.
Same definition as extract_edges.md ({{GUIDANCE}}, "justify from the text"),
asked as one pair at a time with the full node list. The options
(enables / blocks / none) are in prompts.KEV_ARROW_OPTIONS.
Vars: {{HEAD}}, {{TAIL}}, {{NODE_LIST}}. Partials: {{GUIDANCE}}.
-->
{{GUIDANCE}}

The nodes of this text's causal graph are:
{{NODE_LIST}}

Based only on what you can justify from the text: what is the direct causal
relation from "{{HEAD}}" to "{{TAIL}}"? Direct means "{{HEAD}}" is a causal
factor for "{{TAIL}}" not only through another listed node.
