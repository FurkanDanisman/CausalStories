<!--
EVAL PROMPT: semantic node alignment (extracted -> gold). Vars: {{EXTRACTED}}, {{GOLD}}.
Used only for scoring; not part of extraction.
-->
Two annotators described the same story with differently-worded nodes.

For each EXTRACTED node, choose the GOLD node that refers to the same underlying
event or participant, or null if none does. Match by referent, not surface form:
"court ruled the appeal was flawed" and "ruled that the original appeal was
flawed" are the same event; a difference in polarity (eligible vs not eligible)
still refers to the same underlying event and should be matched.

EXTRACTED nodes:
{{EXTRACTED}}

GOLD nodes:
{{GOLD}}

Return a mapping from every extracted node id to a gold node id or null.
