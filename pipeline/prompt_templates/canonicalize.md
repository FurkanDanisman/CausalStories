<!--
AGGREGATION PROMPT: cluster nodes from N retellings of the SAME story into
canonical concepts, so the same event/participant worded differently is unified.
Vars: {{TEXT}}, {{NODES}}.
-->
Several people retold the same real events. Below are the graph nodes extracted
from their separate retellings. The same underlying event or participant is often
worded differently across retellings (e.g. "suffocated Aguda", "smothered her",
"pressed a pillow over her face" all denote one event).

Group the nodes that denote the SAME underlying event or participant, and give
each group ONE canonical name (a clear subject-verb-object phrase for an event; a
short name for a participant). Return a mapping from EVERY listed node, verbatim,
to its canonical name. Nodes that denote the same thing MUST map to the same
canonical name; genuinely distinct nodes keep their own.

Story (for reference):
"""{{TEXT}}"""

Nodes (from all retellings):
{{NODES}}
