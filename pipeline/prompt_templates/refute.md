<!--
IMPUTATION PRE-PASS: for each candidate event NOT found in a narrator's graph,
decide whether the narrative RULES IT OUT (refuted -> a hard 0) versus simply
does not mention it (leave as missing/NA for MICE). Vars: {{TEXT}}, {{EVENTS}}.
-->
Read the narrative below. For each candidate event, decide whether the narrative
indicates that event did NOT happen to this person — i.e. it is contradicted or
clearly ruled out by what they say (refuted = true) — as opposed to simply not
being mentioned (refuted = false).

Only mark refuted = true when the text implies the event did not occur. If the
text is silent about it, mark refuted = false.

Narrative:
"""{{TEXT}}"""

Candidate events:
{{EVENTS}}
