<!--
SECTION 2, STEP 1 (naming): give each group of nodes a high-level concept name
(constructive abstraction, Beckers & Halpern 2019). The grouping itself is already
decided by Kev; this call only names. Vars: {{KIND}}, {{NAMING}}, {{GROUPS}}.
-->
Nodes from different people's stories have been grouped: each group below holds {{KIND}}
that describe the same thing. Give each group a high-level name.

The groups are a low-level description of the stories; the names are a high-level
description of the same stories (Beckers and Halpern, "Abstracting Causal Models"). A
name must cover every member of its group and describe nothing the members do not.
Members marked "(opposite)" mean the opposite of the group: name the group by the
meaning of its other members. For a group with opposite members, also give the
high-level name of the opposite meaning (e.g. "eviction" and "keeping the home"; "rent
arrears" and "paying rent on time").

{{NAMING}}

Groups:
{{GROUPS}}

Return a JSON object whose "mapping" maps EVERY group id (e.g. "g0") to its name, and,
for each group with opposite members, "<id>-opposite" (e.g. "g0-opposite") to the name
of its opposite meaning.
Give different groups different names.
