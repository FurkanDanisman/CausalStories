<!--
SECTION 2, STEP 3: merge nodes into shared concepts (constructive abstraction,
Beckers & Halpern 2019). One call per subgroup per round; all items are of ONE
kind (events or participants), so rule R6 is enforced by the caller.
Vars: {{KIND}}, {{NAMING}}, {{ITEMS}}.
-->
You merge the {{KIND}} nodes of causal graphs from different narratives into shared
concepts. Each item below is a node (or a concept from an earlier round), with the
narrative(s) it comes from, what it causes, and what causes it.

Rules:
  R1. Every item goes to exactly one concept, or to "DROP" if it has no causal role.
  R2. Use only concepts that the items support; do not invent concepts.
  R3. Merge items only if they have the same causal role: the same kind of causes
      and the same kind of effects.
  R4. If two concepts cannot vary separately (one always comes with the other),
      merge them into one.
  R5. Never merge two items when one causes the other: they are cause and effect,
      not the same thing.

{{NAMING}}

Items:
{{ITEMS}}

Return a JSON object whose "mapping" maps EVERY item id (e.g. "n0") to its concept
name, or to "DROP". Items with the same concept name are merged.
