<!--
SECTION 2, STEP 3: merge nodes into shared concepts (constructive abstraction,
Beckers & Halpern 2019, "Abstracting Causal Models"). One call per subgroup per round;
all items are of ONE kind (events or participants), so rule R6 is enforced by the caller.
Vars: {{KIND}}, {{NAMING}}, {{ITEMS}}.
-->
You merge the {{KIND}} nodes of causal graphs from different narratives into shared
concepts. Each item below is a node (or a concept from an earlier round), with the
narrative(s) it comes from, what it causes, and what causes it.

Definitions (Beckers and Halpern, "Abstracting Causal Models"):
  * Low level and high level. The items are a low-level description of the stories;
    the concepts you create are a high-level description of the same stories. The
    high level is an abstraction of the low level only if it keeps the causal
    relations: what a group of items causes at the low level, their concept must cause
    at the high level.
  * Constructive abstraction. Each concept is made from one group of items. The groups
    do not overlap: every item is in exactly one group. Items that play no causal role
    may be left out (DROP); every other item belongs to exactly one concept.
  * Items must work the same way. Items can be grouped only if they have the same
    effect when they happen. The paper's example: do not combine X, Y and Z into
    "X + Y + Z" if two settings with the same sum lead to different outcomes. In
    stories: do not merge "lost my job" and "got a raise" into "job change", because
    they lead to different outcomes.
  * No extra detail at the high level. The high level must not describe anything the
    low level does not: every concept must come from items.
  * Concepts must be able to vary separately. If two concepts can never change one
    without the other, they are one concept.

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
