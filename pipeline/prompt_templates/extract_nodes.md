<!--
STAGE 1 PROMPT: node extraction. Vars: {{TEXT}}. Partials: {{GUIDANCE}}, {{WORKED_EXAMPLE}}.
-->
You identify the nodes of a causal graph for a natural language text.

{{GUIDANCE}}

Nodes are of two kinds:
  * "event": a salient happening, in subject-verb-object form (a short natural language description, not a single token). Assign 1-3 FrameNet/MAVEN
    `event_types` (e.g. "Releasing", "Legal_rulings", "Change_of_leadership").
  * "participant": a person, organization, or thing that acts in the story.
    Leave `event_types` empty for participants.

Extract only nodes that participate in the causal story. Omit purely decorative mentions. Use concise, self-contained node ids that will make sense on their own.

{{WORKED_EXAMPLE}}

Now extract the nodes for this text:
"""{{TEXT}}"""
