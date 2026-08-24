<!--
STAGE 1 PROMPT: node extraction. Vars: {{TEXT}}. Partials: {{GUIDANCE}}, {{WORKED_EXAMPLE}}.
-->
You identify the nodes of a causal graph for a natural language text.

{{GUIDANCE}}

Nodes are of two kinds:
  * "event": a salient happening, in subject-verb-object form (a short natural language description, not a single token). Assign 1-3 FrameNet/MAVEN
    `event_types` (e.g. "Releasing", "Legal_rulings", "Change_of_leadership").
  * "participant": a person, organization, or thing — a grammatical subject or
    object associated with the events. Leave `event_types` empty for participants.

Extract the salient events and participants of the causal story; omit mentions
that do not contribute to it. Use concise, self-contained node ids in
subject-verb-object form that will make sense on their own.

{{WORKED_EXAMPLE}}

Now extract the nodes for this text:
"""{{TEXT}}"""
