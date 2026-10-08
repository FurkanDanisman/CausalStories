<!--
NODE STEP, part 1: events only. Same definitions and inclusion rule as
extract_nodes.md, restricted to events. Vars: {{TEXT}}, {{NODE_LIMIT}}.
Partials: {{GUIDANCE}}.
-->
You identify the events of a causal graph for a natural language text.

{{GUIDANCE}}

An event is a salient happening, in subject-verb-object form (a short natural
language description, never a single token). Set "kind" to "event" and assign 1-3
FrameNet/MAVEN `event_types` (e.g. "Releasing", "Legal_rulings", "Change_of_leadership").

Include an event only if it causally contributes to the story and connects into the
causal chain. Omit mentions that neither cause nor are caused by an event and only add
detail; fold such detail into the single event it modifies. Use concise,
self-contained ids.
{{NODE_LIMIT}}

--- worked example ---
Text: "The rebels ousted the leader to end the conflict."
Events:
  - {"id": "ousted the leader", "kind": "event", "event_types": ["Change_of_leadership"]}
  - {"id": "the conflict", "kind": "event", "event_types": ["Military_operation"]}
--- end example ---

Now extract the events for this text:
"""{{TEXT}}"""
