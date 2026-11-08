<!--
NODE STEP, part 1: events only. Same definitions and inclusion rule as
extract_nodes.md, restricted to events. Vars: {{TEXT}}, {{NODE_LIMIT}}.
Partials: {{GUIDANCE_EVENTS}}.
-->
You identify the events of a causal graph for a natural language text.

{{GUIDANCE_EVENTS}}

An event is a salient happening, named by what happens (a short natural language
description). Name the event at a high level, the way it happens in the story:
"eviction", "being homeless", "not paying the rent", not "eviction happened", or
"paying the rent" when the rent was not paid. Leave out who does it: "covering rent", not "sister covered
rent". People, organizations, places and things are not events. Set "kind" to "event" and assign 1-3
FrameNet/MAVEN `event_types` (e.g. "Releasing", "Legal_rulings", "Change_of_leadership").

Include an event only if it causally contributes to the story and connects into the
causal chain. Omit mentions that neither cause nor are caused by an event and only add
detail; fold such detail into the single event it modifies.
Each event must be a different thing that happened. Never list two events for one
happening, even when the story describes it twice or in two steps: "the car broke
down" and "the car stopped working" are one event; "a flood warning" and "the flood"
are one event. List it once, with the most general name. Never list an event together
with its opposite: when the story says something was avoided, name the event that was
avoided ("the flood"), not its opposite ("staying dry").
Use concise, self-contained ids.
{{NODE_LIMIT}}

--- worked example ---
Text: "The rebels ousted the leader to end the conflict."
Events:
  - {"id": "ousting the leader", "kind": "event", "event_types": ["Change_of_leadership"]}
  - {"id": "the conflict", "kind": "event", "event_types": ["Military_operation"]}
--- end example ---

Now extract the events for this text:
"""{{TEXT}}"""
