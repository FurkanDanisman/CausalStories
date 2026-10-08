<!--
NODE STEP, part 2: participants only. Vars: {{TEXT}}. Partials: {{GUIDANCE}}.
-->
You identify the participants of a causal graph for a natural language text.

{{GUIDANCE}}

A participant is a person, organization, or thing named in the text: a grammatical
subject or object associated with the events. Set "kind" to "participant" and leave
`event_types` empty.

Include every participant who carries out or initiates an event in the text (for
example, someone who pays, helps, evicts, fires, decides, or leaves). Include the
narrator ("I") whenever the narrator carries out or initiates an event. Use short ids
(e.g. "I", "sister", "landlord", "employer").

--- worked example ---
Text: "My sister covered the rent, so the landlord did not evict me."
Participants:
  - {"id": "sister", "kind": "participant", "event_types": []}
  - {"id": "landlord", "kind": "participant", "event_types": []}
--- end example ---

Now extract the participants for this text:
"""{{TEXT}}"""
