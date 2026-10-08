<!--
NODE STEP, part 2: participants only. A participant is a person or organization that
carries out or initiates an event in the story (not things or places). Vars: {{TEXT}}.
-->
You identify the participants of a causal graph for a natural language text.

A participant is a person or organization that carries out or initiates an event in
the story: someone who pays, helps, evicts, fires, hires, decides, or leaves. Things
and places (e.g. work, street, flat, bed, shelter) are not participants. Include the
narrator ("I") only when the narrator carries out or initiates an event, not when
events only happen to them. Set "kind" to "participant" and leave `event_types` empty.
Use short ids (e.g. "I", "sister", "landlord", "employer").

--- worked example ---
Text: "My sister covered the rent, so the landlord did not evict me."
Participants:
  - {"id": "sister", "kind": "participant", "event_types": []}
  - {"id": "landlord", "kind": "participant", "event_types": []}
--- end example ---

Now extract the participants for this text:
"""{{TEXT}}"""
