<!--
PARTIAL: injected wherever {{WORKED_EXAMPLE}} appears.
The paper's own example (Fig. 2 instance graph + Fig. 4 event types) for
"The rebels ousted the leader to end the conflict." Arrows and event types are
exactly those the paper draws; it is disjoint from any dataset text we evaluate on.
-->
--- worked example ---
Text: "The rebels ousted the leader to end the conflict."
Nodes:
  - {"id": "rebels", "kind": "participant", "event_types": []}
  - {"id": "leader", "kind": "participant", "event_types": []}
  - {"id": "ousted the leader", "kind": "event", "event_types": ["Change_of_leadership"]}
  - {"id": "the conflict", "kind": "event", "event_types": ["Military_operation"]}
Arrows (head -> tail):
  - rebels -> ousted the leader        (the rebels initiate the ousting)
  - rebels -> leader                   (the rebels act on the leader)
  - ousted the leader -> leader        (the ousting removes the leader: a causal endpoint)
  - ousted the leader -> the conflict  (the ousting ends the conflict)
--- end example ---
