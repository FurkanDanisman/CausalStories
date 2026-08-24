<!--
PARTIAL: injected wherever {{WORKED_EXAMPLE}} appears.
A single worked example, DISJOINT from any dataset text we evaluate on (the Fig. 2
"rebels" sentence from Michael's paper), so it never leaks answers into eval.
Edit freely; add more examples if the model needs them.
-->
--- worked example ---
Text: "The rebels ousted the leader to end the conflict."
Nodes:
  - {"id": "rebels", "kind": "participant", "event_types": []}
  - {"id": "leader", "kind": "participant", "event_types": []}
  - {"id": "rebels ousted the leader", "kind": "event", "event_types": ["Change_of_leadership"]}
  - {"id": "the conflict", "kind": "event", "event_types": ["Hostile_encounter"]}
Arrows (head -> tail):
  - rebels -> rebels ousted the leader          (the rebels are the agent of the ousting)
  - rebels ousted the leader -> the conflict    (the ousting ends the conflict: still an arrow)
--- end example ---
