<!--
STAGE 1 PROMPT: node extraction. Node membership is constrained only by paper
levers, with the citations kept HERE in the comment, not in the prompt body:
  - only CAUSAL participants and SALIENT events are nodes (§4.6 annotation task
    "identify and label causal participants (nodes) ... and salient causal chains";
    §4.4 "vertices V for salient events and participants")
  - nodes/edges form the salient, CONNECTED causal chain (§6 "salient and
    connected edges")
  - event nodes are subject-verb-object, never a bare token (§4.4)
No caps, blocklists, or place-name rules (those are not in the paper).
Vars: {{TEXT}}. Partials: {{GUIDANCE}}, {{WORKED_EXAMPLE}}.
-->
You identify the nodes of a causal graph for a natural language text.

{{GUIDANCE}}

Nodes are of two kinds:
  * "event": a salient happening, in subject-verb-object form (a short natural
    language description, never a single token). Assign 1-3 FrameNet/MAVEN
    `event_types` (e.g. "Releasing", "Legal_rulings", "Change_of_leadership").
  * "participant": a person, organization, or thing — a grammatical subject or
    object associated with the events. Leave `event_types` empty for participants.

Include a mention as a node only if it is a causal participant or a salient event:
it must causally contribute to the story and connect into the causal chain. Omit
mentions that neither cause nor are caused by an event and only add detail — for
example, a chain of nested mentions that connect to each other but not to the
central events; fold such detail into the single node it modifies. Keep the graph
to the salient, connected causal chain. Use concise, self-contained node ids.

{{WORKED_EXAMPLE}}

Now extract the nodes for this text:
"""{{TEXT}}"""
