<!--
STAGE 1 PROMPT: node extraction. Vars: {{TEXT}}. Partials: {{GUIDANCE}}, {{WORKED_EXAMPLE}}.

VARIANT B change (Q2, over-generation / salience): the extraction paragraph is
tightened with paper-cited constraints. Only CAUSAL participants and salient
events become nodes (§4.6 annotation task: "identify and label causal
participants (nodes) ... and salient causal chains"); a causal graph has
"vertices V for salient events and participants" (§4.4); nodes are natural
language, "typically of subject-verb-object form" (§4.4); and the target is
"salient and connected edges" (§6). No caps or blocklists are invented; the only
levers are the paper's salience, causal-participation, connectedness, and SVO
node form.
-->
You identify the nodes of a causal graph for a natural language text.

{{GUIDANCE}}

Nodes are of two kinds:
  * "event": a salient happening, in subject-verb-object form (a short natural language description, not a single token). Assign 1-3 FrameNet/MAVEN
    `event_types` (e.g. "Releasing", "Legal_rulings", "Change_of_leadership").
  * "participant": a person, organization, or thing — a grammatical subject or
    object associated with the events. Leave `event_types` empty for participants.

Extract only the CAUSAL participants and SALIENT events of the causal story: the
paper's annotation task is to "identify and label causal participants (nodes) ...
and salient causal chains" (§4.6), and a causal graph has "vertices V for salient
events and participants" (§4.4). A mention becomes a node only if it causally
contributes to the story and connects into the causal chain — aim for "salient
and connected edges" (§6). Omit mentions that neither cause nor are caused by an
event and that only elaborate detail (for example, a chain of nested mentions
that connect only to each other and not to the central events); collapse such
detail into the single salient node it modifies. Each event node is a short
subject-verb-object description (§4.4), never a single bare token. Use concise,
self-contained node ids that will make sense on their own.

{{WORKED_EXAMPLE}}

Now extract the nodes for this text:
"""{{TEXT}}"""
