# Open problem: a cause whose effect was blocked is answered "none" (2026-11-26)

Status: noted, not fixed. Come back to this before any rerun.

## What happens
- The true world (worlds_random.py): eviction -> homelessness 1.0, blocked by providing a place to stay 1.0.
  In a story where family took the person in, eviction is still a cause of homelessness; its effect was blocked.
- Gemma 4 31B (grid_v8c, homelessness), arrow question eviction -> homelessness, stories with both nodes:
  - evicted and homeless (55 stories): enables
  - evicted, family took them in, not homeless (81 stories): none
  - average over 155 stories: enables 0.355, none 0.645 -> the world graph has no eviction -> homelessness arrow.
- Same in the earlier Gemma run (grid_v8: 0.355 / 0.645), so the new opposite question did not cause it.
  Qwen keeps the arrow only narrowly (enables 0.38, blocks 0.30, none 0.32): it answered "blocks" in many family stories.
- The protection itself is found (family housing support -| homelessness, 1.0). The cause it blocks gets diluted.

## Why
Arrow question (pipeline/prompt_templates/kev_arrow.md): "Based only on what you can justify from the text: what is
the direct causal relation from HEAD to TAIL?" In a blocked story the text shows only that TAIL did not happen, so the
model answers "did HEAD cause TAIL in this story?" -> no -> none. Nothing tells it that a cause still counts when
something else stopped its effect.

## Scope
Every cause-protection pair: can't pay rent -> eviction when savings cover it (if told), loneliness -> depression when
therapy helps, financial struggle -> can't pay rent with savings. Largest for eviction -> homelessness because the
protection is common (81 of 136 stories with eviction).
Open question: does the paper the method builds on (and the competing approaches) have the same problem?

## Possible fix (not applied; needs the user's approval of the wording)
One sentence in the arrow prompt, e.g. "If the text shows that something stopped TAIL from happening, HEAD still enables
TAIL when it would have made it more likely." Requires rerunning the chain walk and every later step.

## Regan et al. 2023 (arXiv 2303.15381), checked 2026-11-26
Our causal-edge definition (pipeline/prompt_templates/guidance_events.md) is their Table 2, but only 8 of its 12 rows.
Their Table 2 also has, under each relation:
- Enables / Without effect: "Despite expectations, no enabling effect" (cues: despite, even though); example about failing to get the ball rolling.
- Blocks / Without effect: "Despite expectations, no blocking effect".
- Enables / Unknown and Blocks / Unknown: uncertain relation (cues: questions, modality).
So in their ontology a cause whose effect was stopped is still Enables ("Without effect"). Our family stories are that case.
A.5: these rows are about 15% of Torquestra's relations, are "difficult to agree on", and more than half have no explicit cue word.
Fix consistent with the paper: add the "Without effect" rows to guidance_events.md (instead of the ad-hoc sentence above).

## One-story test, 2026-11-26 (ho002: "I got evicted, but my brother took me in, so I never ended up on the street")
Arrow eviction -> being homeless, Gemma 4 31B:
- cluster vLLM, chain walk, current definition: none 0.997
- OpenRouter, current definition, identical prompt (3,346 chars; same nodes, text, option order): enables 0.93
- OpenRouter, definition + Regan's two "Without effect" rows: none 1.00 (worse)
The "Without effect" variant is saved at $CLAUDE_JOB_DIR/tmp/guidance_events.without_effect.md; the repo file was restored.
Open: why the cluster and OpenRouter answers differ (serving / thinking switch / first-token logprobs?). Test planned:
send this exact prompt to the cluster vLLM Gemma and print the first token and its top alternatives.

## Decision models (grid_v8_or, homelessness, same Gemma nodes): eviction -> homelessness average "enables"
Gemma 0.35 (arrow lost) | Gemini Decision 1 0.56 | GPT-6 Luna Decisions 0.64 | Jev 1.14 0.54 (all keep the arrow).

## Next (user, 2026-11-27)
After the Llama-3.3-70B results: focus on why eviction -> homelessness fails and experiment with a few prompts
(the user expected Regan's two rows to fix it; they did not in the one-story test). Plan the experiments with the user
before running anything.

## Cluster test, 2026-11-27 (test_ho002_arrow.sbatch; our vLLM Gemma 4 31B, the chain walk's exact prompt)
- current definition, 5x: none 0.997 (blocks 0.003) every time = the chain walk's answer. First token 'C' (= none),
  alternatives B, A: the logprob reading is correct; our Gemma really answers none.
- with Regan's two "Without effect" rows, 5x: none 0.966 (blocks 0.033) every time.
- OpenRouter's Gemma 4 31B gave enables 0.93 on the same prompt: a difference in the served model (provider
  precision/serving), not in our code. Our results use our own full-weight Gemma.

## OpenRouter test, 2026-11-27 (same script, TEST_MODEL=llm:google/gemma-4-31b-it): the answer depends on the provider
current definition: DeepInfra none 1.00 (no logprobs: answer letter only) | NextBit none 0.967 | Parasail none 1.00 |
  CoreWeave enables 0.93 (2x)
Regan's rows: DeepInfra none 1.00 (3x, no logprobs) | Parasail none 0.962 | CoreWeave blocks 0.75
-> Gemma 4 31B answers none on every provider except CoreWeave; the earlier "OpenRouter enables 0.93" was CoreWeave.
   Our cluster Gemma agrees with the majority. The problem is the model's reading of the question, not serving.
