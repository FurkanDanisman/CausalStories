# Benchmark worlds and stories (spec)

Code: `benchmark/worlds_random.py`. Stories: `benchmark/stories_v7/<world>.json` (200 per world).
Scorer: `benchmark/evaluate_run.py` (reads `stories_v7` by default; `STORIES=stories_v3` rescores the earlier
benchmark). Earlier worlds and stories (`world_homelessness.py`, `world_mental.py`, `stories_v3`-`v6`) are kept
only for the earlier results.

## 1. The worlds

Each world is a causal Bayesian network over binary nodes. Roots happen with a rate. Every other event follows a
noisy-OR with blockers: each present cause u fires with chance p_u, each present blocker b fires with chance
q_b, and the event happens if some cause fired and no blocker fired:

P(v = 1 | parents) = [1 − Π over present causes (1 − p_u)] × Π over present blockers (1 − q_b).

The chance in each arrow stands for everything outside the graph (a patient landlord, a severance payment, ...).
Participants (UPPERCASE) carry out one event. Each blocker protects exactly one event. No hidden variables, no
distractors.

**Homelessness**

| Root | Rate |
|---|---|
| EMPLOYER | 0.55 |
| divorce | 0.45 |
| savings | 0.30 |
| FAMILY MEMBER | 0.80 |

| Event | Causes (p) | Blocker (q) |
|---|---|---|
| job loss | EMPLOYER 1.00 | |
| providing a place to stay | FAMILY MEMBER 0.50 | |
| financial struggle | job loss 1.00, divorce 0.60 | |
| can't pay rent | financial struggle 1.00 | savings 0.80 |
| eviction | can't pay rent 1.00 | |
| homelessness | eviction 1.00 | providing a place to stay 1.00 |

**Mental health**

| Root | Rate |
|---|---|
| losing a family member | 0.70 |
| breakup | 0.45 |
| CLOSE FRIENDS | 0.80 |
| THERAPIST | 0.40 |

| Event | Causes (p) | Blocker (q) |
|---|---|---|
| social support | CLOSE FRIENDS 0.50 | |
| therapy | THERAPIST 1.00 | |
| loneliness | losing a family member 1.00, breakup 0.60 | social support 0.60 |
| depression | loneliness 0.70 | therapy 0.70 |

## 2. Who tells a story

A person drawn at random tells a story if at least one starting event happened to them (homelessness: job loss
or divorce; mental health: losing a family member or a breakup). This selects on parentless events only, so every
P(child | parents) is unchanged (assumption E1 of the method). 200 people per world (seeds 900, 901).

## 3. What a story tells

From the person's true values:
- every starting event that happened, and the participant who carried it out;
- for every event at stake (at least one of its causes happened):
  - if it happened: every cause that happened brought it on;
  - if it did not happen and its protection was there: the protection kept it from happening;
  - if it did not happen and no protection was there: it did not follow;
  - and its protection, if it has one and it could happen (a root, or its own cause happened): there or not
    there. A protection that could not happen (no family member, no close friends) is not told;
  - for financial struggle only: if it did not happen and job loss did not happen, "I still had my job" (job loss told as 0, no arrow;
    it is the absent cause, so the arrow job loss -> financial struggle already covers it); likewise for loneliness:
    if it did not happen and losing a family member did not happen, "I still had my family";
- nothing about events that were not at stake.

So nothing is told that played no part in the person's life, and everything that did is told, including what
did not happen.

## 4. Truth of a story

`truth["nodes"]`: every told node with its value (1 or 0). `truth["arrows"]`: the arrows above (carries out,
enables, risk, stopped, failed). The scorer's true story table is these values; a node a story does not tell is
unknown for that story.

## 5. Writing and checking the stories

Stories are written by Claude from each person's facts (writing guide: natural first-person speech, explicit
causal words, nothing beyond the facts, no timing between unrelated events, each protection linked only to its
own event). Every story was then checked by independent reviewers against its facts (contradictions, unnatural
or robotic wording, implied links not in the facts, missing facts, added content, ambiguous pronouns), fixed,
and re-checked until no story was flagged.

## 6. Perfect method

A method that outputs each story's true graph, the true groups, the true world graph and each story's told values
scores 1.00 on every metric (node F1, arrows, SID, KL; Table 1 and Table 2) at every usage cut from 5% to 20%.
