# Benchmark worlds and stories (spec, 2026-11-23)

Code: `benchmark/worlds_random.py`. Stories: `benchmark/stories_v5/`. Earlier worlds (`world_homelessness.py`,
`world_mental.py`, `stories_v3`, `stories_v4`) are kept only for the earlier results.

## 1. The worlds

The original graphs (`benchmark/draft_graphs/`) with two removals, nothing added:
- no house fire (homelessness);
- savings blocks only *can't pay rent* (not homelessness), so every blocker blocks exactly one event.

No hidden variables, no distractors. Every node is 0 or 1.

**Homelessness**

| Root | Rate |
|---|---|
| EMPLOYER | 0.78 |
| FAMILY MEMBER | 0.45 |
| divorce | 0.60 |
| savings | 0.30 |

| Event | Causes (strength p) | Blockers (strength q) |
|---|---|---|
| job loss | EMPLOYER 0.98 | |
| providing a place to stay | FAMILY MEMBER 0.90 | |
| can't pay rent | job loss 0.95, divorce 0.60 | savings 0.80 |
| eviction | can't pay rent 0.65 | |
| homelessness | eviction 0.90 | providing a place to stay 0.95 |

**Mental health**

| Root | Rate |
|---|---|
| EMPLOYER | 0.50 |
| THERAPIST | 0.40 |
| CLOSE FRIENDS | 0.45 |
| losing a loved one | 0.80 |

| Event | Causes (strength p) | Blockers (strength q) |
|---|---|---|
| job loss | EMPLOYER 0.98 | |
| therapy | THERAPIST 1.00 | |
| social support | CLOSE FRIENDS 0.90 | |
| loneliness | losing a loved one 0.90 | social support 0.60 |
| chronic stress | job loss 0.72, loneliness 0.70 | |
| depression | chronic stress 0.72 | therapy 0.70 |

Arrow strengths are the original ones, except EMPLOYER → job loss in mental health (0.05 with EMPLOYER at
100% before; now 0.98 with EMPLOYER at 50%, as in homelessness) and THERAPIST → therapy (0.95 → 1.00). Root rates are raised from the real-world
rates so that a random person has events to tell.

**Mechanism** (noisy-OR with blockers): for each event, in the order above, each present cause fires with its
p and each present blocker fires with its q; the event happens if some cause fired and no blocker fired.
P(event | parents) = [1 − Π over present causes (1 − p)] × Π over present blockers (1 − q).

## 2. Who gets a story

People are drawn at random. A person to whom nothing happened (every node 0, about 3%) is drawn again.
Nothing else is selected. 200 people per world (seeds 800, 801).

## 3. What a story tells

From the person's true values:
1. every event that happened, and every participant who acted;
2. every cause that brought on an event that happened;
3. every block: a blocker that stopped a threatened event, with what threatened it; or a blocker that was
   there but did not stop it;
4. what did not happen, where it matters:
   - (a) a blocker of a happened or threatened event that was absent ("I had no savings");
   - (b) an event that a happened event can bring on, but which did not follow ("I was never evicted").
5. if a cause happened, the event did not follow, and a blocker of it was present, the story credits the
   blocker, as a reader would ("my sister took me in, so I never ended up on the street"). Only the story's
   truth changes; the person's values are the same.

Writing rules for the two delicate cases: a prevented event is told as "A happened; B did not happen,
because of C" (no "close to", no "could have"); an event with no arrow is told in its own sentence, clearly
separate from the rest, never next to something that did not happen.

Stories are written by Claude from these facts: first person, everyday words, nothing beyond the facts, no
other people, nothing about what would have happened (rules of `three_worlds.md` §8, plus rule 4).

## 4. Truth of a story

Every node the story tells, with its value (1 or 0), and the arrows that fired (`truth["nodes"]`,
`truth["arrows"]`). A node the story does not tell is not part of that story's truth.

## 5. Known consequences of the original graph

- A participant acts whenever present (with its strength), whether or not it was needed. A story can
  therefore mention help that nothing called for (e.g. "my sister offered me a place to stay").
