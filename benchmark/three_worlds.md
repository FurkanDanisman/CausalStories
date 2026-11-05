# Benchmark worlds: homelessness, career success, mental health

Status (2026-10-29): smaller worlds; 600 stories written (`benchmark/stories_v3/`). The earlier,
larger worlds and their stories were removed; the benchmark uses only these worlds.

## 1. What this is

Three synthetic causal worlds. Each world generates people; for each world there are
**200 stories**: 100 people who reached the outcome and 100 who would have reached it but were
stopped by a named blocker. Every event in a story comes from the world; the writer only puts the
given events into words. Because we know each person's events and the world's graph, we know the
true individual graph of every story and the true aggregated graph.

**Goal:** a method that extracts perfectly recovers the whole true graph of each world from its
200 stories with a **10% usage cut**, in at least **95% of draws** of 200 stories; distractors are
mentioned in at most 5% of stories.

**Why the worlds are small:** with 11–13 events per world, a person carried 8–9 nodes on average
and up to 17, and stories with that many connections read like lists. With 7–8 events per world a
person carries about 4–7 nodes, so each story can be told the way a person would tell it.

Scripts: `benchmark/world_homelessness.py`, `benchmark/world_career.py`, `benchmark/world_mental.py`
(each prints its population rates, node shares and the check); `benchmark/story_v2.py` (true graph of
one person, sampling, the check: `python story_v2.py check`).
Graphs: `benchmark/draft_graphs/` (`plot_drafts.py` draws them).

## 2. How a world generates people

- **Events and participants are binary.** Participants (UPPERCASE) carry out events
  (participant → event arrow). EMPLOYER (mental health) and COMPANY (career) are present for
  everyone; other participants are present at the stated rate.
- **Independent causes with blockers** (noisy-OR with inhibitors; Pearl, 1988). For an event
  with enablers *e* (each with chance *p*) and blockers *b* (each with strength *q*):

      P(event | parents) = [1 − Π over present enablers (1 − p)] × Π over present blockers (1 − q)

  Each enabler is a separate chance to cause the event; a blocker cuts that chance by its
  strength and causes nothing on its own.
- **Closed world (no leak):** an event happens only through one of its named causes, so every
  person in the data has a nameable reason.
- **No dependence without an arrow:** there are no hidden common causes.
- **Distractors** are events with no arrows, at a 4% rate (below the 5% cap).
- Each arrow is drawn as its own random chance for each person, so each person's
  counterfactuals are defined.

## 3. Which people become stories

1. Simulate people from the world.
2. **Outcome group:** people who reached the outcome.
3. **Stopped group:** people who did not reach it, but would have if every blocker event had
   been switched off (same random draws), and in whose story a blocker stopped the outcome itself.
4. Keep only people whose events all connect to the outcome; draw 100 from each group
   (`story_v2.sample_v2`, seeds 700, 701, 702). Each person becomes one story.

Many people share the same graph in small worlds (distinct fact sets among the 200: homelessness
24, career 40, mental health 11); each still has their own story text.

## 4. The true graph of one story, and the check

The true graph of a story contains only what that person lived (`story_v2.lived_graph`):

- A → B (enables / carries out): A and B both happened;
- C ⊣ B (stopped): C happened and B was threatened (would have happened with every blocker off)
  and did not happen; C ⊣ B (failed): C happened, B was threatened and happened anyway;
- A → B (risk), one step: A happened and put the person at risk of a stopped B
  (career: "on track for");
- a blocker event, and the participant behind it, is in the story only when it stood in the way
  of something that was about to happen.

**Check:** draw 200 stories (100 + 100) 1,000 times; the world passes if every node of the true
graph is in at least 10% of the stories in at least 95% of draws, and no distractor reaches 10%.

## 5. Homelessness

Graph: `draft_graphs/1_homelessness.png`. 8 events, 2 participants.

| Node | Rate per year | Source / note |
|---|---|---|
| EMPLOYER (layoff) | 4.7% | BLS ~1.8%; raised |
| FAMILY MEMBER (provides a place to stay) | 80% | assumed |
| divorce | 10.9% | assumed; carried over from the larger world, well above the real yearly rate |
| house fire | 4.7% | NFPA ~0.3%; raised |
| savings | 63% | Federal Reserve SHED 2024 (63% could cover $400 in cash) |
| getting a pet, taking up running | 4% each | distractors |

| Event | Enablers | Blockers |
|---|---|---|
| job loss | EMPLOYER 0.98 | |
| providing a place to stay | FAMILY MEMBER 0.9 | |
| can't pay rent | job loss 0.95, divorce 0.6 | savings 0.80 |
| eviction | can't pay rent 0.65 | |
| homelessness | eviction 0.9, house fire 0.4 | providing a place to stay 0.95, savings 0.95 |

| Figure | World | Real | Source |
|---|---|---|---|
| homelessness per year | 0.96% | ~0.4–0.6% (sheltered only, a lower bound) | HUD AHAR Part 2, 2022 |
| eviction per year | 3.43% | ~2.3–3% of renter households | Eviction Lab |
| homeless given evicted | 0.22 | ~0.22 | Durham, NC public housing study |

**Check:** whole graph survives the 10% cut in 100% of draws. Lowest shares: house fire 25%,
EMPLOYER and job loss 32.5%, savings 37%. Distractors 4%.

## 6. Career success

Graph: `draft_graphs/2_career_success.png`. 8 events, 2 participants.

| Node | Rate per year | Source / note |
|---|---|---|
| COMPANY | present for everyone | everyone has an employer |
| MENTOR | 30% | ~26–38% have a mentor (Accountemps 2016; Environics 2025) |
| degree | 38.6% | Census 2024 (adults 25+) |
| facing discrimination | 19% | ADP Global Workforce Survey 2025 |
| moving apartment, buying a car | 4% each | distractors |

| Event | Enablers | Blockers |
|---|---|---|
| receiving guidance | MENTOR 0.95 | |
| gaining skills | degree 0.5, receiving guidance 0.4 | |
| getting noticed | receiving guidance 0.4 | |
| layoffs | COMPANY 0.12 | |
| strong performance | gaining skills 0.6 | |
| promotion | strong performance 0.56, getting noticed 0.2 | facing discrimination 0.7, layoffs 0.8 |

strong performance → promotion was raised from 0.25 to 0.56 after networking, training, working
long hours and the manager route were removed, to keep promotion at ~9% a year.

| Figure | World | Real | Source |
|---|---|---|---|
| promotion per year | 9.0% | ~9% | ADP State of the Workforce 2019 |
| degree | 38.6% | 38.6% | Census 2024 |
| discrimination | 19% | ~19% | ADP 2025 |
| layoffs | ~12% of workers per year | 1.1% of employment per month | BLS JOLTS 2024 |

**Check:** whole graph survives the 10% cut in 100% of draws. Lowest shares: COMPANY and layoffs
24%, getting noticed 34%, facing discrimination 35%. Distractors 4%.

## 7. Mental health (outcome: depression)

Graph: `draft_graphs/3_mental_health.png`. 7 events, 3 participants.

| Node | Rate per year | Source / note |
|---|---|---|
| EMPLOYER | present for everyone | |
| THERAPIST (provides therapy) | 25% | about 23% of adults received mental-health treatment (NSDUH 2023) |
| CLOSE FRIENDS (give social support) | 60% | assumed |
| losing a loved one | 25% | ~45% over two years (Georgia BRFSS 2019) |
| new phone, learning to cook | 4% each | distractors |

| Event | Enablers | Blockers |
|---|---|---|
| job loss | EMPLOYER 0.05 | |
| therapy | THERAPIST 0.95 | |
| social support | CLOSE FRIENDS 0.9 | |
| loneliness | losing a loved one 0.9 | social support 0.6 |
| chronic stress | job loss 0.72, loneliness 0.7 | |
| depression | chronic stress 0.72 | therapy 0.7 |

job loss → chronic stress is 0.72 (the earlier job loss → financial stress → chronic stress,
0.9 × 0.8). chronic stress → depression was raised from 0.6 to 0.72 after insomnia was removed, to
keep depression at 8.3%.

| Figure | World | Real | Source |
|---|---|---|---|
| depression (major depressive episode) | 8.35% | 8.3% | NIMH, 2021 |
| daily loneliness | 15.2% | ~20% | Gallup 2024 |
| mental-health treatment | 23.7% | 23.0% | NSDUH 2023 |

**Check:** whole graph survives the 10% cut in 99.9% of draws. Lowest shares: EMPLOYER and job
loss 18%, THERAPIST and therapy 28%. Distractors 4%.

## 8. Writing the stories

The 600 stories in `benchmark/stories_v3/<world>.json` were written by Claude from each person's
facts (fields: `id`, `world`, `group`, `truth` {arrows, distractors}, `facts`, `text`). Rules:

- first person, the way an ordinary person would tell it, in everyday words rather than node
  names ("my employer let me go", "people at work knew my name", "I was stressed all the time");
  no limit on length; small details that are not events (time, feelings) are allowed;
- every connection in the facts must be recoverable; nothing beyond the facts: no other event,
  cause, consequence or person (no landlord, parent, manager, ...);
- only lived facts: no claims about what would have happened; risk is told as "I was close to …",
  "I was in line for …";
- stopped things clearly did not happen; side events are mentioned but tied to nothing.

## 9. Evaluation metrics

An estimated causal graph (nodes, arrows with enable/block sign, one probability per arrow) is
compared with the true world graph:

| Metric | What it checks | Perfect |
|---|---|---|
| **Node precision / recall** | share of estimated nodes that are real / share of real nodes found | 1 / 1 |
| **SHD** (Tsamardinos, Brown & Aliferis, 2006), with sign errors | arrow edits to turn the estimate into the truth: missing, extra, reversed, wrong sign | 0 |
| **SID** (Peters & Bühlmann, 2015) | node pairs (A, B) whose P(B \| do(A)) would be inferred wrongly from the estimated graph | 0 |
| **KL of conditionals** (true ‖ estimated) | how far the method's P(child \| parent) is from the stories' true one, for every true arrow | 0 |

- **Node matching first**, by a separate LLM judge (not by hand): it checks whether an extracted
  node means the same as a true event; if so it is renamed to the true name, otherwise it stays an
  extra node. Individual stories are matched against that story's own events; aggregated graphs
  against the world's nodes.
- **SHD:** one error per node pair; arrows touching extra nodes count as extra arrows.
- **SID** (common node set): extra nodes are contracted (parents joined to children); cycles are
  broken by dropping the weakest arrow.
- **KL of conditionals (per arrow):** what the methods output is one conditional probability per arrow,
  P(child | parent). For every true arrow u → v, the true P(v | u) is counted from the **true story table**
  (each story's true events, from the generator) and compared with the method's P(v | u) for its arrow u → v,
  counted the same way from **its own story table** (Step 8 of the method, per arrow); KL is the mean of the
  Bernoulli KL over the true arrows. A true arrow the method does not have, or a node it misses, gets 0.5; a
  method without a story table uses its arrow's own number. The reference is these 200 stories, not the
  population, so how the stories were sampled (half reach the outcome) does not count against any method, and
  a method with the true graph that reads every story right scores exactly 0. Joint probabilities over several
  causes are not scored: a story only reports the events of its own chain, so almost no story knows all the
  causes of a node with several.
- Not computed yet: node matching is not built.

### Summary scores (0–1, 1 is perfect)

Individual graphs and aggregated graphs are scored separately. An individual graph is one person's
story: it has no probabilities (so no KL) and is not a population model (so no SID); it is scored on
structure only. Each metric is put on a 0–1 scale and the metrics are combined with the geometric
mean, so being bad at one thing cannot be hidden by the others (a 0 on any metric gives 0).

**Table 1: individual graphs** (one score per story, then the mean over the world's 200 stories)

| Metric | 0–1 version |
|---|---|
| Nodes | F1 of node precision and recall |
| Arrows (SHD, with sign) | 1 − SHD / (true arrows + estimated arrows); a wrong arrow type (enables / stopped / failed / risk) counts as one SHD error |
| **Story score** | geometric mean of the two |
| **World score** | mean of the story scores over the 200 stories |

**Table 2: aggregated graph** (one per world)

| Metric | 0–1 version |
|---|---|
| Nodes | F1 of node precision and recall |
| Arrows (SHD, with sign) | 1 − SHD / (true arrows + estimated arrows) |
| SID | 1 − wrong pairs / all ordered node pairs |
| KL of conditionals | e^(−KL) (1 when KL = 0, toward 0 as the conditionals diverge) |
| **World score** | geometric mean of the four |

**Benchmark:** one row per table, with a column per world and the mean over the three worlds:

| | Homelessness | Career | Mental health | Benchmark (mean) |
|---|---|---|---|---|
| Individual graphs | world score | world score | world score | mean |
| Aggregated graph | world score | world score | world score | mean |

**Run time.** Every run also reports the wall-clock time of the whole pipeline on the 200 stories
of a world (all stages: node extraction, arrows, groups, names, did-it-happen, combine; node
matching and scoring are not counted), with the hardware stated (GPU type and count). A model
combination is described by its score and its time.

**Synthetic example (made up for illustration; metric values computed on it).** Suppose the method
returned, for homelessness, an aggregated graph where the judge matched 9 of 10 true nodes (FAMILY
MEMBER missed, merged into "providing a place to stay"), added one extra node ("financial stress",
from "money worries"), missed FAMILY MEMBER → providing a place to stay and savings ⊣ homelessness,
and added divorce → eviction plus job loss → financial stress → can't pay rent. For one stopped
story (ho132: EMPLOYER → job loss → can't pay rent → eviction ⇢ homelessness, FAMILY MEMBER →
providing a place to stay ⊣ homelessness), it found all 7 nodes plus one extra, 5 of 6 arrows (missed
the risk arrow), and one extra arrow.

| Individual graph (ho132) | Value |
|---|---|
| Nodes: F1 (P 7/8, R 7/7) | 0.93 |
| Arrows: 1 − 2 / (6 + 6) | 0.83 |
| **Story score** √(0.93 × 0.83) | **0.88** |

| Aggregated graph (homelessness) | Value |
|---|---|
| Nodes: F1 (P 9/10, R 9/10) | 0.90 |
| Arrows: 1 − 5 / (10 + 11) | 0.76 |
| SID: 1 − 6/90 | 0.93 |
| KL: e^(−0.60) (full-joint KL, computed before the conditionals-only rule) | 0.55 |
| **World score** (geometric mean) | **0.77** |

In the example, missing savings ⊣ homelessness costs one SHD point but five SID pairs (savings becomes
a hidden common cause of homelessness and its other causes), and merging FAMILY MEMBER into its event
counts against nodes, SID and KL, not only SHD.

## 10. Assumptions to state

- Synthetic worlds; rates are US statistics where available, the rest assumed.
- Noisy-OR with inhibitors, a closed world and no hidden common causes are modelling choices.
- Some causes are raised above real rates (layoff, house fire, divorce) so that every route is
  recoverable from 200 stories; arrow chances and blocker strengths are tuned to the population
  figures above, not measured.
- Stories are selected on the outcome (100 + 100), so within the 200 stories, independent causes
  can look related; the answer key is the world graph, not correlations in the sample.
- Node names are everyday wording in the stories; recovering them is the node matcher's job.

## Sources

- HUD AHAR 2022 sheltered estimates: https://abtglobal.com/insights/publications/report/2022-annual-estimates-of-sheltered-homelessness-in-the-united-states
- Eviction Lab national estimates: https://evictionlab.org/national-estimates/
- Eviction and homelessness, Durham public housing: https://pmc.ncbi.nlm.nih.gov/articles/PMC9279006
- NFPA home structure fires: https://bcn.boulder.co.us/emergency/fourmile/HomeStructureFires.pdf
- BLS Displaced Workers Summary: https://www.bls.gov/news.release/disp.nr0.htm
- Federal Reserve SHED 2024: https://www.federalreserve.gov/publications/2025-economic-well-being-of-us-households-in-2024-executive-summary.htm
- ADP State of the Workforce 2019 (promotions): https://mediacenter.adp.com/2019-04-16-ADP-Research-Institute-Releases-2019-State-of-the-Workforce-Report-Pay-Promotions-and-Retention-Providing-Unprecedented-View-of-the-Organization-of-U-S-Firms
- ADP 2025 discrimination: https://adpresearch.com/wp-content/uploads/2025/02/PAW2025_Discrimination-Final.pdf
- Accountemps mentors (2016): https://www.benefitspro.com/2016/01/15/few-employees-have-work-mentors?amp=1
- Environics / Future Skills Centre mentorship (2025): https://fsc-ccf.ca/wp-content/uploads/2025/04/mentorship-in-the-workplace-employee-participation-and-perspectives-1.pdf
- BLS JOLTS 2024 annual rates: https://www.bls.gov/news.release/archives/jolts_03112025.htm
- Census educational attainment 2024: https://www.census.gov/newsroom/press-releases/2025/educational-attainment-data.html
- NIMH / NSDUH depression: https://www.samhsa.gov:443/data/sites/default/files/reports/rpt42731/2022-nsduh-main-highlights.pdf
- Gallup daily loneliness 2024: https://news.gallup.com/poll/651881/daily-loneliness-afflicts-one-five.aspx
- NSDUH 2023 treatment: https://bhbusiness.com/2024/08/13/nearly-1-in-4-u-s-adults-sought-mental-health-treatment-in-past-year/
- Bereavement prevalence, Georgia BRFSS 2019: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10262108/
