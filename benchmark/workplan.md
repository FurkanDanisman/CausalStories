# Workplan (updated 2026-11-02)

Status per item in %. Killarney runs **GPU stages only**; every CPU step (Steps 5-6: usage +
averaged matrix + intervals; scoring, PC/GES, A1–A3, Causal Map clustering) runs on the laptop after Killarney pushes
the results back. All new runs use cluster models only (no OpenRouter; that budget is used up).

**Overall benchmark: ~45%**

---

## 1. Benchmark setup — 95%

| Item | % | Notes |
|---|---|---|
| Three semi-synthetic worlds (smaller: 7–8 events each), calibrated to real rates | 100 | three_worlds.md §5–7; draft_graphs/ |
| Coverage check (whole graph survives the 10% cut in ≥95% of draws) | 100 | 100% / 100% / 99.9% |
| 600 stories (100 outcome + 100 stopped per world), human-sounding | 100 | stories_v3/ |
| Evaluation design: two score tables (individual, aggregated), 0–1 scores, geometric mean, run time | 100 | three_worlds.md §9 |
| KL of conditionals (true model scores 0) | 100 | |
| Judge (Qwen3 235B on OpenRouter; Gemma 4 31B on the cluster) | 100 | Llama dropped as judge |
| **Open decision:** score a method's own probabilities in KL when it has them (now every method's are recounted from its story table, so our meta-analysis estimates are never used) | 0 | found while testing A3 (consensus) vs ours |
| Results page: Benchmark section (worlds, evaluation, 20 stories, model results) | 80 | fills in as results come back |

## 2. Our method on OpenRouter (Gemma / Qwen / Llama × Jev / Kev 4B / GPT-6 Luna) — closed at 44%

| Item | % | Notes |
|---|---|---|
| 12 of 27 runs valid and scored | 44 | stopped: budget used up; 15 runs invalid/failed (in grid_invalid/). **Table 2 (aggregated) of these 12 is invalid**: they ran the old Oct 15 combine (MICE, no connect), not the agreed method; Table 1 (individual graphs) stands |
| Career + Qwen + Jev judged | 0 | runs in the cluster judge job |
| "Same LLM" decision runs | – | dropped (one partial run kept in grid_stopped/) |

## 3. Our method with Kev-27B on Killarney — 10%

| Item | % | Notes |
|---|---|---|
| Job script (bench_kev27b.sbatch), pushed on branch `benchmark-v3` | 100 | |
| Smoke job (Gemma, homelessness, SMOKE=1): check + measure Kev-27B speed | 0 | **next step**; 3-hour limit is the main risk |
| 9 jobs: 3 generators × 3 worlds (GPU part) | 0 | |
| Local: combine + scoring for the 9 runs | 0 | after `git pull` of results_cluster/ |

## 4. Competing approaches (whole approaches vs ours) — 15%

| # | Approach | Code % | Run % | Where |
|---|---|---|---|---|
| C1 | Naive LLM, one prompt per story (ReCITE) + our aggregation | 100 | 0 | Killarney (3 jobs) + local combine |
| C2 | Whole-graph prompting on our nodes + our aggregation | 100 | 0 | Killarney (3 jobs) + local combine |
| C3 | Pairwise query prompting (Kıcıman) + our aggregation | 100 | 0 | Killarney (3 jobs) + local combine |
| C4 | BFS query prompting (Jiralerspong) + our aggregation | 100 | 0 | Killarney (3 jobs) + local combine |
| C5 | AI causal mapping (Causal Map): claim coding + their clustering/counting | 100 | 0 | Killarney (1 job) + local clustering |
| C6 | Whole corpus in one prompt | 100 | 0 | Killarney (1 job) |
| C7 | PC / GES on the story × event table | 100 | 30 | local; done on the Gemma + Jev homelessness run |

Code tested against a mock LLM server only; the first real runs may need prompt/parsing fixes.

## 5. Aggregation comparison (same input: our story graphs) — 30%

| # | Aggregation | Code % | Run % | Notes |
|---|---|---|---|---|
| A1 | Frequency map (Causal Map overview) | 100 | 30 | homelessness Gemma + Jev: 0.71 (ours 0.78) |
| A2 | FCM averaging | 100 | 30 | 0.73 |
| A3 | Consensus (median) graph | 100 | 30 | **0.79 — slightly above ours**; see the KL decision in §1 |
| A4 | LLM merge | 100 | 0 | Killarney (in the judge job) |
| A5 | PC / GES | 100 | 30 | PC 0.41, GES 0.20 |
| | Remaining inputs: other worlds, Gemma + Kev-27B story graphs | | 0 | after §3 |

## 6. Judging and results — 25%

| Item | % | Notes |
|---|---|---|
| Judge job (GPU matching only; bench_judge.sbatch) | 100 code / 0 run | after §3–§5 |
| Local scoring of everything from the judge caches | 0 | |
| Results page: three tables (ours by decision model, competing approaches, aggregation) | 20 | |

## 7. Human annotation — 0%

| Item | % |
|---|---|
| Tasks and subtasks for annotators | 0 |
| Annotation guideline | 0 |
| Annotation server / tool | 0 |
| Pilot + agreement | 0 |

## 8. My to-do (theory / writing)

- [ ] Identifiability — formalize this — 0%
- [ ] Cyclic — state the assumptions — 0%
- [ ] Benchmarks — 45% (this file)
- [ ] Human annotation / tasks / subtasks / guideline / server — 0%
- [ ] Prepare the methodology — 0%
- [ ] Intervention feasibility — 0%

---

## How to run (Killarney, GPU stages)

```
git fetch && git checkout benchmark-v3 && git pull
sbatch --export=ALL,GEN=gemma,WORLD=homelessness,SMOKE=1 bench_kev27b.sbatch     # smoke first
sbatch --export=ALL,GEN=<gemma|qwen|llama>,WORLD=<homelessness|career|mental_health> bench_kev27b.sbatch
sbatch --export=ALL,METHOD=<C1|C2|C3|C4>,WORLD=<world> bench_baseline.sbatch    # C2-C4 after the Gemma Kev-27B run of that world
sbatch --export=ALL,METHOD=C5 bench_baseline.sbatch
sbatch --export=ALL,METHOD=C6 bench_baseline.sbatch
sbatch bench_judge.sbatch                                                        # last
bash bench_pack.sh && git add results_cluster && git commit -m "cluster results" && git push
```
Laptop: `git pull`, unpack results_cluster/, run Steps 5-6 (usage + averaged matrix), C5 clustering and
A1–A3/PC/GES (alt_aggregate.py), then scoring (evaluate_run.py) and grid_summary.py.
