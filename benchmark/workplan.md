# Workplan (updated 2026-11-02)

Status per item in %. Killarney runs **GPU stages only**; every CPU step (Steps 5-6 = `run.py --mode matrix`; scoring, PC/GES, A1–A3, Causal Map clustering) runs on the laptop after Killarney pushes
the results back. All new runs use cluster models only (no OpenRouter; that budget is used up).

**Overall benchmark: ~25%** (reset on 2026-11-02: the aggregated results came from the old Oct 15 combine,
with MICE and no connect step, not from the agreed method. Everything aggregated is back to 0% until it is rerun.)

---

## 0. Pipeline = the agreed method only — 100%

| Item | % | Notes |
|---|---|---|
| Repo cleaned: only the agreed method (method_Oct.tex, Oct 23) is left | 100 | old TORQUE / LLM-merge / MICE / Bayesian-network code removed |
| Section 2 Steps 3-4 in the runners (node-happened, implied, connect) | 100 | run_api_or.sh, bench_lib.sh |
| Steps 5-6 in the pipeline (`run.py --mode matrix`, pipeline/matrix.py) | 100 | checked identical to the agreed report on out_api_world100 (cells, 56 arrows) |
| Scorer: agreed usage rule; conditional probabilities only over stories where both are known | 100 code / 0 tested | needs a connected benchmark run |

## 1. Benchmark setup — 95%

| Item | % | Notes |
|---|---|---|
| Three semi-synthetic worlds (smaller: 7–8 events each), calibrated to real rates | 100 | three_worlds.md §5–7; draft_graphs/ |
| Coverage check (whole graph survives the 10% cut in ≥95% of draws) | 100 | 100% / 100% / 99.9% |
| 600 stories (100 outcome + 100 stopped per world), human-sounding | 100 | stories_v3/ |
| Evaluation design: two score tables (individual, aggregated), 0–1 scores, geometric mean, run time | 100 | three_worlds.md §9 |
| KL of conditionals (true model scores 0) | 100 | |
| Judge (Qwen3 235B on OpenRouter; Gemma 4 31B on the cluster) | 100 | Llama dropped as judge |
| **Open decision:** score a method's own probabilities in KL when it has them (now every method is recounted from its story table) | 0 | |
| Results page: Benchmark section | 40 | aggregated numbers on it are from the old pipeline and must be replaced |

## 2. Our method on OpenRouter (Gemma / Qwen / Llama × Jev / Kev 4B / GPT-6 Luna)

| Item | % | Notes |
|---|---|---|
| Table 1 (individual graphs), 12 of 27 runs | 44 | **valid**: Steps 1-2 are the agreed method; 15 runs failed (budget, network) |
| Table 2 (aggregated), 12 runs | **0** | old combine (MICE, no connect); rerun needs node-happened + implied + connect (decision-model calls) + matrix |
| "Same LLM" decision runs | – | dropped |

## 3. Our method with Kev-27B on Killarney — 5%

| Item | % | Notes |
|---|---|---|
| Job script with the agreed Section 2 (connect added) | 100 | pushed on `benchmark-v3`; pull before submitting |
| Smoke job (Gemma, homelessness, SMOKE=1): check + speed | 0 | **next step**; connect adds requests, so the 3-hour limit is the main risk |
| 9 jobs: 3 generators × 3 worlds (GPU part) | 0 | |
| Local: matrix + scoring for the 9 runs | 0 | |

## 4. Competing approaches (whole approaches vs ours) — 10%

| # | Approach | Code % | Run % | Where |
|---|---|---|---|---|
| C1 | Naive LLM, one prompt per story (ReCITE) + our aggregation | 100 | 0 | Killarney (3 jobs) + local matrix |
| C2 | Whole-graph prompting on our nodes + our aggregation | 100 | 0 | Killarney (3 jobs) + local matrix |
| C3 | Pairwise query prompting (Kıcıman) + our aggregation | 100 | 0 | Killarney (3 jobs) + local matrix |
| C4 | BFS query prompting (Jiralerspong) + our aggregation | 100 | 0 | Killarney (3 jobs) + local matrix |
| C5 | AI causal mapping (Causal Map): claim coding + their clustering/counting | 100 | 0 | Killarney (1 job) + local clustering |
| C6 | Whole corpus in one prompt | 100 | 0 | Killarney (1 job) |
| C7 | PC / GES on the story × event table | 100 | 0 | local; earlier numbers compared against the old aggregation |

Code tested against a mock LLM server only; the first real runs may need prompt/parsing fixes.

## 5. Aggregation comparison (same input: our story graphs) — 10%

| # | Aggregation | Code % | Run % | Notes |
|---|---|---|---|---|
| A1 | Frequency map (Causal Map overview) | 100 | 0 | its own number (0.71) stands, but "ours 0.78" was the old pipeline |
| A2 | FCM averaging | 100 | 0 | 0.73 (own number stands) |
| A3 | Consensus (median) graph | 100 | 0 | 0.79 (own number stands); compare again with the agreed aggregation |
| A4 | LLM merge | 100 | 0 | Killarney (in the judge job) |
| A5 | PC / GES | 100 | 0 | PC 0.41, GES 0.20 (own numbers stand) |

## 6. Judging and results — 0%

| Item | % | Notes |
|---|---|---|
| Judge job (GPU matching only; bench_judge.sbatch) | 100 code / 0 run | after §3–§5 |
| Local scoring of everything from the judge caches | 0 | |
| Results page: three tables (ours by decision model, competing approaches, aggregation) | 0 | |

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
- [ ] Benchmarks — 25% (this file)
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
Laptop: `git pull`, unpack results_cluster/, run Steps 5-6 (`run.py --mode matrix --graphs-glob '<run>/*__gemma-kev.chain.json' --outdir <run>`), C5 clustering and
A1–A3/PC/GES (alt_aggregate.py), then scoring (evaluate_run.py) and grid_summary.py.
