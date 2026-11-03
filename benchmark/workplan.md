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
| Table 1 (individual graphs) scored | 37 | 10 of the 27 planned runs (homelessness 5, career 2, mental health 3); career + Qwen + Jev ran but is not judged; homelessness + Gemma + Kev 4B was deleted by mistake on 2026-11-02; the other 15 never finished (budget, network) |
| Table 2 (aggregated) | **0** | no valid result: all runs used the old combine (MICE, no connect) |
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

Generators (every method that needs an LLM): GEN = gemma (Gemma 4 31B) | qwen (Qwen3.5-27B) | llama (Llama-3.1-8B) |
gptoss (gpt-oss-20b). Decision models: DEC = kev27b (default) | kev9b. The judge stays Gemma 4 31B.

**1. Smoke test (one job, all pieces on 2 stories):** `sbatch smoke_all.sbatch` -> `grid_cluster/_smoke/summary.tsv`
(every LLM step x 4 generators; every Kev step x 2 decision models, on our story graphs and C1's). All rows must say pass.

**2. Full runs, after the smoke test passes: 9 jobs**, each loading every model once per phase and covering all
three worlds (resumable: resubmitting a job skips finished steps):
```
git pull
sbatch --export=ALL,GEN=gemma,DEC=kev27b  bench_run.sbatch   # ours + C1-C6, all worlds
sbatch --export=ALL,GEN=qwen,DEC=kev27b   bench_run.sbatch
sbatch --export=ALL,GEN=llama,DEC=kev27b  bench_run.sbatch
sbatch --export=ALL,GEN=gptoss,DEC=kev27b bench_run.sbatch
sbatch --export=ALL,GEN=gemma,DEC=kev9b   bench_run.sbatch   # ours only, all worlds
sbatch --export=ALL,GEN=qwen,DEC=kev9b    bench_run.sbatch
sbatch --export=ALL,GEN=llama,DEC=kev9b   bench_run.sbatch
sbatch --export=ALL,GEN=gptoss,DEC=kev9b  bench_run.sbatch
# after all 8 have finished:
sbatch bench_final.sbatch      # A4 x 4 generators, A1-A3, A5 (PC/GES), C5 clustering, judge (Gemma)
bash bench_pack.sh && git add results_cluster && git commit -m "cluster results" && git push
```
Laptop: `git pull`, unpack results_cluster/, run Steps 5-6 (`run.py --mode matrix --graphs-glob '<run>/*__gemma-kev.chain.json' --outdir <run>`)
for our runs and C1-C4, then scoring (evaluate_run.py, from the judge caches) and grid_summary.py.
