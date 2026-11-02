# Workplan: benchmark experiments on Killarney (2026-11-02)

All new runs use **cluster models only** (no OpenRouter). Every result is scored with the same two
tables (three_worlds.md §9) by the same judge, and every run records its wall time and hardware.

## 0. Preparation (laptop)

1. Fixes before any new run:
   - the grid stops on billing / network errors instead of recording empty runs as finished;
   - combine survives the numerical error (SVD did not converge) seen with Llama + Kev 4B.
2. Local backends: every LLM step runs on vLLM (generator, judge, baselines); Kev-27B served with
   `kev_serve_2gpu.py`.
3. Move the benchmark inputs into the repo (`pipeline/benchmark/`): stories_v3, world files,
   evaluate_run.py, so judging and scoring can run on the cluster.
4. Commit and push; on Killarney: `git pull`, then `sbatch` (one job per logical unit, ≤ 3 h, no job
   arrays, no dependencies).

Cluster models (from `/model-weights`): Gemma 4 31B, Qwen 3.5 27B, Llama 3.1 8B, Kev-27B.
Judge: **Gemma 4 31B** (local). Hardware for all timed runs: 2 GPUs per job (stated in run_info).

## 1. Our method with Kev-27B

| Generator | × worlds | Jobs |
|---|---|---|
| Gemma 4 31B, Qwen 3.5 27B, Llama 3.1 8B | homelessness, career, mental health | 9 (one per generator × world) |

## 2. Competing approaches (whole approaches vs ours)

Each uses Gemma 4 31B as its LLM (same model as our generator). One job per approach, 3 worlds inside.

| # | Approach | Individual graph | Aggregated graph |
|---|---|---|---|
| C1 | Naive LLM, one prompt per story (ReCITE-style; Saklad et al., ACL 2026) | LLM returns nodes + arrows in one answer | our aggregation |
| C2 | Whole-graph prompting (Long et al. 2023; Causal-LLM, EMNLP Findings 2025) | LLM gets story + its events, returns all arrows at once | our aggregation |
| C3 | Pairwise query prompting (Kıcıman et al. 2023) | "does A cause / block B?" for every pair | our aggregation |
| C4 | BFS query prompting (Jiralerspong et al. 2024) | from the outcome, "what caused this?" step by step | our aggregation |
| C5 | AI causal mapping, Causal Map workflow (Powell & Cabral 2025; Powell et al. 2025) | LLM codes every causal claim | their own: cluster labels, count links |
| C6 | Whole corpus in one prompt | – | long-context LLM reads all 200 stories, returns the world graph |
| C7 | PC / GES (classical causal discovery) | – | learned from the story × event table |

Scored on Table 1 (C1–C5) and Table 2 (all).

## 3. Aggregation comparison (same input: our individual graphs)

| # | Aggregation | How |
|---|---|---|
| A1 | Frequency counting / majority vote (Causal Map overview) | count stories per link, keep links above the usage cut |
| A2 | Fuzzy cognitive map averaging (Özesmi & Özesmi 2004) | average the story graphs as signed weighted matrices |
| A3 | Consensus (median) graph | graph with the smallest total SHD to all story graphs |
| A4 | LLM merge (Gemma 4 31B) | LLM merges the 200 story graphs into one |
| A5 | PC / GES | world graph from the story × event table only |
| – | Ours | grouping, did-it-happen, imputation, meta-analysis |

Input: our individual graphs from Gemma + Jev (OpenRouter run, already done) and Gemma + Kev-27B (step 1).
Probabilities for KL: methods without their own get conditional probabilities counted from the stories.
Scored on Table 2. A1–A3 and A5 need no GPU (laptop); A4 runs on the cluster.

## 4. Judging and scoring (cluster)

- One judging job: Gemma 4 31B judges every new run (steps 1–3) at all usage cuts, and re-judges the 12
  OpenRouter runs, so all results share one judge.
- Summary: `benchmark/grid_summary.py` extended with the competing approaches and the aggregation table.

## 5. Results

- Results page, Benchmark section, part 4: our method by decision model; competing approaches;
  aggregation comparison; each with score, time, hardware.
- Order: step 1 and step 2 jobs submitted together; step 3 after step 1 finishes; judging last.

## Job list (approximate)

| Step | Jobs |
|---|---|
| 1 Kev-27B | 9 |
| 2 Competing approaches C1–C6 (C7 on laptop) | 6 |
| 3 LLM merge A4 (A1–A3, A5 on laptop) | 1 |
| 4 Judging | 1–2 |
