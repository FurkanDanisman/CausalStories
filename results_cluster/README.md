# Results

- `results_cluster/<grid>/*.tar.gz` — one pack per cluster run (method output only, independent of any judge):
  story graphs (`*.chain.json`), did-it-happen and connect answers, groups and names (`abstraction.json`,
  `implied.json`), extracted nodes, timings and token logs (`timing.tsv`, `inference.jsonl`), and the judge's cached
  answers (`judge_cache_<judge>.json`).
- `results_cluster/scored/<grid>__<judge>.tar.gz` — the scores of every run with that judge, as reported:
  `network.json` (Steps 5-8, computed locally; recomputable from the packs with `run.py --mode matrix`),
  `evaluation_<judge>_cutXX.json` (and `_flip_` analyses), `score.log`, and the judge cache.

Grids: grid_v7 (stories_v7, first prompt), grid_v7b (stories_v7, direct-only arrow prompt), grid_v8 (stories_v8;
Gemma, Qwen, Llama-3.1-8B), grid_v8c (stories_v8, Gemma, grouping with the new "opposite" question).
A new judge writes new files (`judge_cache_<new judge>.json`, `evaluation_<new judge>_*.json`); nothing of the
Gemma 4 31B judge is overwritten.
