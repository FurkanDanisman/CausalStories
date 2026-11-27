# Note: evaluating interventional distributions (planned, not started — 2026-11-26)

Step 9 of method_Oct.tex (truncated factorization on our causal Bayesian network) is specified but not implemented
or evaluated. Agreed plan, to do later:

- **Truth**: exact interventional values from the benchmark world (simulate with do(x), keep storytellers).
- **Population**: storytellers (interventions below do not change who tells a story; no do() on starting events).
- **Queries (fixed in advance)**: homelessness target: do(savings), do(family help), do(financial struggle),
  do(can't pay rent), each = 1 and = 0. Depression target: do(social support), do(therapy), do(loneliness).
  14 interventions, 7 effects.
- **Ours**: Step 9 on the network (grouping B, 10% cut), groups matched by the judge, opposite-side groups flipped.
- **Scores (main text)**: effect error |our effect - true effect| (effect = P(y|do(x=1)) - P(y|do(x=0))),
  sign accuracy over the 7 effects, coverage. Per-query probability errors in the appendix.
- **Reference rows**: perfect reader (Step 9 on the true graph and the stories' true values) and association
  (observational P(y|x=1) - P(y|x=0) from the same stories).
- **Generators**: Gemma, Qwen, Llama (8B; 70B if it passes its test). Laptop only: existing graphs, tables, judge.
- Issues to handle: missing parent settings (Step 9 fallback, marked; count in coverage); opposite naming.
