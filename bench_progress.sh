#!/bin/bash
# Progress of the full bench_run jobs, from the .done_<step> files each step leaves.
# Configurations (CONFIGS, run-folder suffix of our method): kev9b-llmgroup (Kev + LLM grouping; its jobs also run
# C1-C6's story graphs) and llm-llmgroup (no decision model). Our method = 7 steps (nodes, chain walk, groups, names,
# node-happened, implied, connect); C1-C6 = 1 step each (story graphs). Groups and connect are the long steps.
#   bash bench_progress.sh                     CONFIGS="kev9b" bash bench_progress.sh   (the v4 runs)
CONFIGS=${CONFIGS:-"kev9b-llmgroup llm-llmgroup"}
for cfg in $CONFIGS; do
  echo "== $cfg"; tot=0; all=0
  for g in gemma qwen llama gptoss; do for w in homelessness career mental_health; do
    d=grid_cluster/${w}__${g}__${cfg}; [ -d "$d" ] || continue
    n=$(ls -a "$d" | grep -c '^\.done_'); e=7
    for m in C1 C2 C3 C4 C5 C6; do                # competing approaches of the same job (if it runs them)
      c=grid_cluster/${w}__${g}__${m}-${cfg}; [ "$cfg" = kev9b ] && c=grid_cluster/${w}__${g}__${m}
      [ -d "$c" ] && { n=$(( n + $(ls -a "$c" | grep -c '^\.done_story_graphs') )); e=$(( e + 1 )); }
    done
    printf "  %-7s %-14s %3d%%  (%2d/%2d steps)\n" "$g" "$w" $(( 100 * n / e )) "$n" "$e"; tot=$(( tot + n )); all=$(( all + e ))
  done; done
  [ "$all" -gt 0 ] && printf "  overall %3d%%  (%d/%d steps)\n" $(( 100 * tot / all )) "$tot" "$all"
done
echo
echo "running now (last step started in each job's log):"
for j in $(squeue --me -h -n bench-run -t R -o %i); do
  f=bench-run-bench-run-$j.out
  [ -f "$f" ] || continue
  last=$(grep '^--- ' "$f" | tail -1 | sed 's/^--- //'); d=${last% *}; s=${last##* }
  extra=""; if [ "$s" = connect ] && [ -d "$d" ]; then   # inside a long step: stories connected so far
    extra="  ($(ls "$d" | grep -c '__connect\.json$') of $(ls "$d" | grep -c '__gemma-kev\.chain\.json$') stories connected)"; fi
  echo "  $j  $last$extra"
done
echo "waiting in the queue: $(squeue --me -h -n bench-run -t PD | wc -l)"

echo
echo "bench_final (aggregation comparison and judge):"
n_a4=$(ls -d grid_cluster/agg__A4-*/ 2>/dev/null | while read d; do [ -f "$d/network.json" ] && echo; done | wc -l)
n_agg=$(for m in A1 A2 A3 pc ges; do ls -d grid_cluster/agg__${m}__*/ 2>/dev/null; done | while read d; do [ -f "$d/network.json" ] && echo; done | wc -l)
n_c5=$(ls -d grid_cluster/agg__C5agg__*/ 2>/dev/null | while read d; do [ -f "$d/network.json" ] && echo; done | wc -l)
n_judge=0; n_judgeable=0
for R in grid_cluster/*/; do R=${R%/}; case $(basename "$R") in _*) continue;; esac
  [ -f "$R/abstraction.json" ] || continue
  if [ -f "$R/.done_groups" ] && [ ! -f "$R/.done_connect" ]; then continue; fi
  n_judgeable=$(( n_judgeable + 1 )); [ -f "$R/.judged" ] && n_judge=$(( n_judge + 1 ))
done
echo "  A4 (LLM merge):           $n_a4 of 12 (4 generators x 3 worlds)"
echo "  A1-A3, PC, GES:           $n_agg of 15 (5 methods x 3 worlds)"
echo "  C5 clustering:            $n_c5 of 12"
echo "  judged:                   $n_judge of $n_judgeable finished outputs"
for j in $(squeue --me -h -n bench-final -t R -o %i); do
  f=bench-final-$j.out; [ -f "$f" ] && echo "  running $j: $(grep '^##########\|^--- ' "$f" | tail -1)"
done
