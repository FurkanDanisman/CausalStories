#!/bin/bash
# Progress of the 12 full bench_run jobs (4 generators x 3 worlds), from the .done_<step> files each step leaves.
# One job = 33 steps: ours 7 (nodes, chain walk, groups, names, node-happened, implied, connect), C1-C4 6 each
# (story graphs + the same aggregation steps), C5 and C6 1 each (story graphs). Steps differ in length: groups and
# connect are the long ones, so the step share runs a little ahead of the time share.
#   bash bench_progress.sh
DEC=${DEC:-kev9b}; tot=0
for g in gemma qwen llama gptoss; do for w in homelessness career mental_health; do
  n=0
  for m in $DEC C1 C2 C3 C4 C5 C6; do
    d=grid_cluster/${w}__${g}__$m
    [ -d "$d" ] && n=$(( n + $(ls -a "$d" | grep -c '^\.done_') ))
  done
  printf "%-7s %-14s %3d%%  (%2d/33 steps)\n" "$g" "$w" $(( 100 * n / 33 )) "$n"; tot=$(( tot + n ))
done; done
printf "overall           %3d%%  (%d/396 steps)\n\n" $(( 100 * tot / 396 )) "$tot"
echo "running now (last step started in each job's log):"
for j in $(squeue --me -h -n bench-run -t R -o %i); do
  f=bench-run-bench-run-$j.out
  [ -f "$f" ] && echo "  $j  $(grep '^--- ' "$f" | tail -1 | sed 's/^--- //')"
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
  if ls -a "$R" | grep -q '^\.done_' && [ ! -f "$R/.done_connect" ]; then continue; fi
  n_judgeable=$(( n_judgeable + 1 )); [ -f "$R/.judged" ] && n_judge=$(( n_judge + 1 ))
done
echo "  A4 (LLM merge):           $n_a4 of 12 (4 generators x 3 worlds)"
echo "  A1-A3, PC, GES:           $n_agg of 15 (5 methods x 3 worlds)"
echo "  C5 clustering:            $n_c5 of 12"
echo "  judged:                   $n_judge of $n_judgeable finished outputs"
for j in $(squeue --me -h -n bench-final -t R -o %i); do
  f=bench-final-$j.out; [ -f "$f" ] && echo "  running $j: $(grep '^##########\|^--- ' "$f" | tail -1)"
done
