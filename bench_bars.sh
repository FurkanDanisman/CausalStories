#!/bin/bash
# One progress bar per Causal Stories run of a grid: 7 steps (nodes, chain walk, groups, names, did it happen,
# implied, connect); inside the chain walk and connect, the share of the 200 stories done so far.
#   GRID=grid_v10 bash bench_bars.sh
GRID=${GRID:-grid_v10}
bar() { local p=$1 n=$(( $1 / 5 )); printf "[%-20s] %3d%%" "$(printf '#%.0s' $(seq 1 $n) 2>/dev/null | head -c $n)" "$p"; }
for g in ${GENS:-gemma qwen llama70 gptoss}; do for w in ${WORLDS:-homelessness mental_health}; do
  d=$GRID/${w}__${g}__llm; [ -d "$d" ] || { printf "  %-8s %-14s not started\n" $g $w; continue; }
  done_n=0; now=""
  for s in nodes chain_walk groups names node-happened implied connect; do
    if [ -f "$d/.done_$s" ]; then done_n=$((done_n + 100)); continue; fi
    case $s in
      chain_walk) part=$(( 100 * $(ls "$d" | grep -c '__gemma-kev\.chain\.json$') / 200 ));;
      connect)    part=$(( 100 * $(ls "$d" | grep -c '__connect\.json$') / 200 ));;
      *)          part=0;;
    esac
    done_n=$((done_n + part)); now=$s; break
  done
  p=$(( done_n / 7 )); [ -z "$now" ] && now="done"
  printf "  %-8s %-14s %s  %s\n" $g $w "$(bar $p)" "$now"
done; done
