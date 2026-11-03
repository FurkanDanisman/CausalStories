#!/bin/bash
# On Killarney, after the benchmark jobs: pack every finished cluster run compactly into results_cluster/
# (run info, judge scores, groups, story graphs, did-it-happen and connect files, the world graph), then
# commit and push:   bash bench_pack.sh && git add results_cluster && git commit -m "cluster results" && git push
set -euo pipefail
source .venv/bin/activate
mkdir -p results_cluster
for R in grid_cluster/*/; do
  R=${R%/}; name=$(basename "$R")
  [ -f "$R/run_info.json" ] || [ -f "$R/network.json" ] || continue
  tmp=$(mktemp -d); mkdir -p "$tmp/$name"
  cp "$R"/*.json "$tmp/$name/" 2>/dev/null || true
  tar -czf "results_cluster/$name.tar.gz" -C "$tmp" "$name"; rm -rf "$tmp"
  echo "packed $name"
done
