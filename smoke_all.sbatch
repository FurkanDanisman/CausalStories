#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=smoke-all
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=smoke-all-%j.out
#
# One smoke test for every GPU piece of the benchmark, on the first 2 homelessness stories:
#   Part A, each generator (gemma qwen llama gptoss): our node extraction, C1-C6, group naming, A4 (LLM merge)
#   Part B, each decision model (kev27b kev9b): our chain walk, then our aggregation (groups, did it happen,
#           connect) on our story graphs and on C1's story graphs
#   Part C, the final job's pieces: A1-A3, A5 (PC / GES), C5 clustering, and the judge (Gemma) on one run
# A step passes when it exits 0 AND no story failed (a story that fails is otherwise kept as an empty graph).
# Writes grid_cluster/_smoke/summary.tsv (step, model, pass/FAIL, seconds, stories ok / total).
#   sbatch smoke_all.sbatch
set -uo pipefail
S=grid_cluster/_smoke
OUT=$S
rm -rf "$S"; mkdir -p "$S"
source bench_lib.sh
python -c "import json;json.dump(json.load(open('homelessness_v3.json'))[:2],open('$S/texts.json','w'))"
[ -d grid/homelessness__gemma__jev ] || tar -xzf grid_seed.tar.gz grid/homelessness__gemma__jev   # input for naming + A4
printf "step\tmodel\tresult\tseconds\tstories_ok\n" > "$S/summary.tsv"

check() {                    # check <step> <model> <rc> <t0> <dir or file> : count failed stories, write one row
  python - "$@" "$S/summary.tsv" <<'PY'
import json, glob, os, sys, time
step, model, rc, t0, target, tsv = sys.argv[1:7]
ok = tot = 0
if os.path.isdir(target):
    for f in glob.glob(f"{target}/*__gemma-kev.chain.json"):
        g = json.load(open(f)); tot += 1; ok += not (g.get("failed") or (g.get("raw") or {}).get("failed"))
    if os.path.exists(f"{target}/chain_nodes.json"):
        it = json.load(open(f"{target}/chain_nodes.json")); tot = len(it); ok = sum(1 for x in it if x["nodes"] and not x["raw"]["failed"])
elif os.path.exists(target):
    ok = tot = 1
res = "pass" if rc == "0" and tot and ok == tot else "FAIL"
open(tsv, "a").write(f"{step}\t{model}\t{res}\t{int(time.time()) - int(t0)}\t{ok}/{tot}\n")
print(f"=== {step} [{model}]: {res} ({ok}/{tot})", flush=True)
PY
}

echo "############ Part A: generators"
for G in gemma qwen llama gptoss; do
  D=$S/$G; mkdir -p "$D"
  start_llm "$G" 65536
  t0=$(date +%s); python run.py --mode chain-nodes --backend openai --base-url "http://127.0.0.1:$LLM_PORT/v1" --model llm \
    --texts-file "$S/texts.json" --max-nodes 8 --outdir "$D/ours" --workers 16; check nodes "$G" $? $t0 "$D/ours"
  for M in C1 C2 C3 C4 C5 C6; do
    t0=$(date +%s); python benchmark/baselines.py $M "$S/texts.json" "$D/$M" --nodes "$D/ours/chain_nodes.json" --workers 16
    rc=$?; [ $M = C6 ] && check $M "$G" $rc $t0 "$D/$M/network.json" || check $M "$G" $rc $t0 "$D/$M"
  done
  mkdir -p "$D/names"; cp grid/homelessness__gemma__jev/abstraction.json "$D/names/"
  t0=$(date +%s); python run.py --mode name-groups --backend openai --base-url "http://127.0.0.1:$LLM_PORT/v1" --model llm \
    --outdir "$D/names"; check name-groups "$G" $? $t0 "$D/names/abstraction.json"
  t0=$(date +%s); python benchmark/alt_aggregate.py A4 grid/homelessness__gemma__jev "$D/A4"; check A4 "$G" $? $t0 "$D/A4/network.json"
  stop_llm
done

echo "############ Part B: decision models"
for K in kev27b kev9b; do
  DEC=$K; D=$S/$K; mkdir -p "$D"
  start_kev
  t0=$(date +%s); python run.py --mode chain-kev --kev-url "http://127.0.0.1:$KEV_PORT" --nodes-file "$S/gemma/ours/chain_nodes.json" \
    --tag gemma-kev --outdir "$D/ours"; check chain-walk "$K" $? $t0 "$D/ours"
  mkdir -p "$D/C1"; cp "$S"/gemma/C1/*.chain.json "$D/C1/" 2>/dev/null
  for R in ours C1; do                     # group names are not needed to test these steps (tested in Part A)
    for M in merge-kev node-happened implied connect; do
      t0=$(date +%s); python run.py --mode $M --kev-url "http://127.0.0.1:$KEV_PORT" --graphs-glob "$D/$R/*__gemma-kev.chain.json" \
        --outdir "$D/$R"; rc=$?
      f=$( [ $M = merge-kev ] && echo abstraction.json || { [ $M = implied ] && echo implied.json || echo .; } )
      if [ "$f" = . ]; then n=$(ls "$D/$R"/*__$( [ $M = connect ] && echo connect || echo happened ).json 2>/dev/null | wc -l | tr -d " ")
        printf "%s\t%s\t%s\t%s\t%s\n" "$M ($R)" "$K" "$( [ $rc = 0 ] && [ "$n" = 2 ] && echo pass || echo FAIL)" $(( $(date +%s) - t0 )) "$n/2" >> "$S/summary.tsv"
      else check "$M ($R)" "$K" $rc $t0 "$D/$R/$f"; fi
    done
  done
  stop_kev
done

echo "############ Part C: final job (aggregations without a model, C5 clustering, judge)"
for M in A1 A2 A3 pc ges; do                # input: a full 200-story run (PC needs real data; 2 stories are degenerate)
  t0=$(date +%s); python benchmark/alt_aggregate.py $M grid/homelessness__gemma__jev "$S/agg_$M"; check $M "-" $? $t0 "$S/agg_$M/network.json"
done
t0=$(date +%s); python benchmark/alt_aggregate.py C5agg "$S/gemma/C5" "$S/agg_C5agg"; check C5agg "-" $? $t0 "$S/agg_C5agg/network.json"
start_llm gemma 32768
t0=$(date +%s); JUDGE_URL=$LLM_URL JUDGE=llm JUDGE_SLUG=gemma-4-31b-it JUDGE_ONLY=1 \
  python benchmark/evaluate_run.py homelessness "$S/kev27b/ours" 10; check judge gemma $? $t0 "$S/kev27b/ours/judge_cache_gemma-4-31b-it.json"
stop_llm

echo "############ summary"; column -t -s $'\t' "$S/summary.tsv"
