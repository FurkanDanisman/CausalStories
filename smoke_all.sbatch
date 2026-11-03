#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=smoke
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=smoke-%j.out
#
# Smoke test of every GPU piece of the benchmark, one job per model so they run in parallel (2 stories each):
#   GEN=<gemma|qwen|llama|gptoss>  our node extraction, C1-C6, group naming, A4 (LLM merge)
#                                  with GEN=gemma also: A1-A3, A5 (PC / GES), C5 clustering, and the judge
#   DEC=<kev27b|kev9b>             our chain walk, then our aggregation (groups, did it happen, connect) on our story
#                                  graphs and on C1-format graphs; input = 2 stories of the seed run (no LLM needed)
# A step passes when it exits 0 AND no story failed (a failed story is otherwise kept as an empty graph).
# Each job writes grid_cluster/_smoke/<model>/summary.tsv (step, model, pass/FAIL, seconds, stories ok / total).
#   for g in gemma qwen llama gptoss; do sbatch --export=ALL,GEN=$g smoke_all.sbatch; done
#   for k in kev27b kev9b; do sbatch --export=ALL,DEC=$k smoke_all.sbatch; done
set -uo pipefail
M=${GEN:-${DEC:?set GEN or DEC}}
S=grid_cluster/_smoke/$M
OUT=$S
rm -rf "$S"; mkdir -p "$S"
source bench_lib.sh
K=${K:-8}; export MAX_EVENTS=$K              # event budget K: our Prompt 1a and C1, C5, C6 (same for every method)
SEED=$S/seed                                            # fresh copy of the seed run (200 homelessness stories)
mkdir -p "$SEED" && tar -xzf grid_seed.tar.gz -C "$SEED" grid/homelessness__gemma__jev && SEED=$SEED/grid/homelessness__gemma__jev
find "$S" -name "._*" -delete
python -c "import json;json.dump(json.load(open('homelessness_v3.json'))[:2],open('$S/texts.json','w'))"
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
        it = json.load(open(f"{target}/chain_nodes.json")); tot = len(it)
        ok = sum(1 for x in it if x["nodes"] and not x["raw"].get("failed"))
elif os.path.exists(target):
    ok = tot = 1
res = "pass" if rc == "0" and tot and ok == tot else "FAIL"
open(tsv, "a").write(f"{step}\t{model}\t{res}\t{int(time.time()) - int(t0)}\t{ok}/{tot}\n")
print(f"=== {step} [{model}]: {res} ({ok}/{tot})", flush=True)
PY
}
count() {                    # count <step> <model> <rc> <t0> <glob> : pass when 2 files exist
  local n; n=$(ls $5 2>/dev/null | wc -l | tr -d " ")
  printf "%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$( [ "$3" = 0 ] && [ "$n" = 2 ] && echo pass || echo FAIL)" $(( $(date +%s) - $4 )) "$n/2" >> "$S/summary.tsv"
}

if [ -n "${GEN:-}" ]; then
  echo "############ generator $GEN"
  start_llm "$GEN" 65536
  t0=$(date +%s); python run.py --mode chain-nodes --backend openai --base-url "http://127.0.0.1:$LLM_PORT/v1" --model llm \
    --texts-file "$S/texts.json" --max-nodes $K --outdir "$S/ours" --workers 16; check nodes "$GEN" $? $t0 "$S/ours"
  for C in C1 C2 C3 C4 C5 C6; do
    t0=$(date +%s); python benchmark/baselines.py $C "$S/texts.json" "$S/$C" --nodes "$S/ours/chain_nodes.json" --workers 16
    rc=$?; if [ $C = C6 ]; then check $C "$GEN" $rc $t0 "$S/$C/network.json"; else check $C "$GEN" $rc $t0 "$S/$C"; fi
  done
  mkdir -p "$S/names"; cp "$SEED/abstraction.json" "$S/names/"
  t0=$(date +%s); python run.py --mode name-groups --backend openai --base-url "http://127.0.0.1:$LLM_PORT/v1" --model llm \
    --outdir "$S/names"; check name-groups "$GEN" $? $t0 "$S/names/abstraction.json"
  t0=$(date +%s); python benchmark/alt_aggregate.py A4 "$SEED" "$S/A4"; check A4 "$GEN" $? $t0 "$S/A4/network.json"
  if [ "$GEN" = gemma ]; then
    echo "############ final-job pieces: aggregations without a model (input: the 200-story seed run), C5 clustering, judge"
    for A in A1 A2 A3 pc ges; do
      t0=$(date +%s); python benchmark/alt_aggregate.py $A "$SEED" "$S/agg_$A"; check $A "-" $? $t0 "$S/agg_$A/network.json"
    done
    t0=$(date +%s); python benchmark/alt_aggregate.py C5agg "$S/C5" "$S/agg_C5agg"; check C5agg "-" $? $t0 "$S/agg_C5agg/network.json"
    J=$S/judge; mkdir -p "$J"; cp "$SEED/abstraction.json" "$SEED"/ho00[01]__gemma-kev.chain.json "$J/"
    t0=$(date +%s); JUDGE_URL=$LLM_URL JUDGE=llm JUDGE_SLUG=gemma-4-31b-it JUDGE_ONLY=1 \
      python benchmark/evaluate_run.py homelessness "$J" 10; check judge gemma $? $t0 "$J/judge_cache_gemma-4-31b-it.json"
  fi
  stop_llm
else
  echo "############ decision model $DEC (input: the first 2 stories of the seed run)"
  mkdir -p "$S/ours" "$S/C1"
  python - "$SEED" "$S" <<'PY'
import json, sys
sys.path.insert(0, "benchmark")
from baselines import graph
seed, s = sys.argv[1:3]
items = json.load(open(f"{seed}/chain_nodes.json"))[:2]
json.dump(items, open(f"{s}/ours/chain_nodes.json", "w"))
for it in items:                                     # C1-format story graphs (every pair answered) from the seed graphs
    g = json.load(open(f"{seed}/{it['id']}__gemma-kev.chain.json"))
    edges = [(e["head"], e["tail"], e["rel"], e["prob"]) for e in g["edges"]]
    json.dump(graph(it["id"], it["text"], g["nodes"], g["kinds"], edges, "C1"), open(f"{s}/C1/{it['id']}__gemma-kev.chain.json", "w"))
PY
  start_kev
  t0=$(date +%s); python run.py --mode chain-kev --kev-url "http://127.0.0.1:$KEV_PORT" --nodes-file "$S/ours/chain_nodes.json" \
    --tag gemma-kev --outdir "$S/ours"; check chain-walk "$DEC" $? $t0 "$S/ours"
  for R in ours C1; do                     # group names are not needed to test these steps (tested in the generator jobs)
    t0=$(date +%s); python run.py --mode merge-kev --kev-url "http://127.0.0.1:$KEV_PORT" --graphs-glob "$S/$R/*__gemma-kev.chain.json" --outdir "$S/$R"
    check "groups ($R)" "$DEC" $? $t0 "$S/$R/abstraction.json"
    t0=$(date +%s); python run.py --mode node-happened --kev-url "http://127.0.0.1:$KEV_PORT" --graphs-glob "$S/$R/*__gemma-kev.chain.json" --outdir "$S/$R"
    count "node-happened ($R)" "$DEC" $? $t0 "$S/$R/*__happened.json"
    t0=$(date +%s); python run.py --mode implied --kev-url "http://127.0.0.1:$KEV_PORT" --graphs-glob "$S/$R/*__gemma-kev.chain.json" --outdir "$S/$R"
    check "implied ($R)" "$DEC" $? $t0 "$S/$R/implied.json"
    t0=$(date +%s); python run.py --mode connect --kev-url "http://127.0.0.1:$KEV_PORT" --graphs-glob "$S/$R/*__gemma-kev.chain.json" --outdir "$S/$R"
    count "connect ($R)" "$DEC" $? $t0 "$S/$R/*__connect.json"
  done
  stop_kev
fi

echo "############ summary"; column -t -s $'\t' "$S/summary.tsv"; cat "$S/model_loads.tsv" 2>/dev/null || true
