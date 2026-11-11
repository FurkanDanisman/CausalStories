#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=group-test
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=group-test-%x-%j.out
#SBATCH --signal=B:USR1@600
#
# Grouping test (Section 2, Step 1 + naming only; no did-it-happen, connect or later steps).
# Kev's story-by-story grouping (run.py --mode merge-kev) on the v5 story graphs of one generator
# (grid_cluster/<world>__<GEN>__kev9b-llmgroup, new extraction prompt), with the same/opposite/different
# questions answered by
#   ANSWER=kev9b  Kev-9B
#   ANSWER=llm    the generator LLM itself (option probabilities from the logprobs of the answer letter)
# then the generator names the groups (Prompt 4c). Output: grid_cluster/grouptest/<world>__<GEN>__<ANSWER>/abstraction.json
#   sbatch --export=ALL,GEN=gemma,ANSWER=kev9b bench_group_test.sbatch
# Resumable (.done_<step>); 10 min before the limit the job submits itself again (at most MAX_ROUNDS rounds).
set -uo pipefail
GEN=${GEN:?GEN}; ANSWER=${ANSWER:?ANSWER}
WORLDS=${WORLDS:-"homelessness career mental_health"}; WORLDS=${WORLDS//+/ }
ROUND=${ROUND:-1}; MAX_ROUNDS=${MAX_ROUNDS:-4}
OUT=grid_cluster/grouptest/_logs/${GEN}__${ANSWER}; mkdir -p "$OUT"
DEC=$ANSWER; [ "$ANSWER" = llm ] && DEC=llm
source bench_lib.sh
LLM_PID=""; KEV_PIDS=""
requeue() { trap - USR1
  if [ "$ROUND" -lt "$MAX_ROUNDS" ]; then echo "########## time limit: requeueing as round $(( ROUND + 1 ))"
    (cd "$SLURM_SUBMIT_DIR" && env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /') sbatch \
      --export=ALL,GEN=$GEN,ANSWER=$ANSWER,WORLDS=${WORLDS// /+},ROUND=$(( ROUND + 1 )),MAX_ROUNDS=$MAX_ROUNDS bench_group_test.sbatch)
  else echo "########## time limit: round $ROUND of $MAX_ROUNDS, not requeueing"; fi
  kill $LLM_PID $KEV_PIDS 2>/dev/null; exit 0; }
trap requeue USR1
step() {                                    # step <dir> <name> <command...>: skip if done; time it
  local d=$1 s=$2; shift 2
  [ -f "$d/.done_$s" ] && { echo "--- skip $d $s (done)"; return 0; }
  echo "--- $d $s"; local t=$(date +%s)
  INFER_LOG="$PWD/$d/inference.jsonl" INFER_STEP="$s" "$@" & wait $! && { touch "$d/.done_$s"; printf "%s\t%s\n" "$s" $(( $(date +%s) - t )) >> "$d/timing.tsv"; } \
       || echo "!!! FAILED $d $s"
}
src() { echo "grid_cluster/${1}__${GEN}__kev9b-llmgroup"; }
dst() { echo "grid_cluster/grouptest/${1}__${GEN}__${ANSWER}"; }
ready=""
for W in $WORLDS; do
  n=$(ls $(src $W)/*__gemma-kev.chain.json 2>/dev/null | wc -l)
  if [ "$n" -ne 200 ]; then echo "!!! $(src $W): $n of 200 story graphs, skipped"; else ready="$ready $W"; mkdir -p "$(dst $W)"; fi
done
echo "########## grouping ($ANSWER answers) for:$ready"
start_kev                                   # ANSWER=llm: starts the generator LLM as the decision model
for W in $ready; do
  step "$(dst $W)" groups python run.py --mode merge-kev --kev-url "$KEV_URL" --graphs-glob "$(src $W)/*__gemma-kev.chain.json" --outdir "$(dst $W)"
done
[ "$ANSWER" = llm ] || { stop_kev; start_llm "$GEN" 65536; }   # with ANSWER=llm the generator is already up
echo "########## names ($GEN)"
for W in $ready; do
  [ -f "$(dst $W)/.done_groups" ] && step "$(dst $W)" names python run.py --mode name-groups --backend openai \
    --base-url "http://127.0.0.1:$LLM_PORT/v1" --model llm --outdir "$(dst $W)"
done
stop_llm
echo "########## done"
