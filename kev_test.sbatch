#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=kev-test
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=kev-test-%j.out
#
# Re-asks two suspicious grouping questions of the Gemma x homelessness run to Kev-9B: as the pipeline asked them
# (full request), alone (x3) and in reverse order (benchmark/kev_test.py). One GPU, one Kev-9B server.
#   sbatch kev_test.sbatch
set -uo pipefail
OUT=grid_cluster/_logs/kev_test; mkdir -p "$OUT"
source bench_lib.sh
start_kev
R=grid_cluster/homelessness__gemma__kev9b
python benchmark/kev_test.py $R ho060 "being homeless" "$KEV_URL"
echo
python benchmark/kev_test.py $R ho111 "having somewhere to live" "$KEV_URL"
stop_kev
