#!/usr/bin/env bash
set -euo pipefail
cd /home/jehc223/Hate-follow-up
source /home/jehc223/miniconda3/bin/activate HateVideo
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out=runs/20261005_m1_program/r3_handles_full_main_analysis
mkdir -p "$out"
exec >>"$out/run.log" 2>&1
hostname
printf '%s\n' "$$" >"$out/run.pid"
python -u experiments/20261005_m1_program/handle_analyze.py --revision 3 --stage prepare
python -u experiments/20261005_m1_program/handle_analyze.py --revision 3 --stage evaluate --name base
python -u experiments/20261005_m1_program/handle_analyze.py --revision 3 --stage evaluate --name optimized
python -u experiments/20261005_m1_program/handle_analyze.py --revision 3 --stage report
