#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
source /home/jehc223/miniconda3/bin/activate HateVideo
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p runs/20261005_m1_program/r1_handles_full_main_analysis
exec > >(tee -a runs/20261005_m1_program/r1_handles_full_main_analysis/run.log) 2>&1
hostname
echo "$BASHPID" > runs/20261005_m1_program/r1_handles_full_main_analysis/run.pid
python -u experiments/20261005_m1_program/handle_analyze.py --stage prepare
python -u experiments/20261005_m1_program/handle_analyze.py --stage evaluate --name base
python -u experiments/20261005_m1_program/handle_analyze.py --stage evaluate --name optimized
python -u experiments/20261005_m1_program/handle_analyze.py --stage report
