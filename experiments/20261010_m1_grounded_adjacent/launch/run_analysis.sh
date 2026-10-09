#!/usr/bin/env bash
# CPU only, on sc474397: STAGE=prepare [SMOKE=1] | evaluate | report.
set -euo pipefail
cd /home/jehc223/Hate-follow-up
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
script=experiments/20261010_m1_grounded_adjacent/analyze.py
case ${STAGE:?} in
  prepare) .cache/envs/HateVLM/bin/python -u "$script" --stage prepare ${SMOKE:+--smoke};;
  evaluate) source /home/jehc223/miniconda3/bin/activate HateVideo; for name in grounded accept_all random_0 random_1 inverted; do python -u "$script" --stage evaluate --name "$name"; done;;
  report) source /home/jehc223/miniconda3/bin/activate HateVideo; python -u "$script" --stage report;;
  *) echo FAILED_invalid_stage; exit 2;;
esac
echo C41_ANALYSIS_STAGE_DONE
