#!/usr/bin/env bash
# CPU only, on sc474397: STAGE=selfcheck | prepare [SMOKE=1] | build | evaluate [THREADS=8] | report
set -euo pipefail
cd /home/jehc223/Hate-follow-up
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
script=experiments/20261013_m1_pairwise_evidence/compare_analyze.py
case ${STAGE:?} in
  selfcheck|prepare|build) .cache/envs/HateVLM/bin/python -u "$script" --stage "$STAGE" ${SMOKE:+--smoke};;
  evaluate) source /home/jehc223/miniconda3/bin/activate HateVideo; python -u "$script" --stage evaluate --threads "${THREADS:-8}";;
  report) source /home/jehc223/miniconda3/bin/activate HateVideo; python -u "$script" --stage report;;
  *) echo FAILED_invalid_stage; exit 2;;
esac
echo C42_ANALYSIS_STAGE_DONE
