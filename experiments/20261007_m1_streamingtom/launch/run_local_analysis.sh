#!/usr/bin/env bash
# CPU only, on sc474397: STAGE=prepare [SMOKE=1] | evaluate | report.
set -euo pipefail
cd /home/jehc223/Hate-follow-up
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
script=experiments/20261007_m1_streamingtom/local_analyze.py
case ${STAGE:?} in
  prepare) .cache/envs/HateVLM/bin/python -u "$script" --stage prepare ${SMOKE:+--smoke};;
  evaluate)
    source /home/jehc223/miniconda3/bin/activate HateVideo
    for name in custom_native local_clean no_remote_replay; do python -u "$script" --stage evaluate --name "$name"; done;;
  report) source /home/jehc223/miniconda3/bin/activate HateVideo; python -u "$script" --stage report;;
  *) echo FAILED_invalid_stage; exit 2;;
esac
echo LOCAL_ANALYSIS_STAGE_DONE
