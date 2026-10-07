#!/usr/bin/env bash
# CPU only, on sc474397 after both control runs are returned: JOB=dualpath|uniform STAGE=prepare|evaluate, or STAGE=report.
set -euo pipefail
cd /home/jehc223/Hate-follow-up
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
script=experiments/20261007_m1_streamingtom/controls_analyze.py
case ${STAGE:?} in
  prepare) .cache/envs/HateVLM/bin/python -u "$script" --stage prepare --job "${JOB:?}" ${SMOKE:+--smoke};;
  evaluate)
    source /home/jehc223/miniconda3/bin/activate HateVideo
    if [[ $JOB == dualpath ]]; then names=(replay no_remote nearest); else names=(uniform); fi
    for name in "${names[@]}"; do python -u "$script" --stage evaluate --job "$JOB" --name "$name"; done;;
  report) source /home/jehc223/miniconda3/bin/activate HateVideo; python -u "$script" --stage report;;
  *) echo FAILED_invalid_stage; exit 2;;
esac
echo CONTROLS_ANALYSIS_STAGE_DONE
