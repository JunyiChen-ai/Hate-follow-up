#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
source /home/jehc223/miniconda3/bin/activate HateVLM
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mode="${1:?smoke or main}"
arm="${2:-soft}"
case "$mode" in
  smoke) opts=(--smoke);;
  main) opts=();;
  *) exit 2;;
esac
nvidia-smi
python -u experiments/20261004_m1_acoustic/extract.py "${opts[@]}"
python -u experiments/20261004_m1_acoustic/measure.py "${opts[@]}" --arm "$arm"
echo ACQUISITION_DONE
