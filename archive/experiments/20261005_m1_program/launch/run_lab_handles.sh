#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
source /home/jehc223/miniconda3/bin/activate HateVLM
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case "${1:?smoke or main}" in
  smoke) opts=(--smoke);;
  main) opts=();;
  *) exit 2;;
esac
nvidia-smi
python -u experiments/20261005_m1_program/handle_extract.py "${opts[@]}"
python -u experiments/20261005_m1_program/handle_measure.py "${opts[@]}"
echo ACQUISITION_DONE
