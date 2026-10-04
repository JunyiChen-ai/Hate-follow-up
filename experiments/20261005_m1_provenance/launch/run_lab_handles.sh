#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
source /home/jehc223/miniconda3/bin/activate HateVLM
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
case "${1:?smoke or main}" in
  smoke) opts=(--smoke);;
  main) opts=();;
  *) exit 2;;
esac
nvidia-smi
python -u experiments/20261005_m1_provenance/extract_handles.py "${opts[@]}"
python -u experiments/20261005_m1_provenance/measure_handles.py "${opts[@]}"
echo ACQUISITION_DONE
