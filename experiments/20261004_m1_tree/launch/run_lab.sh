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
revision=${2:-r1}
case "$revision" in r1|r2|r3) ;; *) exit 2;; esac
nvidia-smi
if [[ $revision == r1 ]]; then
  python -u experiments/20261004_m1_tree/extract.py "${opts[@]}"
fi
python -u experiments/20261004_m1_tree/measure.py "${opts[@]}" --revision "$revision"
echo ACQUISITION_DONE
