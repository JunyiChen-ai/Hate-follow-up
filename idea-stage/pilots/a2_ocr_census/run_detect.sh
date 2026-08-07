#!/bin/bash
# Detached CPU launcher for the A2 OCR census detect stage.
# Resumes from per_frame.jsonl (dedupe by video id).
cd "$(dirname "$0")" || exit 1
source ~/venvs/SafetyContradiction/bin/activate
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
rm -f STATUS
python ocr_census.py --stage detect --nproc 8 --threads 2 --device cpu
rc=$?
if [ $rc -eq 0 ]; then
  echo "DETECT_DONE $(date -Is)" > STATUS
else
  echo "DETECT_FAIL rc=$rc $(date -Is)" > STATUS
fi
