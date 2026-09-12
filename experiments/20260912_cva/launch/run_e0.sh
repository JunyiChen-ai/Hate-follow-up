#!/usr/bin/env bash
# CVA E0: the declared main arm on both corpora, then compose + evaluate.
# The keeponly control arm is a separate launch (run_keeponly.sh) so E0's timing stays clean.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${CVA_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
export PYTHONPATH=.

for spec in "HateMM:e0_hatemm" "HateClipSeg:e0_hcs"; do
  ds="${spec%%:*}"; run="${spec##*:}"
  "$PY" experiments/20260912_cva/cva.py --run-name "$run" --datasets "$ds" --read exclude
done

# one predictions file over both corpora, so the evaluator sees the same shape as SPVL-r2's
mkdir -p runs/20260912_cva/e0
cat runs/20260912_cva/e0_hatemm/predictions.jsonl runs/20260912_cva/e0_hcs/predictions.jsonl \
  > runs/20260912_cva/e0/predictions.jsonl
"$PY" experiments/20260912_cva/cva_compose.py --run-dir runs/20260912_cva/e0 \
  --intercept zv_plus_mean --residual rank
echo "CVA E0 DONE"
