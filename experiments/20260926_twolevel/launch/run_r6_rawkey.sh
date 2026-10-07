#!/usr/bin/env bash
# README §24: r6_bma on the eight MLLMs with the raw video key (--key raw) instead of the calibrated key (--key calib).
# Every other flag is identical to the robustness lines of run_r6.sh. CPU, cached reads.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
S="experiments/20260926_twolevel/twolevel_r2.py --noleak --transform nscore --key raw --arm m2 --duration bma --bma-prior length"
M=runs/20260910_spvl/mllm; OUT=runs/20260926_twolevel/robust
for m in q3vl-2b q3vl-4b q3vl-8b q3vl-32b q25vl-7b internvl35-8b llava-ov-7b gemma3-12b; do
  "$PY" $S --min-windows 2 --bma-grid 6 --run $M/$m/full --tag ${m}_r6_rawkey --out-root $OUT; done
echo ALL_DONE
