#!/usr/bin/env bash
# CPU: primary window-level comparison + derived runs, then frame-level evaluation of base and derived runs through
# experiments/20260922_til/til_infer.py (shared evaluator inside), with and without the duration prior.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
"$PY" experiments/20260926_glr/glr_analyze.py
"$PY" experiments/20260922_til/til_infer.py --runs runs/20260926_glr/base_gridA --model none --dwell 0 --tag base_gridA_spvlr2 --out-root runs/20260926_glr/infer
for D in 80 0; do
  "$PY" experiments/20260922_til/til_infer.py --runs runs/20260926_glr/base_gridA --model average --fusion max --dwell $D \
      --tag base_gridA_d$D --out-root runs/20260926_glr/infer
  for V in full_assistant full_document none_assistant none_document; do
    "$PY" experiments/20260922_til/til_infer.py --runs runs/20260926_glr/derived/$V --model average --fusion max --dwell $D \
        --tag glr_${V}_d$D --out-root runs/20260926_glr/infer
  done
done
echo ANALYSIS_DONE
