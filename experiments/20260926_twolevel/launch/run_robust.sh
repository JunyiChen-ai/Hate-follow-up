#!/usr/bin/env bash
# README §12: current vs new method on the family-study runs (other MLLMs) and the Qwen3-VL-8B reading ablations.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
OUT=runs/20260926_twolevel/robust
M=runs/20260910_spvl/mllm
run_pair() {   # $1 = cached run dir, $2 = tag prefix
  "$PY" experiments/20260922_til/til_infer.py --runs "$1" --model average --fusion max --dwell 80 --tag "$2_cur" --out-root $OUT
  "$PY" experiments/20260926_twolevel/twolevel_r2.py --run "$1" --noleak --k 4 --arm m2 --key calib --tag "$2_new" --out-root $OUT
}
for m in q3vl-2b q3vl-4b q3vl-8b q3vl-32b q25vl-7b internvl35-8b llava-ov-7b gemma3-12b; do run_pair $M/$m/full $m; done
for a in full nostance noctx noframes joint winonly; do run_pair $M/q3vl-8b/$a $a; done
"$PY" experiments/20260926_twolevel/summarize_robust.py
echo ALL_DONE
