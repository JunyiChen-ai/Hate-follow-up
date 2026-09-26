#!/usr/bin/env bash
# Round 3 of the time level (README §14): normal-score reads before EM. Main-run arms, then the robustness runs.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
S="experiments/20260926_twolevel/twolevel_r2.py --noleak --transform nscore --key calib"
"$PY" $S --k 4 --arm m2 --tag r3_m2 --selftest
"$PY" $S --k 4 --arm full --tag r3_full
"$PY" $S --k 1 --arm m2 --tag r3_k1
"$PY" $S --k 4 --arm m2 --nocoupling --tag r3_nocoupling
"$PY" $S --k 2 --arm m2 --tag r3_k2
"$PY" $S --k 8 --arm m2 --tag r3_k8
"$PY" $S --k 4 --arm m2 --d-gap 40 --d-hate 40 --tag r3_d40
"$PY" $S --k 4 --arm m2 --d-gap 160 --d-hate 160 --tag r3_d160
"$PY" experiments/20260926_twolevel/analyze.py --round 5
M=runs/20260910_spvl/mllm; OUT=runs/20260926_twolevel/robust
for m in q3vl-2b q3vl-4b q3vl-8b q3vl-32b q25vl-7b internvl35-8b llava-ov-7b gemma3-12b; do
  "$PY" $S --k 4 --arm m2 --run $M/$m/full --tag ${m}_r3 --out-root $OUT; done
for a in full nostance noctx noframes joint winonly; do
  "$PY" $S --k 4 --arm m2 --run $M/q3vl-8b/$a --tag ${a}_r3 --out-root $OUT; done
"$PY" experiments/20260926_twolevel/summarize_robust.py --suffix r3
echo ALL_DONE
