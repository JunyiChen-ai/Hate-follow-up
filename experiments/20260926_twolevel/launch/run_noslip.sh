#!/usr/bin/env bash
# README §15.7: two phases with learned geometric durations (s_noslip) - persistence ablation and robustness runs.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
S="experiments/20260926_twolevel/slip.py --noslip"
"$PY" $S --arm m2 --nocoupling --tag s_noslip_nocoupling
M=runs/20260910_spvl/mllm; OUT=runs/20260926_twolevel/robust
for m in q3vl-2b q3vl-4b q3vl-8b q3vl-32b q25vl-7b internvl35-8b llava-ov-7b gemma3-12b; do
  "$PY" $S --arm m2 --run $M/$m/full --tag ${m}_sn --out-root $OUT; done
for a in full nostance noctx noframes joint winonly; do
  "$PY" $S --arm m2 --run $M/q3vl-8b/$a --tag ${a}_sn --out-root $OUT; done
"$PY" experiments/20260926_twolevel/summarize_robust.py --suffix sn
echo ALL_DONE
