#!/usr/bin/env bash
# README §15: reference arms (speech-only chain, visual-only chain) and the verdict-gated arm, per corpus set.
# Usage: bash experiments/20260928_infer/launch/run_gate.sh dehate|main    (CPU, cached reads; not uoa-lab1)
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
SET=$1
OUT=runs/20260928_infer/gate
mkdir -p $OUT
echo "host $(hostname) date $(date -Is) commit $(git rev-parse --short HEAD) set $SET" | tee -a $OUT/launch_$SET.log
T=experiments/20260926_twolevel/twolevel_r2.py
S="$T --noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2 --out-root $OUT"
if [ "$SET" = dehate ]; then
  R=runs/20260927_dehate_external/reads_gridA; RD=runs/20260928_infer/dehate; DS="--datasets DeHate"; P=dh
  ln -sfn ../../20260926_twolevel/final_dehate/final_m2 $OUT/${P}_final_m2
  ln -sfn ../dehate/m1_joint $OUT/${P}_joint
else
  R=runs/20260926_glr/base_gridA; RD=runs/20260928_infer/main; DS="--datasets HateMM HateClipSeg"; P=mn
  "$PY" $S $DS --run $R --tag ${P}_final_m2
  "$PY" $S $DS --run $RD/reads_joint --tag ${P}_joint
fi
"$PY" $S $DS --run $R --chains z_speech --tag ${P}_speechonly
"$PY" $S $DS --run $R --chains z_visual --tag ${P}_visualonly
"$PY" $S $DS --run $R --gate verdicts --gate-visual $RD/reads_noctx --gate-speech $RD/reads_noframes --tag ${P}_gate1
"$PY" experiments/20260927_dvd/analyze_dvd.py --root $OUT $DS --arms ${P}_final_m2 ${P}_joint ${P}_speechonly ${P}_visualonly ${P}_gate1 --out $OUT/analysis_$SET
"$PY" experiments/20260928_infer/gate_diag.py --or-arm $OUT/${P}_final_m2 --speech-arm $OUT/${P}_speechonly --gated-arm $OUT/${P}_gate1 $DS --out $OUT/analysis_$SET/gate_diag.txt
echo ALL_DONE $SET | tee -a $OUT/launch_$SET.log
