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
run_arm() {  # tag, flags...   (skipped when the arm already has RUN_DONE)
  local tag=$1; shift
  if grep -q RUN_DONE $OUT/$tag/run.log 2>/dev/null; then echo "ARM_SKIP $tag" | tee -a $OUT/launch_$SET.log; return; fi
  "$PY" $S $DS --tag $tag "$@"
}
if [ "$SET" = dehate ]; then
  R=runs/20260927_dehate_external/reads_gridA; RD=runs/20260928_infer/dehate; DS="--datasets DeHate"; P=dh
  ln -sfn ../../20260926_twolevel/final_dehate/final_m2 $OUT/${P}_final_m2
  ln -sfn ../dehate/m1_joint $OUT/${P}_joint
else
  R=runs/20260926_glr/base_gridA; RD=runs/20260928_infer/main; DS="--datasets HateMM HateClipSeg"; P=mn
  run_arm ${P}_final_m2 --run $R
  run_arm ${P}_joint --run $RD/reads_joint
fi
run_arm ${P}_speechonly --run $R --chains z_speech
run_arm ${P}_visualonly --run $R --chains z_visual
run_arm ${P}_gate1 --run $R --gate verdicts --gate-visual $RD/reads_noctx --gate-speech $RD/reads_noframes
run_arm ${P}_gateV --run $R --gate verdicts --gate-visual $RD/reads_noctx        # visual chain gated only
run_arm ${P}_gateS --run $R --gate verdicts --gate-speech $RD/reads_noframes     # speech chain gated only
"$PY" experiments/20260927_dvd/analyze_dvd.py --root $OUT $DS --arms ${P}_final_m2 ${P}_joint ${P}_speechonly ${P}_visualonly ${P}_gate1 ${P}_gateV ${P}_gateS --out $OUT/analysis_$SET
"$PY" experiments/20260928_infer/gate_diag.py --or-arm $OUT/${P}_final_m2 --speech-arm $OUT/${P}_speechonly --gated-arm $OUT/${P}_gate1 $DS --out $OUT/analysis_$SET/gate_diag.txt
echo ALL_DONE $SET | tee -a $OUT/launch_$SET.log
