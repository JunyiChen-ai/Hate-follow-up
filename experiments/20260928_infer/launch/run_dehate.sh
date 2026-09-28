#!/usr/bin/env bash
# README §10: the three redesigns on DeHate (external, not a gate), requested by the user 2026-09-29. CPU, cached
# reads runs/20260927_dehate_external/reads_gridA. Runs on a lab machine other than uoa-lab1. Needs (rsync from
# uoa-lab1): data/gt_4fps/DeHate.npz, data/asr_whisper_large_v3/DeHate, data/frames_k20/DeHate,
# runs/20260927_dehate_external/reads_gridA, runs/20260926_twolevel/final_dehate/final_m2.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
OUT=runs/20260928_infer/dehate
R=runs/20260927_dehate_external/reads_gridA
T=experiments/20260926_twolevel/twolevel_r2.py
S="$T --noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --out-root $OUT --run $R --datasets DeHate"
mkdir -p $OUT
echo "host $(hostname)" > $OUT/launch_host.txt
ln -sfn ../../20260926_twolevel/final_dehate/final_m2 $OUT/final_m2
"$PY" experiments/20260928_infer/window_conditions.py --reads $R --datasets DeHate --out $OUT/conditions/reads_gridA.json
C=$OUT/conditions/reads_gridA.json
"$PY" $S --arm m2 --tag c1_plumb --selftest
"$PY" $S --arm m2 --conditions $C --conditions-mode both --tag c1_cond
"$PY" $S --arm m2 --conditions $C --conditions-mode shuf --tag c1_shuf
"$PY" $S --arm m2 --kind joint --tag j_m2
"$PY" $S --arm m2 --kind three_nested --tag l3_m2
"$PY" experiments/20260927_dvd/analyze_dvd.py --root $OUT --arms final_m2 c1_cond c1_shuf j_m2 l3_m2 c1_plumb \
  --datasets DeHate --out $OUT/analysis
echo ALL_DONE
