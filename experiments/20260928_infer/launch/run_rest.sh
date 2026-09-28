#!/usr/bin/env bash
# Second half of run_all.sh after the joint-chain crash (emission array sized for two previous-level ids; fixed to
# n_plv x S). The c1_* arms of run_all.sh are unaffected (kind two, n_plv = 2) and are not re-run.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
OUT=runs/20260928_infer
T=experiments/20260926_twolevel/twolevel_r2.py
S="$T --noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --out-root $OUT"
"$PY" $S --arm m2 --kind joint --tag j_m2 --selftest
"$PY" $S --arm full --kind joint --tag j_full
"$PY" $S --arm m2 --kind three_nested --tag l3_m2
"$PY" $S --arm m2 --kind three_free --tag l3_free
"$PY" $S --arm full --kind three_nested --tag l3_full
"$PY" experiments/20260927_dvd/analyze_dvd.py --root $OUT --arms final_m2 c1_cond c1_vis c1_sp c1_shuf j_m2 l3_m2 l3_free \
  final_sharedchain final_nocoupling c1_plumb --out $OUT/analysis
echo ALL_DONE
