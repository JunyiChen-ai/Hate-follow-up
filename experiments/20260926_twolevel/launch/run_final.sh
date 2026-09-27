#!/usr/bin/env bash
# Final run of the current method (README §20): main arms, ablations, smoothing controls, DeHate (external).
# CPU, cached reads. Time level: r6_bma (README §19), no constant in seconds.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
T=experiments/20260926_twolevel/twolevel_r2.py
TL="--duration bma --bma-prior length --min-windows 2 --bma-grid 6"
S="$T --noleak --transform nscore --key calib $TL"
"$PY" $S --arm m2 --tag final_m2 --selftest
"$PY" $S --arm full --tag final_full
"$PY" $S --arm m2 --nocoupling --tag final_nocoupling
"$PY" $T --noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 0.5 --bma-grid 6 --arm m2 --tag final_k1
"$PY" $S --arm m2 --sharedchain --tag final_sharedchain
"$PY" $T --noleak --transform none --key calib $TL --arm m2 --tag final_rawscale
"$PY" $T --noleak --transform nscore --key raw $TL --arm m2 --tag final_rawkey
"$PY" $T --noleak --transform nscore --key none $TL --arm m2 --tag final_nokey
"$PY" $S --arm m2 --norank --tag final_norank
for sg in 2 4 8 16; do
  "$PY" experiments/20260922_til/til_infer.py --runs runs/20260926_glr/base_gridA --model gauss --sigma $sg --fusion max \
    --tag final_gauss$sg --out-root runs/20260926_twolevel; done
D="--run runs/20260927_dehate_external/reads_gridA --datasets DeHate --out-root runs/20260926_twolevel/final_dehate"
"$PY" $S --arm m2 $D --tag final_m2
"$PY" $S --arm full $D --tag final_full
"$PY" $S --arm m2 --nocoupling $D --tag final_nocoupling
for sg in 4 8; do
  "$PY" experiments/20260922_til/til_infer.py --runs runs/20260927_dehate_external/reads_gridA --datasets DeHate --model gauss \
    --sigma $sg --fusion max --tag final_gauss$sg --out-root runs/20260926_twolevel/final_dehate; done
"$PY" experiments/20260926_twolevel/analyze.py --round 11
"$PY" experiments/20260927_dvd/analyze_dvd.py --root runs/20260926_twolevel --arms final_m2 final_nocoupling final_k1 \
  final_sharedchain final_rawscale final_rawkey final_nokey final_norank --out runs/20260926_twolevel/analysis_final_pooled
"$PY" experiments/20260927_dvd/analyze_dvd.py --root runs/20260926_twolevel/final_dehate --arms final_m2 final_nocoupling \
  --datasets DeHate --out runs/20260926_twolevel/analysis_final_dehate
echo ALL_DONE
