#!/usr/bin/env bash
# Composition step (README §11): calibrated video key on the reduced round-2 time level.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
S="experiments/20260926_twolevel/twolevel_r2.py --noleak --k 4"
"$PY" $S --arm m2 --key calib --tag c_m2
"$PY" $S --arm full --key calib --tag c_full
"$PY" $S --arm m2 --key calib --norank --tag c_norank
"$PY" $S --arm m2 --key none --tag c_nokey
for s in 0.5 0.25 0.125; do "$PY" $S --arm m2 --key scaled --key-scale $s --tag c_s$s; done
"$PY" experiments/20260926_twolevel/analyze.py --round 4
echo ALL_DONE
