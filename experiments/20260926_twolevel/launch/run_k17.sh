#!/usr/bin/env bash
# Concern K7 (README §17): per-window frames under the current time level. Round 4 (§16) did not pass, so the time
# level is r3_m2. CPU, cached reads.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
S="experiments/20260926_twolevel/twolevel_r2.py --noleak --transform nscore --key calib --k 4 --arm m2"
"$PY" $S --run runs/20260910_spvl/full2_dual_evid_stance --tag k17_k20
"$PY" $S --run runs/20260910_spvl/full3_dual_evid_stance_w8 --tag k17_w8
"$PY" experiments/20260926_twolevel/analyze.py --round 8
echo ALL_DONE
