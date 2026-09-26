#!/usr/bin/env bash
# Round 2: all declared arms of README §10.4 (CPU, cached reads of runs/20260926_glr/base_gridA), then the bootstrap
# analysis. The `current` arm is the round-1 run (runs/20260926_twolevel/current), not re-run.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
S=experiments/20260926_twolevel/twolevel_r2.py
"$PY" $S --arm m2 --k 4 --tag r2_m2 --selftest
"$PY" $S --arm full --k 4 --tag r2_full
"$PY" $S --arm m2 --k 1 --tag r2_k1
"$PY" $S --arm m2 --k 2 --tag r2_k2
"$PY" $S --arm m2 --k 8 --tag r2_k8
"$PY" $S --arm m2 --k 4 --d-gap 40 --d-hate 40 --tag r2_d40
"$PY" $S --arm m2 --k 4 --d-gap 160 --d-hate 160 --tag r2_d160
"$PY" $S --arm m2 --k 4 --nocoupling --tag r2_nocoupling
"$PY" $S --arm m2 --k 4 --noleak --tag r2_noleak
"$PY" $S --arm m2 --k 4 --fusion carrier --tag r2_carrier
"$PY" experiments/20260926_twolevel/analyze.py --round 2
echo ALL_DONE
