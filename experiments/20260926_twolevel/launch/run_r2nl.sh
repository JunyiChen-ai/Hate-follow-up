#!/usr/bin/env bash
# Ablations of the reduced round-2 model (README §10.9): r2_m2 with --noleak. r2_noleak itself is the run of run_r2.sh.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
S=experiments/20260926_twolevel/twolevel_r2.py
"$PY" $S --noleak --arm full --k 4 --tag r2nl_full
"$PY" $S --noleak --arm m2 --k 1 --tag r2nl_k1
"$PY" $S --noleak --arm m2 --k 2 --tag r2nl_k2
"$PY" $S --noleak --arm m2 --k 8 --tag r2nl_k8
"$PY" $S --noleak --arm m2 --k 4 --d-gap 40 --d-hate 40 --tag r2nl_d40
"$PY" $S --noleak --arm m2 --k 4 --d-gap 160 --d-hate 160 --tag r2nl_d160
"$PY" $S --noleak --arm m2 --k 4 --nocoupling --tag r2nl_nocoupling
"$PY" $S --noleak --arm m2 --k 4 --fusion carrier --tag r2nl_carrier
"$PY" experiments/20260926_twolevel/analyze.py --round 3
echo ALL_DONE
