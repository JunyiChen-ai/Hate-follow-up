#!/usr/bin/env bash
# All declared arms of README §3 (CPU, cached reads of runs/20260926_glr/base_gridA), then the bootstrap analysis.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
S=experiments/20260926_twolevel/twolevel.py
"$PY" $S --arm current --tag current --selftest
"$PY" $S --arm m2 --tag m2
"$PY" $S --arm full --tag full
"$PY" $S --arm full --tag full_pooled --scope pooled
"$PY" $S --arm full --tag abl_nocoupling --nocoupling
"$PY" $S --arm full --tag abl_sharedchain --sharedchain
"$PY" $S --arm full --tag abl_noleak --noleak
"$PY" $S --arm full --tag abl_noatleast --noatleast
"$PY" $S --arm full --tag abl_vverdict --video verdict
"$PY" $S --arm full --tag abl_vreads --video reads
"$PY" experiments/20260926_twolevel/analyze.py
echo ALL_DONE
