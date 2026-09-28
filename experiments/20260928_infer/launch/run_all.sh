#!/usr/bin/env bash
# Three redesigns of the inference after the reads (README.md). CPU, cached reads runs/20260926_glr/base_gridA.
# §1 condition-dependent evidence (c1_*), §2 three read levels (l3_*; pre-check failed, run for information only),
# §3 joint segmentation (j_*). Reference: runs/20260926_twolevel/final_m2 (= r6_bma), linked into this run directory.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
OUT=runs/20260928_infer
T=experiments/20260926_twolevel/twolevel_r2.py
S="$T --noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --out-root $OUT"
C=$OUT/conditions/base_gridA.json
mkdir -p $OUT
for ref in final_m2 final_sharedchain final_nocoupling; do ln -sfn ../20260926_twolevel/$ref $OUT/$ref; done
"$PY" $S --arm m2 --tag c1_plumb --selftest
"$PY" $S --arm m2 --conditions $C --conditions-mode both --tag c1_cond
"$PY" $S --arm m2 --conditions $C --conditions-mode vis --tag c1_vis
"$PY" $S --arm m2 --conditions $C --conditions-mode sp --tag c1_sp
"$PY" $S --arm m2 --conditions $C --conditions-mode shuf --tag c1_shuf
"$PY" $S --arm full --conditions $C --conditions-mode both --tag c1_full
"$PY" $S --arm m2 --kind joint --tag j_m2
"$PY" $S --arm full --kind joint --tag j_full
"$PY" $S --arm m2 --kind three_nested --tag l3_m2
"$PY" $S --arm m2 --kind three_free --tag l3_free
"$PY" $S --arm full --kind three_nested --tag l3_full
"$PY" experiments/20260927_dvd/analyze_dvd.py --root $OUT --arms final_m2 c1_cond c1_vis c1_sp c1_shuf j_m2 l3_m2 l3_free \
  final_sharedchain final_nocoupling c1_plumb --out $OUT/analysis
echo ALL_DONE
