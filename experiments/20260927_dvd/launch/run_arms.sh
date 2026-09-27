#!/usr/bin/env bash
# DVD arms (README §4) on HateMM + HateClipSeg. Time level = r3_m2 (the current method when these run). CPU, cached reads.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
R=runs/20260927_dvd; OUT=$R/arms
S="experiments/20260926_twolevel/twolevel_r2.py --noleak --transform nscore --key calib --k 4 --out-root $OUT"
D="--dvd-reads $R/reads_main/reads.jsonl"
"$PY" $S --arm m2 --tag base
"$PY" $S --arm m2 $D --dvd-conds viol,T,E --tag dvd
"$PY" $S --arm m2 $D --dvd-conds viol,E --tag dvd_noT
"$PY" $S --arm m2 $D --dvd-conds viol,T --tag dvd_noE
"$PY" $S --arm m2 $D --dvd-conds A,T,E --tag dvd_ATE
"$PY" $S --arm full --tag base_full
"$PY" $S --arm full $D --dvd-conds viol,T,E --tag dvd_full
"$PY" experiments/20260927_dvd/analyze_dvd.py --root $OUT --arms base dvd dvd_noT dvd_noE dvd_ATE --out $R/analysis
echo ALL_DONE
