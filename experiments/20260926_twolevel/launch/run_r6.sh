#!/usr/bin/env bash
# Round 6 of the time level (README §19): per-video segment lengths, prior uniform in length. Main-run arms,
# robustness on the eight MLLMs, DeHate (external). CPU, cached reads.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
S="experiments/20260926_twolevel/twolevel_r2.py --noleak --transform nscore --key calib --arm m2 --duration bma --bma-prior length"
"$PY" $S --min-windows 2 --bma-fixed 80 --tag r6_plumb_fixed80
"$PY" $S --min-windows 2 --bma-grid 6 --tag r6_bma --selftest
"$PY" $S --min-windows 2 --bma-grid 4 --tag r6_bma_g4
"$PY" $S --min-windows 2 --bma-grid 10 --tag r6_bma_g10
"$PY" $S --min-windows 0.5 --bma-grid 6 --tag r6_k1
"$PY" experiments/20260926_twolevel/twolevel_r2.py --noleak --transform nscore --key calib --arm full --duration bma --bma-prior length \
  --min-windows 2 --bma-grid 6 --tag r6_full
M=runs/20260910_spvl/mllm; OUT=runs/20260926_twolevel/robust
for m in q3vl-2b q3vl-4b q3vl-8b q3vl-32b q25vl-7b internvl35-8b llava-ov-7b gemma3-12b; do
  "$PY" $S --min-windows 2 --bma-grid 6 --run $M/$m/full --tag ${m}_r6 --out-root $OUT; done
"$PY" $S --min-windows 2 --bma-grid 6 --run runs/20260927_dehate_external/reads_gridA --tag r6_bma \
  --out-root runs/20260927_dehate_external --datasets DeHate
"$PY" experiments/20260926_twolevel/analyze.py --round 10
"$PY" experiments/20260926_twolevel/summarize_robust.py --suffix r6 --base r3
echo ALL_DONE
