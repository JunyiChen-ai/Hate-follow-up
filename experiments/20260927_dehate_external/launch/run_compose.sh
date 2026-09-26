#!/usr/bin/env bash
# The three frozen compositions of README §2 on the DeHate reads (CPU, uoa-lab1), then ZS-ImageBind's evaluation.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
R=runs/20260927_dehate_external; READS=$R/reads_gridA
"$PY" experiments/20260922_til/til_infer.py --runs $READS --model none --dwell 0 --tag spvl_r2 --out-root $R --datasets DeHate
"$PY" experiments/20260922_til/til_infer.py --runs $READS --model average --fusion max --dwell 80 --tag current --out-root $R --datasets DeHate
"$PY" experiments/20260926_twolevel/twolevel_r2.py --noleak --transform nscore --key calib --k 4 --arm m2 --run $READS \
  --tag r3_m2 --out-root $R --datasets DeHate
if [ -f $R/zs_imagebind/predictions.jsonl ]; then
  PYTHONPATH=. "$PY" src/eval/evaluate_four_datasets.py --predictions $R/zs_imagebind/predictions.jsonl \
    --gt-dir data/gt_4fps --out $R/zs_imagebind/metrics.json --datasets DeHate > /dev/null
fi
echo COMPOSE_DONE
