#!/usr/bin/env bash
# Decode every arm with the current time level (r6_bma, raw key; the settings of the adopted decode
# runs/20261031_window_evidence/decoded_main/adopted_rawkey/params.json) and the single evaluator. CPU, uoa-lab1.
# Merges the HateMM and HateClipSeg read sets of each arm first; `base` re-decodes the stored adopted reads unchanged.
set -euo pipefail
cd /home/jehc223/Hate-follow-up
source /home/jehc223/miniconda3/bin/activate HateVideo
R=runs/20261101_video_plan
for arm in own_plan other_plan; do
  mkdir -p $R/merged/$arm
  cat $R/HateMM/$arm/predictions.jsonl $R/HateClipSeg/$arm/predictions.jsonl > $R/merged/$arm/predictions.jsonl
  cp $R/HateMM/$arm/config.json $R/merged/$arm/config_HateMM.json; cp $R/HateClipSeg/$arm/config.json $R/merged/$arm/config_HateClipSeg.json
done
for arm in base own_plan other_plan; do
  run=$R/merged/$arm; [ $arm = base ] && run=runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned
  python -u experiments/20260926_twolevel/twolevel_r2.py --run $run --datasets HateMM HateClipSeg --noleak --transform nscore --key raw \
    --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2 --out-root $R/decoded --tag $arm > $R/decoded_$arm.log 2>&1
  python - "$R/decoded/$arm/metrics.json" "$arm" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for d in m["per_dataset"]:
    print(f"{sys.argv[2]:11s} {d['dataset']:12s} pooledROC {d['frame_ROC_AUC']:.4f} pooledPR {d['frame_PR_AUC']:.4f} within {d['within_video_macro_ROC_AUC']:.4f} (n_within {d['n_videos_defined']})  {sys.argv[1]}")
PY
done
echo DECODE_EVAL_DONE
