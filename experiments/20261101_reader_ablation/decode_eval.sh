#!/usr/bin/env bash
# Decode the reader-ablation read sets and compare with the adopted method (README §3). CPU, uoa-lab1.
#   bash experiments/20261101_reader_ablation/decode_eval.sh main     (HateMM + HateClipSeg)
#   bash experiments/20261101_reader_ablation/decode_eval.sh DeHate
# Time-level arms: the adopted decode flags (r6_bma, raw key); base = the adopted decodes, re-decoded here once as a
# plumbing check (must equal them). asr_windows: SPVL composition on its reads and on the base reads. Paired bootstrap:
# experiments/20260927_dvd/analyze_dvd.py. Single evaluator src/eval/evaluate_four_datasets.py (called by both decoders).
set -euo pipefail
cd /home/jehc223/Hate-follow-up
source /home/jehc223/miniconda3/bin/activate HateVideo
G=${1:?main or DeHate}
R=runs/20261101_reader_ablation
TL_ARMS="no_transcript no_prefix_frames no_stance no_isolation joint"
if [ "$G" = main ]; then
  DS="HateMM HateClipSeg"; BASE_READS=runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned
  BASE_DEC=runs/20261031_window_evidence/decoded_main/adopted_rawkey
  for arm in $TL_ARMS asr_windows; do
    mkdir -p $R/merged/$arm
    cat $R/HateMM/$arm/predictions.jsonl $R/HateClipSeg/$arm/predictions.jsonl > $R/merged/$arm/predictions.jsonl
  done
  READS=$R/merged
else
  DS="DeHate"; BASE_READS=runs/20261031_window_evidence/DeHate/adopted
  BASE_DEC=runs/20261031_window_evidence/decoded/adopted
  READS=$R/DeHate
fi
TL="--noleak --transform nscore --key raw --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2"
python -u experiments/20260926_twolevel/twolevel_r2.py --run $BASE_READS --datasets $DS $TL --out-root $R/decoded --tag base_$G > $R/decoded_base_$G.log 2>&1
for arm in $TL_ARMS; do
  python -u experiments/20260926_twolevel/twolevel_r2.py --run $READS/$arm --datasets $DS $TL --out-root $R/decoded --tag ${arm}_$G > $R/decoded_${arm}_$G.log 2>&1
done
# A4: SPVL composition on both read sets (ASR windows cannot enter the time level)
for a in fixed asr; do
  src=$BASE_READS/predictions.jsonl; [ $a = asr ] && src=$READS/asr_windows/predictions.jsonl
  mkdir -p $R/compose/${a}_$G; cp $src $R/compose/${a}_$G/predictions.jsonl
  python experiments/20260910_spvl/compose.py --run-dir $R/compose/${a}_$G --intercept zv_plus_mean --residual rank --datasets $DS > $R/compose_${a}_$G.log 2>&1
done
# comparison roots: <arm>/predictions.jsonl
A=$R/compare_$G; mkdir -p $A
ln -sfn $PWD/$BASE_DEC $A/base
for arm in $TL_ARMS; do ln -sfn $PWD/$R/decoded/${arm}_$G $A/$arm; done
for a in fixed asr; do mkdir -p $A/spvl_$a; ln -sfn $PWD/$R/compose/${a}_$G/predictions_izv_plus_mean_rrank.jsonl $A/spvl_$a/predictions.jsonl; done
python -u experiments/20260927_dvd/analyze_dvd.py --root $A --datasets $DS --arms base $TL_ARMS --out $R/analysis_$G > $R/analysis_$G.log 2>&1
python -u experiments/20260927_dvd/analyze_dvd.py --root $A --datasets $DS --arms spvl_fixed spvl_asr --out $R/analysis_asr_$G > $R/analysis_asr_$G.log 2>&1
python - "$BASE_DEC/metrics.json" "$R/decoded/base_$G/metrics.json" <<'PY'
import json, sys
a, b = (json.load(open(p))["per_dataset"] for p in sys.argv[1:])
k = ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC")
d = max(abs(x[m] - y[m]) for x, y in zip(a, b) for m in k)
print(f"base re-decode vs adopted decode: max |diff| {d:.2e}")
assert d < 1e-9, "base re-decode does not reproduce the adopted decode"
PY
echo DECODE_EVAL_DONE $G
