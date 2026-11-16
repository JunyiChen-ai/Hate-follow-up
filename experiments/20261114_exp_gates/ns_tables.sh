#!/usr/bin/env bash
# Tables of the no-verdict-turn family (README §14). CPU, uoa-lab1.
#   bash experiments/20261114_exp_gates/ns_tables.sh main   (HateMM + HateClipSeg, when their reads are back)
#   bash experiments/20261114_exp_gates/ns_tables.sh all    (all three corpora)
# Steps: merge the HateMM and HateClipSeg ns reads; SPVL composition of the ns segment-window reads (the fixed-window
# side, runs/20261101_reader_ablation/compose/ns_fixed_*, is the composition of the stored no_stance reads); then
# gate_table.py --family ns --decoder em0 for the Integrators of README §14 (raw, F2, F3) and F5 (HateClipSeg taxonomy
# read). Decodes run inside gate_table.py (runs/20261114_exp_gates/gate_table/decodes/).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
R=runs/20261101_reader_ablation
P=${1:?main or all}
NS="ns_no_transcript ns_no_prefix_frames ns_asr_windows ns_no_isolation ns_joint ns_no_window_evidence"
count() { [ "$(wc -l < "$1")" -eq "$2" ] || { echo "FAILED coverage $1: $(wc -l < "$1") lines, expected $2"; exit 1; }; }
for a in $NS; do
  count $R/ns_HateMM/$a/predictions.jsonl 215; count $R/ns_HateClipSeg/$a/predictions.jsonl 118
  mkdir -p $R/ns_merged/$a
  cat $R/ns_HateMM/$a/predictions.jsonl $R/ns_HateClipSeg/$a/predictions.jsonl > $R/ns_merged/$a/predictions.jsonl
  [ "$P" = all ] && count $R/ns_DeHate/$a/predictions.jsonl 1341
done
compose() {   # group, reads directory, datasets
  mkdir -p $R/compose/ns_asr_$1
  cp $2/ns_asr_windows/predictions.jsonl $R/compose/ns_asr_$1/predictions.jsonl
  $PY experiments/20260910_spvl/compose.py --run-dir $R/compose/ns_asr_$1 --intercept zv_plus_mean --residual rank \
      --datasets $3 > $R/compose_ns_asr_$1.log 2>&1 || { echo "FAILED compose $1"; exit 1; }
}
compose main $R/ns_merged "HateMM HateClipSeg"
[ "$P" = all ] && compose DeHate $R/ns_DeHate "DeHate"
L=runs/20261114_exp_gates/gate_table/logs; mkdir -p $L
for f in raw F2 F3 F5; do
  n=ns_${P}__em0__$f
  $PY experiments/20261114_exp_gates/gate_table.py --family ns --decoder em0 --form $f --part $P --jobs 5 --name $n \
      > $L/$n.log 2>&1 || { echo "FAILED table $n (log $L/$n.log)"; exit 1; }
  grep "^Cells not below" $L/$n.log
done
echo NS_TABLES_DONE $P
