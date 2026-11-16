#!/usr/bin/env bash
# Tables of the merged-row study (README §15), after the combined reader arms are read
# (experiments/20261101_reader_ablation/README.md §9). CPU, uoa-lab1.
#   bash experiments/20261114_exp_gates/mg_tables.sh
# Steps: coverage of the combined arms; merge HateMM + HateClipSeg; merge_rows.py for F5 and F3 (the em0 decodes of the
# new read sets run inside it, runs/20261114_exp_gates/gate_table/decodes/).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
R=runs/20261101_reader_ablation
MG="joint__no_window_evidence joint__no_isolation no_isolation__no_window_evidence joint__no_isolation__no_window_evidence"
count() { [ "$(wc -l < "$1")" -eq "$2" ] || { echo "FAILED coverage $1: $(wc -l < "$1") lines, expected $2"; exit 1; }; }
for a in $MG; do
  count $R/mg_HateMM/$a/predictions.jsonl 215; count $R/mg_HateClipSeg/$a/predictions.jsonl 118
  count $R/mg_DeHate/$a/predictions.jsonl 1341
  mkdir -p $R/mg_merged/$a
  cat $R/mg_HateMM/$a/predictions.jsonl $R/mg_HateClipSeg/$a/predictions.jsonl > $R/mg_merged/$a/predictions.jsonl
done
L=runs/20261114_exp_gates/gate_table/logs; mkdir -p $L
for f in F5 F3; do
  n=merge2__em0__$f
  $PY experiments/20261114_exp_gates/merge_rows.py --form $f --name $n > $L/$n.log 2>&1 \
      || { echo "FAILED table $n (log $L/$n.log)"; exit 1; }
  grep "^| w/ Joint Query, \|^| w/o Query Isolation + Window" $L/$n.log || true
done
echo MG_TABLES_DONE
