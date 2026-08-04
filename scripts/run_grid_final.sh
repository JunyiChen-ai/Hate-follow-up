#!/bin/bash
# End-to-end Phase C/D/E driver (CPU-only). Run after all GPU Phase A+B
# jobs complete. Enumerates all 5 stage-1 slugs + 2b (=6 total).
#
# Idempotent: re-running regenerates all artefacts from the current
# holistic_<slug>/ and offline_test_ih_*.jsonl files on disk.
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

SLUGS="2b qwen2.5-vl-7b gemma-3-12b-it minicpm-v-26 pixtral-12b-2409 internvl3-14b"

echo "=== Phase C: threshold search + emit baseline_preds_v2_<slug>_<crit>.jsonl ==="
python src/boundary_rescue/threshold_search.py --slugs $SLUGS --emit-preds

echo ""
echo "=== Phase D: entropy-above-mean band per slug ==="
# The GMM fit and mean-entropy cut depend only on score distribution,
# not on the binary threshold — so one pass per slug (criterion=protocol)
# is sufficient. grid_eval_all.py still reads the correct oracle-crit
# baseline preds separately.
for S in $SLUGS; do
  python src/boundary_rescue/select_entropy_band.py --model-tag $S --criterion protocol 2>&1 | tail -5
done

echo ""
echo "=== Phase E: grid evaluation ==="
python src/boundary_rescue/grid_eval_all.py --slugs $SLUGS

echo ""
echo "=== Artefacts ==="
ls -la results/boundary_rescue/grid_eval/
