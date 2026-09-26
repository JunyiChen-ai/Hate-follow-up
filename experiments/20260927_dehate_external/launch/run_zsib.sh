#!/usr/bin/env bash
# ZS-ImageBind (scripts/reproduction_baselines/zs_imagebind.py, unchanged) on the DeHate test split.
# Needs the HateVideo env, the videos at the manifest paths, data/assets/imagebind/ and third_party/lavad/libs/ImageBind.
# Usage (lab): cd ~/Hate-follow-up && setsid nohup bash experiments/20260927_dehate_external/launch/run_zsib.sh > runs/20260927_dehate_external/launch_zsib.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN=zs_imagebind; OUT="runs/20260927_dehate_external/$RUN"; mkdir -p "$OUT"
PY="${ZSIB_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
echo "host $(hostname) date $(date -Is) commit $(git rev-parse --short HEAD) run $RUN" | tee -a "$OUT/launch.log"
set +e
"$PY" scripts/reproduction_baselines/zs_imagebind.py --exp-id 20260927_dehate_external --run-name "$RUN" \
  --datasets DeHate --manifest data/manifests/DeHate_test.jsonl "$@" >> "$OUT/launch.log" 2>&1
RC=$?
set -e
tail -3 "$OUT/run.log" || true
if [ $RC -ne 0 ] || ! grep -q "DONE" "$OUT/run.log"; then echo "RUN_FAILED $RUN rc=$RC" | tee -a "$OUT/launch.log"; exit 1; fi
echo "RUN_DONE $RUN $(date -Is)" | tee -a "$OUT/launch.log"
