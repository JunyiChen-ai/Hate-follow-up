#!/usr/bin/env bash
# Resumable validation-only search for every feature-based WS baseline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jehc223/miniconda3/envs/HateVideo/bin/python}"
TRIALS="${TRIALS:-40}"
CORPORA="${CORPORA:-hatemm mhclip_en mhclip_zh hateclipseg}"
METHODS="${METHODS:-vadclip dsanet macilsd macilsd_audio macilsd_visual multihateloc cmhkf fed_wsvad_1client fed_wsvad_3client}"

cd "$ROOT"
for stage in clip vggish vit i3d bert; do
  PYTHON="$PYTHON" bash scripts/duplex/run_reproduction_features.sh "$stage"
done

for corpus in $CORPORA; do
  for method in $METHODS; do
    echo "=== tuning $method / $corpus ==="
    "$PYTHON" scripts/reproduction_baselines/tune_official_val.py \
      --method "$method" --corpus "$corpus" --trials "$TRIALS"
  done
done
