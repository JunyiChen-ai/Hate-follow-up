#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jehc223/miniconda3/envs/HateVideo/bin/python}"
CORPORA="${CORPORA:-hatemm mhclip_en mhclip_zh hateclipseg}"
METHODS="${METHODS:-vadclip dsanet macilsd multihateloc cmhkf fed_wsvad_1client fed_wsvad_3client}"
cd "$ROOT"
for corpus in $CORPORA; do
  for method in $METHODS; do
    "$PYTHON" scripts/reproduction_baselines/confirm_official_val.py \
      --method "$method" --corpus "$corpus"
  done
done
"$PYTHON" scripts/reproduction_baselines/aggregate_official_val.py
