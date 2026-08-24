#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jehc223/miniconda3/envs/HateVideo/bin/python}"
CORPORA="${CORPORA:-hatemm mhclip_en mhclip_zh hateclipseg}"
METHODS="${METHODS:-vadclip dsanet macilsd macilsd_audio macilsd_visual multihateloc cmhkf fed_wsvad_1client fed_wsvad_3client}"
cd "$ROOT"
CODE_COMMIT="${CODE_COMMIT:-$(git rev-parse HEAD)}"
test -z "$(git status --porcelain --untracked-files=no)"
"$PYTHON" scripts/reproduction_baselines/verify_official_val_ready.py \
  --trials "${TRIALS:-40}" --corpora $CORPORA --methods $METHODS
for corpus in $CORPORA; do
  for method in $METHODS; do
    "$PYTHON" scripts/reproduction_baselines/confirm_official_val.py \
      --method "$method" --corpus "$corpus" --code-commit "$CODE_COMMIT"
  done
done
PYTHON="$PYTHON" CORPORA="$CORPORA" CODE_COMMIT="$CODE_COMMIT" \
  bash scripts/reproduction_baselines/run_official_val_vera.sh
