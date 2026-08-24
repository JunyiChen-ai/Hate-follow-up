#!/usr/bin/env bash
# Validation-select VERA's prompt bank, then run one deterministic frozen test.
# Every expensive stage is resumable through vera_adapter.py's per-video files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jehc223/miniconda3/envs/HateVideo/bin/python}"
CORPORA="${CORPORA:-hatemm mhclip_en mhclip_zh hateclipseg}"
TUNING_ROOT="${TUNING_ROOT:-$ROOT/results/reproduction/official_val/tuning}"
FINAL_ROOT="${FINAL_ROOT:-$ROOT/results/reproduction/official_val/final}"

cd "$ROOT"
for corpus in $CORPORA; do
  selection="$TUNING_ROOT/vera/$corpus"
  final="$FINAL_ROOT/vera/$corpus/seed_234"
  raw="$final/raw"
  mkdir -p "$selection" "$final" "$raw"

  "$PYTHON" scripts/reproduction_baselines/vera_adapter.py select \
    --corpus "$corpus" --out-dir "$selection"

  "$PYTHON" scripts/reproduction_baselines/vera_adapter.py infer \
    --corpus "$corpus" --split test --out-dir "$raw" \
    --prompt-json "$selection/selected_prompt.json"
  "$PYTHON" scripts/reproduction_baselines/vera_adapter.py postprocess \
    --corpus "$corpus" --split test --raw-dir "$raw" \
    --out "$final/scores.jsonl"
  "$PYTHON" scripts/reproduction_baselines/eval_baseline_scores.py \
    --corpus "$corpus" --split test --scores "$final/scores.jsonl" \
    --json-out "$final/frame_eval.json"

  "$PYTHON" - "$selection/selected_prompt.json" "$final/frozen_config.json" <<'PY'
import json
import sys
from pathlib import Path

source, out = map(Path, sys.argv[1:])
selected = json.loads(source.read_text())
payload = {
    "method": "vera",
    "corpus": selected["corpus"],
    "seed": 234,
    "deterministic_inference": True,
    "selection_split": selected["selection_split"],
    "selection_metric": selected["metric"],
    "selected": selected["selected"],
    "validation_scores": selected["scores"],
    "backbone": selected["backbone"],
    "attention_backend": selected["attention_backend"],
    "source": str(source),
}
out.write_text(json.dumps(payload, indent=2) + "\n")
PY
done

"$PYTHON" scripts/reproduction_baselines/aggregate_official_val.py
