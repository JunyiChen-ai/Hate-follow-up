#!/usr/bin/env bash
# Validation-select VERA's prompt bank, then run one deterministic frozen test.
# Every expensive stage is resumable through vera_adapter.py's per-video files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jehc223/miniconda3/envs/HateVideo/bin/python}"
CORPORA="${CORPORA:-hatemm mhclip_en mhclip_zh hateclipseg}"
TUNING_ROOT="${TUNING_ROOT:-$ROOT/results/reproduction/official_val/tuning}"
FINAL_ROOT="${FINAL_ROOT:-$ROOT/results/reproduction/official_val/final}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-20480}"

mkdir -p "$FINAL_ROOT/vera"
VERA_LOCK="$FINAL_ROOT/vera/.runner.lock"
exec 9>"$VERA_LOCK"
if ! flock -n 9; then
  echo "another VERA runner owns $VERA_LOCK" >&2
  exit 1
fi

wait_for_vera_gpu() {
  local free_mib
  while true; do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_GPU_MIB )); then
      return 0
    fi
    echo "VERA waiting for ${MIN_FREE_GPU_MIB} MiB free GPU memory (now ${free_mib:-unknown})" >&2
    sleep 60
  done
}

cd "$ROOT"
for corpus in $CORPORA; do
  selection="$TUNING_ROOT/vera/$corpus"
  final="$FINAL_ROOT/vera/$corpus/seed_234"
  raw="$final/raw"
  mkdir -p "$selection" "$final" "$raw"

  wait_for_vera_gpu
  "$PYTHON" scripts/reproduction_baselines/vera_adapter.py select \
    --corpus "$corpus" --out-dir "$selection"

  wait_for_vera_gpu
  "$PYTHON" scripts/reproduction_baselines/vera_adapter.py infer \
    --corpus "$corpus" --split test --out-dir "$raw" \
    --prompt-json "$selection/selected_prompt.json"
  "$PYTHON" scripts/reproduction_baselines/vera_adapter.py postprocess \
    --corpus "$corpus" --split test --raw-dir "$raw" \
    --out "$final/scores.jsonl"
  "$PYTHON" scripts/reproduction_baselines/eval_baseline_scores.py \
    --corpus "$corpus" --split test --scores "$final/scores.jsonl" \
    --require-full-coverage --json-out "$final/frame_eval.json"

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
temporary = out.with_name(out.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(out)
PY
done

"$PYTHON" scripts/reproduction_baselines/aggregate_official_val.py
