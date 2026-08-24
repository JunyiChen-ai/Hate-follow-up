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
CODE_COMMIT="${CODE_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}"

verify_code_commit() {
  local head dirty
  head="$(git -C "$ROOT" rev-parse HEAD)"
  dirty="$(git -C "$ROOT" status --porcelain --untracked-files=no)"
  if [[ "$head" != "$CODE_COMMIT" || -n "$dirty" ]]; then
    echo "VERA code changed: expected $CODE_COMMIT, head=$head, tracked_dirty=$([[ -n "$dirty" ]] && echo true || echo false)" >&2
    exit 1
  fi
}

verify_code_commit

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
  verify_code_commit
  "$PYTHON" scripts/reproduction_baselines/vera_adapter.py select \
    --corpus "$corpus" --out-dir "$selection"
  selection_sha256="$(sha256sum "$selection/selected_prompt.json" | cut -d' ' -f1)"

  wait_for_vera_gpu
  verify_code_commit
  test "$selection_sha256" = "$(sha256sum "$selection/selected_prompt.json" | cut -d' ' -f1)"
  "$PYTHON" scripts/reproduction_baselines/vera_adapter.py infer \
    --corpus "$corpus" --split test --out-dir "$raw" \
    --prompt-json "$selection/selected_prompt.json"
  "$PYTHON" scripts/reproduction_baselines/vera_adapter.py postprocess \
    --corpus "$corpus" --split test --raw-dir "$raw" \
    --out "$final/scores.jsonl"
  "$PYTHON" scripts/reproduction_baselines/eval_baseline_scores.py \
    --corpus "$corpus" --split test --scores "$final/scores.jsonl" \
    --require-full-coverage --json-out "$final/frame_eval.json"

  verify_code_commit
  test "$selection_sha256" = "$(sha256sum "$selection/selected_prompt.json" | cut -d' ' -f1)"
  "$PYTHON" - "$selection/selected_prompt.json" "$final/frozen_config.json" "$CODE_COMMIT" "$selection_sha256" <<'PY'
import json
import sys
from pathlib import Path

source, out = map(Path, sys.argv[1:3])
code_commit = sys.argv[3]
source_sha256 = sys.argv[4]
selected = json.loads(source.read_text())
payload = {
    "method": "vera",
    "corpus": selected["corpus"],
    "seed": 234,
    "code_commit": code_commit,
    "deterministic_inference": True,
    "selection_split": selected["selection_split"],
    "selection_metric": selected["metric"],
    "selected": selected["selected"],
    "validation_scores": selected["scores"],
    "backbone": selected["backbone"],
    "attention_backend": selected["attention_backend"],
    "source": str(source),
    "source_sha256": source_sha256,
}
temporary = out.with_name(out.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(out)
PY
done

"$PYTHON" scripts/reproduction_baselines/aggregate_official_val.py
