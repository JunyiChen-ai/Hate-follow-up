#!/usr/bin/env bash
# Family / size robustness study (README §11): run the SPVL-r2 arms with another MLLM.
# Usage: bash experiments/20260910_spvl/launch/run_mllm.sh <hf_model_id> <model_tag> [arm ...]
#   arms (default: all seven): full winonly noctx noframes asr nostance joint
# Runs go to runs/20260910_spvl/mllm/<model_tag>/<arm>/ ; each arm is composed three ways
# (zv_plus_mean = SPVL-r2, spvl = no M3, and for `full` also intercept-only / residual-only).
set -euo pipefail
cd "$(dirname "$0")/../../.."
MODEL_ID="$1"; TAG="$2"; shift 2
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(full winonly noctx noframes asr nostance joint)
PY="${SPVL_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
BASE="runs/20260910_spvl/mllm/$TAG"; mkdir -p "$BASE"
LOG="$BASE/launch.log"
echo "host $(hostname) date $(date -Is) commit $(git rev-parse --short HEAD) model $MODEL_ID tag $TAG arms ${ARMS[*]}" | tee -a "$LOG"
COMMON=(--model "$MODEL_ID" --model-tag "$TAG" --isolation cache --frames 20 --windows fixed --window-seconds 8)
FULL=(--branches dual --window-question evidence --stance verdict)
arm_args() {
  case "$1" in
    full)     echo "${FULL[@]}" ;;
    winonly)  echo "--frames 0 --no-transcript-context --branches joint --window-question rules" ;;
    noctx)    echo "${FULL[@]} --no-transcript-context" ;;
    noframes) echo "${FULL[@]} --frames 0" ;;
    asr)      echo "${FULL[@]} --windows asr" ;;
    nostance) echo "--branches dual --window-question evidence" ;;
    joint)    echo "--branches joint --window-question evidence --stance verdict" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}
for ARM in "${ARMS[@]}"; do
  OUT="$BASE/$ARM"; mkdir -p "$OUT"
  echo "ARM_START $ARM $(date -Is)" | tee -a "$LOG"
  # later flags override earlier ones (argparse), so arm-specific --frames 0 wins over the common --frames 20
  "$PY" experiments/20260910_spvl/spvl.py --run-name "$ARM" "${COMMON[@]}" $(arm_args "$ARM") 2>&1 | tee -a "$OUT/launch.log" | grep -E "VERIFY|progress|DONE|FAILED|OOM|Traceback|Error" || true
  "$PY" experiments/20260910_spvl/compose.py --run-dir "$OUT" --intercept zv_plus_mean --residual rank 2>&1 | tee -a "$OUT/launch.log" | tail -3
  "$PY" experiments/20260910_spvl/compose.py --run-dir "$OUT" --intercept spvl --residual rank 2>&1 | tee -a "$OUT/launch.log" | tail -3
  if [ "$ARM" = full ]; then
    "$PY" experiments/20260910_spvl/compose.py --run-dir "$OUT" --intercept spvl --residual none 2>&1 | tee -a "$OUT/launch.log" | tail -3
    "$PY" experiments/20260910_spvl/compose.py --run-dir "$OUT" --intercept none --residual rank 2>&1 | tee -a "$OUT/launch.log" | tail -3
  fi
  echo "ARM_DONE $ARM $(date -Is)" | tee -a "$LOG"
done
echo "MODEL_DONE $TAG $(date -Is)" | tee -a "$LOG"
