#!/usr/bin/env bash
# TAD run. Usage: bash experiments/20260912_tad/launch/run_tad.sh <run_name> [extra args]
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN="${1:?run name}"; shift || true
PY="${TAD_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
mkdir -p "runs/20260912_tad/${RUN}"
"$PY" experiments/20260912_tad/tad.py --run-name "$RUN" "$@"
echo TAD_RUN_DONE
