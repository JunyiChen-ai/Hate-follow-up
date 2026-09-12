#!/usr/bin/env bash
# NGA run. Usage: bash experiments/20260912_nga/launch/run_nga.sh <run_name> [extra args]
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN="${1:?run name}"; shift || true
PY="${NGA_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
mkdir -p "runs/20260912_nga/${RUN}"
"$PY" experiments/20260912_nga/nga.py --run-name "$RUN" "$@"
echo NGA_RUN_DONE
