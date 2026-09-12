#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN="${1:?run name}"; shift || true
PY="${BND_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
mkdir -p "runs/20260912_bnd/${RUN}"
"$PY" experiments/20260912_bnd/bnd.py --run-name "$RUN" "$@"
echo BND_RUN_DONE
