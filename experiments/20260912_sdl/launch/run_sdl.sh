#!/usr/bin/env bash
# SDL run. Usage: bash experiments/20260912_sdl/launch/run_sdl.sh <run_name> [extra args]
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN="${1:?run name}"; shift || true
PY="${SDL_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
mkdir -p "runs/20260912_sdl/${RUN}"
"$PY" experiments/20260912_sdl/sdl.py --run-name "$RUN" "$@"
echo SDL_RUN_DONE
