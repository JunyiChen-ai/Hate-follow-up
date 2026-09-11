#!/usr/bin/env bash
# All declared TAD compose variants for one run dir, through the shared evaluator.
# Usage: bash experiments/20260912_tad/launch/compose_all.sh runs/20260912_tad/<run>
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN="${1:?run dir}"
PY="${TAD_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
C=experiments/20260912_tad/tad_compose.py
"$PY" $C --run-dir "$RUN" --curve act                    # = SPVL-r2 baseline row
"$PY" $C --run-dir "$RUN" --curve topic                  # validity: what the topic read alone orders
"$PY" $C --run-dir "$RUN" --curve corrected --beta ols
"$PY" $C --run-dir "$RUN" --curve corrected --beta 1.0
"$PY" $C --run-dir "$RUN" --curve corrected --beta 0.5
"$PY" $C --run-dir "$RUN" --curve permuted  --beta ols   # control: another video's topic curve
echo TAD_COMPOSE_DONE
