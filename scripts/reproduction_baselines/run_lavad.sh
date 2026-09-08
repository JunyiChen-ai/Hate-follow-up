#!/usr/bin/env bash
# LAVAD is a seven-stage, test-only pipeline. This runner deliberately stops
# after input preparation unless RUN_MODE=full is explicitly set: a full run
# downloads/runs five BLIP-2 captioners, ImageBind twice and Llama-2-13B twice.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORPUS="${CORPUS:-hateclipseg}"
PY="${PY:-/home/jehc223/venvs/SafetyContradiction/bin/python}"
INPUT="${ROOT}/runs/legacy_1fps/lab1/reproduction/lavad_inputs/${CORPUS}"
OUT="${ROOT}/runs/legacy_1fps/lab1/reproduction/baselines/lavad/${CORPUS}"
UP="${ROOT}/third_party/lavad"

cd "$ROOT"
"$PY" scripts/reproduction_baselines/lavad/prepare.py --corpus "$CORPUS"
if [ "${RUN_MODE:-prepare}" != full ]; then
    echo "Prepared LAVAD input. Set RUN_MODE=full only after installing the pinned upstream environment and Llama-2-13B-chat weights; see DESIGN_LAVAD.md."
    exit 0
fi

test "$(git -C "$UP" rev-parse HEAD)" = 1ad46c666d1b3cfb262f3dd84769acf873285056
test -n "${LAVAD_ENV_PY:-}"
test -n "${LLAMA2_CKPT:-}"
echo "The full upstream stages are intentionally issued from DESIGN_LAVAD.md one at a time; no opaque long job is started by this guard runner."
exit 2
