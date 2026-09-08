#!/usr/bin/env bash
# Vad-R1 zero-shot inference and scoring on all three test splits.
#
# One GPU, one vLLM engine at a time, three corpora strictly one after another.
# Detached-friendly:
#
#     cd /home/jehc223/Hate-follow-up
#     setsid nohup bash scripts/reproduction_baselines/run_all_vadr1.sh \
#         > runs/legacy_1fps/lab1/reproduction/baselines/run_all_vadr1.log 2>&1 &
#
# Restrict the sweep with CORPORA:
#     CORPORA="hatemm" bash scripts/reproduction_baselines/run_all_vadr1.sh
#
# ARM defaults to the released prompt, verbatim, which is the zero-shot result.
# ARM=hateful runs the term-adaptation ablation described in
# vadr1/run_vadr1_inference.py; it is a second condition on the same test split
# and needs owner approval before it is run.
#
# Inference is resumable: each corpus appends to its own generations.jsonl and
# skips video ids already present, so a killed run is restarted by rerunning
# this script.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jehc223/venvs/SafetyContradiction/bin/python}"
MODEL="${MODEL:-/home/jehc223/data/checkpoints/vad_r1}"
CORPORA="${CORPORA:-hatemm mhclip_en mhclip_zh}"
ARM="${ARM:-anomaly}"
VADR1="${REPO_ROOT}/scripts/reproduction_baselines/vadr1"

cd "${REPO_ROOT}"

echo "=== Vad-R1 preflight ==="
"${PYTHON}" "${VADR1}/run_vadr1_inference.py" --selftest
"${PYTHON}" "${VADR1}/rasterize_and_eval.py" --selftest

for corpus in ${CORPORA}; do
    echo ""
    echo "=== ${corpus}: inference (arm ${ARM}) ==="
    "${PYTHON}" "${VADR1}/run_vadr1_inference.py" \
        --corpus "${corpus}" --model "${MODEL}" --arm "${ARM}"

    echo ""
    echo "=== ${corpus}: rasterise and score ==="
    "${PYTHON}" "${VADR1}/rasterize_and_eval.py" \
        --corpus "${corpus}" --arm "${ARM}"
done

echo ""
echo "=== done ==="
