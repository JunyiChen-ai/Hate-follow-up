#!/usr/bin/env bash
# Train, score and evaluate both baselines on all three corpora.
#
# GPU. Runs six trainings one after another, never two at once, so the machine
# holds at most one GPU job at any moment. Detached-friendly:
#
#     cd /home/jehc223/Hate-follow-up
#     setsid nohup bash scripts/reproduction_baselines/run_all.sh \
#         > runs/legacy_1fps/lab1/reproduction/baselines/run_all.log 2>&1 &
#
# Restrict the sweep with the two environment variables:
#     METHODS="vadclip"          CORPORA="hatemm"     bash .../run_all.sh
#
# Every hyperparameter is the published XD-Violence default except the ones
# listed in PATCHES.md patch O1. Nothing is passed here, so the defaults in
# vadclip/option.py and dsanet/option.py are what runs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jehc223/venvs/SafetyContradiction/bin/python}"
METHODS="${METHODS:-vadclip dsanet}"
CORPORA="${CORPORA:-hatemm mhclip_en mhclip_zh}"
OUT_ROOT="${REPO_ROOT}/runs/legacy_1fps/lab1/reproduction/baselines"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}"

for method in ${METHODS}; do
    for corpus in ${CORPORA}; do
        out="${OUT_ROOT}/${method}/${corpus}"
        mkdir -p "${out}"
        echo "=== ${method} / ${corpus} : train ==="
        "${PYTHON}" scripts/reproduction_baselines/train_${method}_hatemm.py \
            --corpus "${corpus}" --device cuda 2>&1 | tee "${out}/train.log"

        echo "=== ${method} / ${corpus} : score ==="
        "${PYTHON}" scripts/reproduction_baselines/test_${method}_hatemm.py \
            --corpus "${corpus}" --device cuda 2>&1 | tee "${out}/infer.log"

        echo "=== ${method} / ${corpus} : evaluate ==="
        "${PYTHON}" scripts/reproduction_baselines/eval_baseline_scores.py \
            --corpus "${corpus}" --scores "${out}/scores.jsonl" \
            --json-out "${out}/frame_eval.json" 2>&1 | tee "${out}/eval.log"
    done
done

echo "=== done ==="
