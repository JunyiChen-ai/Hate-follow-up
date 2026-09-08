#!/usr/bin/env bash
# Train, score and evaluate the MultiHateLoc reimplementation on all three
# corpora, one after another, one GPU job at a time.
#
#     cd /home/jehc223/Hate-follow-up
#     setsid nohup bash scripts/reproduction_baselines/multihateloc/run_all.sh \
#         > runs/legacy_1fps/lab1/reproduction/baselines/multihateloc_reimpl/run_all.log 2>&1 &
#
# Restrict with:  CORPORA="hatemm" bash .../run_all.sh
#
# Nothing is passed to train.py, so its defaults run: the paper's published
# settings (Adam 1e-4, batch 32, 100 epochs, K=3, lambda 0.1 / 0.2) plus this
# port's protocol (seed 234, 10 % stratified validation carve, selection on
# validation video AP, test split never opened during training).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/home/jehc223/venvs/SafetyContradiction/bin/python}"
CORPORA="${CORPORA:-hatemm mhclip_en mhclip_zh}"
HERE="scripts/reproduction_baselines/multihateloc"
OUT_ROOT="${REPO_ROOT}/runs/legacy_1fps/lab1/reproduction/baselines/multihateloc_reimpl"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}"
echo "RUNNING $(date)" > "${OUT_ROOT}/STATUS"

for corpus in ${CORPORA}; do
    out="${OUT_ROOT}/${corpus}"
    mkdir -p "${out}"

    echo "=== multihateloc / ${corpus} : train + score  $(date) ==="
    "${PYTHON}" "${HERE}/train.py" --corpus "${corpus}" --device cuda \
        2>&1 | tee "${out}/train.log"

    echo "=== multihateloc / ${corpus} : evaluate ==="
    "${PYTHON}" scripts/reproduction_baselines/eval_baseline_scores.py \
        --corpus "${corpus}" --scores "${out}/scores.jsonl" \
        --json-out "${out}/frame_eval.json" 2>&1 | tee "${out}/eval.log"

    "${PYTHON}" "${HERE}/video_auc.py" --corpus "${corpus}" \
        --scores "${out}/scores.jsonl" \
        --json-out "${out}/video_auc.json" 2>&1 | tee "${out}/video_auc.log"
done

echo "DONE $(date)" > "${OUT_ROOT}/STATUS"
touch "${OUT_ROOT}/DONE"
echo "=== done $(date) ==="
