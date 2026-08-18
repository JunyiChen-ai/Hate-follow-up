#!/usr/bin/env bash
# Train, score and evaluate MACIL-SD and its two uni-modal ablations on all
# three corpora.
#
# Separate from run_all.sh on purpose: that script drives the VadCLIP and
# DSANet runs and is not touched here.
#
# GPU. Nine trainings, strictly one after another, so the machine holds at most
# one GPU job at any moment. Detached-friendly:
#
#     cd /home/jehc223/Hate-follow-up
#     setsid nohup bash scripts/reproduction_baselines/run_all_macilsd.sh \
#         > results/reproduction/baselines/run_all_macilsd.log 2>&1 &
#
# Restrict the sweep with the two environment variables:
#     MODALITIES="av"  CORPORA="hatemm"  bash .../run_all_macilsd.sh
#
# MODALITIES
#     av      MACIL-SD proper: the audio-visual model plus its self-distilled
#             uni-modal partner. Writes to baselines/macilsd/<corpus>/.
#     audio   the owner-requested pure-audio row. Upstream's own Single_Model
#             trained on VGGish alone, at the lr/5 upstream trains that module
#             at. Writes to baselines/macilsd_audio/<corpus>/.
#     visual  the same network on I3D alone, so the audio-only number can be
#             read against a matched visual-only number rather than against the
#             audio-visual one. Writes to baselines/macilsd_visual/<corpus>/.
#
# Every hyperparameter is the published default except the ones listed in
# PATCHES.md patch O2. Nothing is passed here, so macilsd/option.py's defaults
# are what runs -- including --grid snippet, the alignment macilsd/align.py
# argues for.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jehc223/venvs/SafetyContradiction/bin/python}"
MODALITIES="${MODALITIES:-av audio visual}"
CORPORA="${CORPORA:-hatemm mhclip_en mhclip_zh}"
OUT_ROOT="${REPO_ROOT}/results/reproduction/baselines"

method_of () {
    case "$1" in
        av)     echo "macilsd" ;;
        audio)  echo "macilsd_audio" ;;
        visual) echo "macilsd_visual" ;;
        *)      echo "unknown modality $1" >&2; exit 1 ;;
    esac
}

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}"

for modality in ${MODALITIES}; do
    method="$(method_of "${modality}")"
    for corpus in ${CORPORA}; do
        out="${OUT_ROOT}/${method}/${corpus}"
        mkdir -p "${out}"
        echo "=== ${method} / ${corpus} : train ==="
        "${PYTHON}" scripts/reproduction_baselines/train_macilsd_hatemm.py \
            --corpus "${corpus}" --modality "${modality}" --device cuda \
            2>&1 | tee "${out}/train.log"

        echo "=== ${method} / ${corpus} : score ==="
        "${PYTHON}" scripts/reproduction_baselines/test_macilsd_hatemm.py \
            --corpus "${corpus}" --modality "${modality}" --device cuda \
            2>&1 | tee "${out}/infer.log"

        echo "=== ${method} / ${corpus} : evaluate ==="
        "${PYTHON}" scripts/reproduction_baselines/eval_baseline_scores.py \
            --corpus "${corpus}" --scores "${out}/scores.jsonl" \
            --json-out "${out}/frame_eval.json" 2>&1 | tee "${out}/eval.log"
    done
done

echo "=== done ==="
