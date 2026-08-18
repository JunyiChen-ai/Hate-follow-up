#!/usr/bin/env bash
# EventVAD (ACM MM 2025) on all three test splits, training free.
#
# One GPU, one model resident at a time, three corpora strictly one after
# another. Detached-friendly:
#
#     cd /home/jehc223/Hate-follow-up
#     setsid nohup bash scripts/reproduction_baselines/run_all_eventvad.sh \
#         > results/reproduction/baselines/run_all_eventvad.log 2>&1 &
#
# Restrict the sweep with CORPORA, and pick the stage with STAGES:
#     CORPORA="mhclip_en" STAGES="segment" bash .../run_all_eventvad.sh
#
# The two GPU stages are separate on purpose. Stage 1 (segment) holds CLIP
# ViT-B/16 and RAFT, about 0.3 GB, and is bound by RAFT's per-frame-pair
# forward. Stage 2 (score) holds VideoLLaMA2.1-7B-16F in fp16, about 16 GB.
# Running them as one process would keep 16 GB pinned through the whole of
# stage 1 for nothing, and splitting them lets stage 1 for the next corpus
# start while stage 2's numbers are being read.
#
# Both stages are resumable: each appends to its own jsonl and skips video ids
# already recorded without an error, so a killed run is restarted by rerunning
# this script. Neither ever deletes; pass --restart to a stage by hand to do
# that.
#
# ARM defaults to `paper`, the Figure 2 prompt reconstruction, which is the
# result to quote. `no_thinking` is the paper's own Table 5 ablation and
# `bounded` states the score range the paper leaves unstated; both are second
# conditions on the same test split and need owner approval before they run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/jehc223/venvs/SafetyContradiction/bin/python}"
MODEL="${MODEL:-/home/jehc223/data/checkpoints/videollama2}"
CORPORA="${CORPORA:-mhclip_zh mhclip_en hatemm}"
STAGES="${STAGES:-segment score eval}"
ARM="${ARM:-paper}"
PRESET="${PRESET:-paper}"
EV="${REPO_ROOT}/scripts/reproduction_baselines/eventvad"

cd "${REPO_ROOT}"

echo "=== EventVAD preflight ==="
CUDA_VISIBLE_DEVICES="" "${PYTHON}" \
    "${REPO_ROOT}/scripts/reproduction_baselines/smoke_cpu_eventvad.py"

# The corpora are ordered shortest-first on purpose: MultiHateClip is 5600 s
# and 4800 s of video against HateMM's 29266 s, so a configuration problem
# surfaces in the first hour rather than the tenth.
for corpus in ${CORPORA}; do
    if [[ " ${STAGES} " == *" segment "* ]]; then
        echo ""
        echo "=== ${corpus}: stage 1, event segmentation (GPU: CLIP + RAFT) ==="
        "${PYTHON}" "${EV}/segment_events.py" \
            --corpus "${corpus}" --device cuda --preset "${PRESET}"
    fi

    if [[ " ${STAGES} " == *" score "* ]]; then
        echo ""
        echo "=== ${corpus}: stage 2, event scoring (GPU: VideoLLaMA2 7B) ==="
        "${PYTHON}" "${EV}/score_events.py" \
            --corpus "${corpus}" --model "${MODEL}" --arm "${ARM}"
    fi

    if [[ " ${STAGES} " == *" eval "* ]]; then
        echo ""
        echo "=== ${corpus}: stage 3, rasterise and score (CPU) ==="
        CUDA_VISIBLE_DEVICES="" "${PYTHON}" "${EV}/rasterize_and_eval.py" \
            --corpus "${corpus}" --arm "${ARM}"
    fi
done

echo ""
echo "=== done ==="
