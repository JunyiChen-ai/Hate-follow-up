#!/usr/bin/env bash
# PWC E0 kill test (README section 6). Usage:
#   cd ~/Hate-follow-up && setsid nohup bash experiments/20260912_pwc/launch/run_e0.sh e0 \
#        > runs/20260912_pwc/launch_e0.out 2>&1 &
# Needs runs/20260910_spvl/full2_dual_evid_stance/predictions.jsonl (pair selection) and the
# data/ caches (frames_k20, asr_whisper_large_v3, gt_4fps).
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN="${1:-e0}"; shift || true
PY="${PWC_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
mkdir -p "runs/20260912_pwc/${RUN}"
"$PY" experiments/20260912_pwc/pilot0.py --run-name "$RUN" "$@"
echo PWC_E0_DONE
