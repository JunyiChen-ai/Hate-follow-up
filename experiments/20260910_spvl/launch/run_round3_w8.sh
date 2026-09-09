#!/usr/bin/env bash
# Round 3: per-window frames (data/frames_w8) so every window has a frame for the visual branch.
set -euo pipefail
cd "$(dirname "$0")/../../.."
L=experiments/20260910_spvl/launch/run_spvl.sh
B="--frames 20 --frame-source w8 --windows fixed --window-seconds 8"
bash $L full3_dual_evid_stance_w8  $B --branches dual  --window-question evidence --stance verdict
bash $L abl3_joint_evid_stance_w8  $B --branches joint --window-question evidence --stance verdict
bash $L abl3_joint_rules_none_w8   $B --branches joint --window-question rules
echo ROUND3_DONE
