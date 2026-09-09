#!/usr/bin/env bash
# Round-2 pilot (stance conditioning, modality branches, evidence question) on the within-defined subset.
set -euo pipefail
cd "$(dirname "$0")/../../.."
L=experiments/20260910_spvl/launch/run_spvl.sh
B="--only-within-defined --frames 20 --windows fixed --window-seconds 8"
bash $L pilot2_q1_joint_evid          $B --branches joint  --window-question evidence
bash $L pilot2_q2_joint_rules_stance  $B --branches joint  --window-question rules    --stance verdict
bash $L pilot2_q3_joint_evid_stance   $B --branches joint  --window-question evidence --stance verdict
bash $L pilot2_q4_dual_rules          $B --branches dual   --window-question rules
bash $L pilot2_q5_dual_evid_stance    $B --branches dual   --window-question evidence --stance verdict
bash $L pilot2_q6_triple_evid_stance  $B --branches triple --window-question evidence --stance verdict
echo PILOT2_DONE
