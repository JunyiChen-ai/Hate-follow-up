#!/usr/bin/env bash
# Round-2 full-set runs: final candidate (dual branches + evidence question + stance) and its ablations.
set -euo pipefail
cd "$(dirname "$0")/../../.."
L=experiments/20260910_spvl/launch/run_spvl.sh
B="--frames 20 --windows fixed --window-seconds 8"
bash $L full2_dual_evid_stance   $B --branches dual  --window-question evidence --stance verdict
bash $L abl2_nostance            $B --branches dual  --window-question evidence
bash $L abl2_rulesq              $B --branches dual  --window-question rules    --stance verdict
bash $L abl2_joint               $B --branches joint --window-question evidence --stance verdict
bash $L abl2_triple              $B --branches triple --window-question evidence --stance verdict
echo ROUND2_FULL_DONE
