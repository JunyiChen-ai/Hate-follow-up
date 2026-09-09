#!/usr/bin/env bash
# E4: full test run (HateMM 215 + HateClipSeg 118) and ablations, sequential on one GPU.
set -euo pipefail
cd "$(dirname "$0")/../../.."
L=experiments/20260910_spvl/launch/run_spvl.sh
PY="${SPVL_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
C=experiments/20260910_spvl/compose.py
bash $L full                 --frames 20 --windows fixed --window-seconds 8
bash $L abl_noctx            --frames 20 --windows fixed --window-seconds 8 --no-transcript-context
bash $L abl_noframes         --frames 0  --windows fixed --window-seconds 8
bash $L abl_noctx_noframes   --frames 0  --windows fixed --window-seconds 8 --no-transcript-context
bash $L abl_causal           --frames 20 --windows fixed --window-seconds 8 --mask causal
bash $L abl_asr_windows      --frames 20 --windows asr
bash $L abl_s4               --frames 20 --windows fixed --window-seconds 4
bash $L abl_s16              --frames 20 --windows fixed --window-seconds 16
bash $L abl_k8               --frames 8  --windows fixed --window-seconds 8
bash $L legacy_chunk_replica --legacy-chunk-arm
# composition ablations on the full run (no new forwards)
"$PY" $C --run-dir runs/20260910_spvl/full --intercept legacy --residual rank
"$PY" $C --run-dir runs/20260910_spvl/full --intercept spvl   --residual none
"$PY" $C --run-dir runs/20260910_spvl/full --intercept none   --residual rank
"$PY" $C --run-dir runs/20260910_spvl/full --intercept spvl   --residual rank --visual-primary
echo ABLATIONS_DONE
