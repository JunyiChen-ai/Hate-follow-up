#!/usr/bin/env bash
# E3 pilot on the within-defined subset (HateMM 84 + HateClipSeg 99 videos). Sequential, one GPU.
set -euo pipefail
cd "$(dirname "$0")/../../.."
L=experiments/20260910_spvl/launch/run_spvl.sh
bash $L pilot_a0_legacy_chunk   --only-within-defined --legacy-chunk-arm
bash $L pilot_a_f0_noctx_asr    --only-within-defined --frames 0 --no-transcript-context --windows asr
bash $L pilot_b_f0_ctx_asr      --only-within-defined --frames 0 --windows asr
bash $L pilot_d_f0_ctx_fixed8   --only-within-defined --frames 0 --windows fixed --window-seconds 8
bash $L pilot_c_f20_ctx_fixed8  --only-within-defined --frames 20 --windows fixed --window-seconds 8
echo PILOT_DONE
