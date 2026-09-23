#!/usr/bin/env bash
# Ablations of the final composition (single grid, scaled max, duration prior D = 80 s) on cached runs. CPU only.
# Every run is evaluated without the prior (--dwell 0) and with it (--dwell 80). Writes runs/20260922_til/ablation_table.txt.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${TIL_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
I="$PY experiments/20260922_til/til_infer.py --model average --out-root runs/20260922_til/ablation"
M=runs/20260910_spvl/mllm/q3vl-8b
{
echo "# SPVL-r2 components (cache path, same model), without / with the prior"
for pair in "r2_full:$M/full" "nostance:$M/nostance" "joint_branch:$M/joint" "no_transcript:$M/noctx" "no_frames:$M/noframes" "asr_windows:$M/asr" "window_alone:$M/winonly" "one_frame_per_window:runs/20260910_spvl/full3_dual_evid_stance_w8" "tilgridA:runs/20260922_til/gridA"; do
  t=${pair%%:*}; r=${pair#*:}
  $I --runs $r --dwell 0 --tag ${t}__noprior; $I --runs $r --dwell 80 --tag ${t}__prior80
done
echo "# round-1 SPVL grid constants (joint branch, no stance), without / with the prior"
for pair in "r1_full_s8:runs/20260910_spvl/full" "r1_s4:runs/20260910_spvl/abl_s4" "r1_s16:runs/20260910_spvl/abl_s16" "r1_k8frames:runs/20260910_spvl/abl_k8"; do
  t=${pair%%:*}; r=${pair#*:}
  $I --runs $r --dwell 0 --tag ${t}__noprior; $I --runs $r --dwell 80 --tag ${t}__prior80
done
echo "# controls on grid A"
$I --runs runs/20260922_til/gridA --model smooth --dwell 0 --tag gridA__kernel_smooth
$I --runs runs/20260922_til/gridA --dwell 80 --shuffle --tag gridA__prior80_shuffled_order
echo "# cross-model (cache path full runs), without / with the prior"
for m in q3vl-2b q3vl-4b q3vl-32b q25vl-7b internvl35-8b llava-ov-7b gemma3-12b; do
  $I --runs runs/20260910_spvl/mllm/$m/full --dwell 0 --tag ${m}__noprior; $I --runs runs/20260910_spvl/mllm/$m/full --dwell 80 --tag ${m}__prior80
done
echo "# HateClipSeg hate-only GT (secondary)"
$I --runs runs/20260922_til/gridA --dwell 0 --gt-dir data/gt_4fps_hate_only --datasets HateClipSeg --tag hateonly__noprior
$I --runs runs/20260922_til/gridA --dwell 80 --gt-dir data/gt_4fps_hate_only --datasets HateClipSeg --tag hateonly__prior80
} 2>&1 | grep -v Warn | tee runs/20260922_til/ablation_table.txt
echo "ABLATIONS_DONE $(date -Is)"
