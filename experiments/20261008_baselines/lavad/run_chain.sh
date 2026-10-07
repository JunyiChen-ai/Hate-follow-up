#!/usr/bin/env bash
# LAVAD chain for one corpus and one id list, in one Slurm job (port of Retrieval-hate
# scripts/repro_campaign/run_lavad_wave1.sh: same stage order, models, batch sizes and prompt).
#   bash experiments/20261008_baselines/lavad/run_chain.sh <DATASET> <IDS_FILE>
# Every stage is idempotent (an existing per-video output is skipped), so re-running resumes.
set -u
cd "$(dirname "$0")/../../.."
PY=${LAVAD_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DS=$1
IDS=$2
L=experiments/20261008_baselines/lavad
echo "=== [$(date -Is)] host $(hostname) dataset $DS ids $IDS ($(wc -l < "$IDS") videos)"
echo "=== [$(date -Is)] stage 01 caption (BLIP-2 blip2-opt-6.7b-coco, 1 fps)"
$PY $L/blip2_caption.py --datasets "$DS" --ids-file "$IDS" --batch-size 48 \
  || { rc=$?; echo "!! caption rc=$rc -- stopping the chain FAILED"; exit "$rc"; }
echo "=== [$(date -Is)] stage 02+03 index + clean captions (ImageBind)"
$PY $L/lavad_chain.py clean --datasets "$DS" --ids-file "$IDS" --center-step 1 \
  || { rc=$?; echo "!! clean rc=$rc -- stopping the chain FAILED"; exit "$rc"; }
echo "=== [$(date -Is)] stage 04a temporal summaries (Llama-2-13b-chat NF4)"
$PY $L/lavad_chain.py summarize --datasets "$DS" --ids-file "$IDS" --center-step 1 --batch-size 48 \
  || { rc=$?; echo "!! summarize rc=$rc -- stopping the chain FAILED"; exit "$rc"; }
echo "=== [$(date -Is)] stage 04b anomaly scores, verbatim prompt"
$PY $L/lavad_chain.py score --datasets "$DS" --ids-file "$IDS" --center-step 1 --batch-size 48 --prompt verbatim \
  || { rc=$?; echo "!! score rc=$rc -- stopping the chain FAILED"; exit "$rc"; }
echo "=== [$(date -Is)] stage 05+06 summary index + refined scores (ImageBind)"
$PY $L/lavad_chain.py refine --datasets "$DS" --ids-file "$IDS" --center-step 1 \
  || { rc=$?; echo "!! refine rc=$rc -- stopping the chain FAILED"; exit "$rc"; }
echo "=== [$(date -Is)] curves"
$PY $L/lavad_chain.py curves --datasets "$DS" --ids-file "$IDS" --center-step 1 \
  || { rc=$?; echo "!! curves rc=$rc -- stopping the chain FAILED"; exit "$rc"; }
echo "=== [$(date -Is)] LAVAD chain finished"
