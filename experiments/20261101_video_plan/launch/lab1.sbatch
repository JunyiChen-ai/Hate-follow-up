#!/usr/bin/env bash
#SBATCH --job-name=video-plan
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261101_video_plan/slurm_%j.out
# Free-form per-video plan in the adopted reader, uoa-lab1 (experiments/20261101_video_plan/README.md):
#   SCOPE=smoke  sbatch experiments/20261101_video_plan/launch/lab1.sbatch   (2 + 2 videos; plan-off reads checked against the stored adopted reads)
#   SCOPE=HateMM sbatch experiments/20261101_video_plan/launch/lab1.sbatch   (one corpus: own_plan then other_plan)
#   SCOPE=HateClipSeg likewise.
set -euo pipefail
cd /home/jehc223/Hate-follow-up
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/home/jehc223/Hate-follow-up/.cache/triton CUDA_CACHE_PATH=/home/jehc223/Hate-follow-up/.cache/cuda
export PYTORCH_ALLOC_CONF=expandable_segments:True
hostname; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
VLM=/home/jehc223/Hate-follow-up/.cache/envs/HateVLM/bin/python
case ${SCOPE:-smoke} in
  smoke) $VLM -u experiments/20261101_video_plan/read_plan.py --smoke;;
  HateMM|HateClipSeg) $VLM -u experiments/20261101_video_plan/read_plan.py --datasets $SCOPE;;
  *) echo FAILED_invalid_scope; exit 2;;
esac
echo VIDEO_PLAN_DONE
