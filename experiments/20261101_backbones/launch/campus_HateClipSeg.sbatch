#!/usr/bin/env bash
#SBATCH --job-name=backbones
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=runs/20261101_backbones/slurm_%j.out
# Backbone study (experiments/20261101_backbones/README.md) on a campus A100 80G: one backbone x HateClipSeg per job,
# the main reads first, then the reader arms (arm-major, resumable). Default backbone q3vl-32b (66 GB in bf16).
#   cd /data/jehc223/Hate-follow-up && sbatch experiments/20261101_backbones/launch/campus_HateClipSeg.sbatch
#   TAG=<tag> ARMS="main ..." sbatch ...   (other backbone / arms)
set -eo pipefail
source /data/jehc223/miniconda3/bin/activate HateVLM
set -u
cd /data/jehc223/Hate-follow-up
export HF_HOME=/data/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
TAG=${TAG:-q3vl-32b}
ARMS=${ARMS:-main no_transcript no_prefix_frames no_stance asr_windows joint}
echo "host $(hostname) job $SLURM_JOB_ID tag $TAG dataset HateClipSeg arms $ARMS"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
python -u experiments/20261101_backbones/read.py --tag "$TAG" --dataset HateClipSeg --arms $ARMS
echo BACKBONES_DONE
