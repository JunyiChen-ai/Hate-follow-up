#!/usr/bin/env bash
#SBATCH --job-name=backbones
#SBATCH --partition=local-sc474398,local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261101_backbones/slurm_%j.out
# Backbone study with the adopted reader (experiments/20261101_backbones/README.md) on uoa-lab3 (sc474398) or uoa-lab2 (sc474399), whichever frees first (same repo and env paths on both).
#   TAGS="q3vl-8b q3vl-2b" SCOPE=smoke sbatch experiments/20261101_backbones/launch/lab23.sbatch  (porting checks, 2+2 videos per backbone)
#   TAG=q3vl-4b DATASET=HateMM sbatch experiments/20261101_backbones/launch/lab23.sbatch          (one backbone x one corpus; ARMS="no_transcript ..." for the reader arms)
# Weights: the repo HF cache (.cache/hf/hub; models copied or symlinked from ~/.cache/huggingface/hub), offline; a job exits FAILED_no_weights if the node lacks them.
set -euo pipefail
cd /home/jehc223/Hate-follow-up
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/home/jehc223/Hate-follow-up/.cache/triton CUDA_CACHE_PATH=/home/jehc223/Hate-follow-up/.cache/cuda
export PYTORCH_ALLOC_CONF=expandable_segments:True
hostname; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
VLM=/home/jehc223/miniconda3/envs/HateVLM/bin/python
declare -A HFDIR=([q3vl-2b]=Qwen--Qwen3-VL-2B-Instruct [q3vl-4b]=Qwen--Qwen3-VL-4B-Instruct [q3vl-8b]=Qwen--Qwen3-VL-8B-Instruct
  [q3vl-32b]=Qwen--Qwen3-VL-32B-Instruct [q25vl-7b]=Qwen--Qwen2.5-VL-7B-Instruct [internvl35-8b]=OpenGVLab--InternVL3_5-8B-HF
  [llava-ov-7b]=llava-hf--llava-onevision-qwen2-7b-ov-hf [gemma3-12b]=google--gemma-3-12b-it)
for t in ${TAGS:-} ${TAG:-}; do
  ls -d "$HF_HOME/hub/models--${HFDIR[$t]}/snapshots/"* > /dev/null 2>&1 || { echo "FAILED_no_weights $t on $(hostname)"; exit 3; }
done
case ${SCOPE:-run} in
  smoke)
    for t in ${TAGS:?set TAGS}; do $VLM -u experiments/20261101_backbones/read.py --tag "$t" --smoke || echo "FAILED_smoke $t"; done;;
  run)
    $VLM -u experiments/20261101_backbones/read.py --tag "${TAG:?set TAG}" --dataset "${DATASET:?set DATASET}" --arms ${ARMS:-main};;
  *) echo FAILED_invalid_scope; exit 2;;
esac
echo BACKBONES_DONE
