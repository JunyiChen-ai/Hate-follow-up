#!/usr/bin/env bash
#SBATCH --job-name=m3-seg
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/jehc223/Hate-follow-up
#SBATCH --output=runs/20261025_m3_coverage/slurm_%j.out
# Per-segment presence reads (read_segments.py) on uoa-lab1. Run from the repo root after mkdir -p runs/20261025_m3_coverage:
#   RUN=<run name> DATASETS="HateMM HateClipSeg" [J=4] [LIMIT=2] [VERIFY=1] sbatch experiments/20261025_m3_coverage/launch/lab1_segments.sbatch
# Smoke: LIMIT=2 VERIFY=1 (2 videos per corpus, z_video and every segment compared with a plain call). DeHate: DATASETS=DeHate.
# Output: runs/20261025_m3_coverage/$RUN/{segments.jsonl, config.json, run.log, run.pid, verify.jsonl}.
set -euo pipefail
cd /home/jehc223/Hate-follow-up
: "${RUN:?set RUN}"
J=${J:-4}
DATASETS=${DATASETS:-HateMM HateClipSeg}
export PATH=/home/jehc223/Hate-follow-up/.cache/envs/HateVLM/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/home/jehc223/Hate-follow-up/.cache/triton CUDA_CACHE_PATH=/home/jehc223/Hate-follow-up/.cache/cuda
export PYTORCH_ALLOC_CONF=expandable_segments:True
hostname; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA in the job'; print(torch.cuda.get_device_name(0))"
# shellcheck disable=SC2086
python -u experiments/20261025_m3_coverage/read_segments.py --run-name "$RUN" --J "$J" --datasets $DATASETS \
  ${LIMIT:+--limit "$LIMIT"} ${VERIFY:+--verify}
grep -q "DONE videos" "runs/20261025_m3_coverage/$RUN/run.log"
echo M3_SEGMENTS_DONE "$RUN"
