#!/usr/bin/env bash
#SBATCH --job-name=c51-read
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/jehc223/Hate-follow-up
#SBATCH --output=runs/20261026_knowledge_verdict/slurm_%j.out
# Candidate 51 reader on uoa-lab1: the r6 read with the agent's knowledge, arms of README §3, on the 40-video subset
# of each corpus (common.subset_rows). Needs runs/20261026_knowledge_verdict/agent/<dataset>/<video_id>.json from
# agent_retrieve.py (uoa-lab2, rsynced back). Submit from the repo root on uoa-lab1:
#   mkdir -p runs/20261026_knowledge_verdict && sbatch experiments/20261026_knowledge_verdict/launch/lab1_read.sbatch
# Optional: ARMS="base K ..." (default all nine), LIMIT=<n> (first n subset videos per corpus, smoke test; needs
# RUN other than "read"), RUN=<name>.
# Output: runs/20261026_knowledge_verdict/$RUN/{run.log, run.pid, config.json}, $RUN_<arm>/predictions.jsonl.
# Markers: C51_READ_DONE at the end; C51_READ FAILED on any error.
set -euo pipefail
trap 'echo "C51_READ FAILED (line $LINENO)"' ERR
cd /home/jehc223/Hate-follow-up
RUN=${RUN:-read}
ARMS=${ARMS:-base K Kp KJ J J0 Kother Kshift Kmark}
LIMIT=${LIMIT:-0}
export PATH=/home/jehc223/Hate-follow-up/.cache/envs/HateVLM/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/home/jehc223/Hate-follow-up/.cache/triton CUDA_CACHE_PATH=/home/jehc223/Hate-follow-up/.cache/cuda
export PYTORCH_ALLOC_CONF=expandable_segments:True
hostname; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
python -c "import torch, transformers; assert torch.cuda.is_available(), 'no CUDA in the job'; print(torch.cuda.get_device_name(0), torch.__version__, transformers.__version__)"
python -u experiments/20261026_knowledge_verdict/kv_measure.py --run-name "$RUN" --arms $ARMS --limit "$LIMIT" \
  --datasets HateMM HateClipSeg DeHate
grep -q "DONE videos=" "runs/20261026_knowledge_verdict/$RUN/run.log"
echo C51_READ_DONE "$RUN"
