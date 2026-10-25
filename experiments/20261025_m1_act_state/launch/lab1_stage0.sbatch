#!/usr/bin/env bash
#SBATCH --job-name=c50-stage0
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/jehc223/Hate-follow-up
#SBATCH --output=runs/20261025_m1_act_state/slurm_%j.out
# Candidate 50, stage 0 (GT-free) on uoa-lab1: arms all_local, A1, A1_other on the first 20 videos by sorted id of
# HateMM, HateClipSeg and DeHate (measure.py), then the three stop statistics (stage0.py, no GT).
# Submit from the repo root on uoa-lab1:
#   mkdir -p runs/20261025_m1_act_state && sbatch experiments/20261025_m1_act_state/launch/lab1_stage0.sbatch
# Optional: RUN=<run name> (default stage0).
# Output: runs/20261025_m1_act_state/$RUN/{run.log (first line hostname), run.pid, config.json, checks.jsonl,
#         <arm>/predictions.jsonl, stage0_summary.json}; Slurm log runs/20261025_m1_act_state/slurm_<job>.out.
# Markers: C50_STAGE0_DONE at the end; C50_STAGE0 FAILED on any error (in-job checks raise CHECK_FAILED).
set -euo pipefail
trap 'echo "C50_STAGE0 FAILED (line $LINENO)"' ERR
cd /home/jehc223/Hate-follow-up
RUN=${RUN:-stage0}
export PATH=/home/jehc223/Hate-follow-up/.cache/envs/HateVLM/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/home/jehc223/Hate-follow-up/.cache/triton CUDA_CACHE_PATH=/home/jehc223/Hate-follow-up/.cache/cuda
export PYTORCH_ALLOC_CONF=expandable_segments:True
hostname; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
python -c "import torch, transformers; assert torch.cuda.is_available(), 'no CUDA in the job'; print(torch.cuda.get_device_name(0), torch.__version__, transformers.__version__)"
python -u experiments/20261025_m1_act_state/measure.py --run-name "$RUN" --datasets HateMM HateClipSeg DeHate \
  --first 20 --arms all_local A1 A1_other
grep -q "RUN_DONE" "runs/20261025_m1_act_state/$RUN/run.log"
python -u experiments/20261025_m1_act_state/stage0.py --reads "runs/20261025_m1_act_state/$RUN" \
  --datasets HateMM HateClipSeg DeHate --first 20
echo C50_STAGE0_DONE "$RUN"
