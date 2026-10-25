#!/usr/bin/env bash
#SBATCH --job-name=c50-stage1
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/jehc223/Hate-follow-up
#SBATCH --output=runs/20261025_m1_act_state/slurm_%j.out
# Candidate 50, stage 1 reads on uoa-lab1: arms base, all_local, A1, A1_other on the full HateMM (215) and HateClipSeg
# (118) test splits (measure.py). Only after stage 0 returned CONTINUE and the rule-6 code review passed.
# Submit from the repo root on uoa-lab1:
#   mkdir -p runs/20261025_m1_act_state && sbatch experiments/20261025_m1_act_state/launch/lab1_stage1.sbatch
# Stage 2 (only after stage 1 PASS) uses the same script:
#   RUN=stage2 DATASETS=DeHate ARMS="all_local A1" sbatch experiments/20261025_m1_act_state/launch/lab1_stage1.sbatch
# The CPU analysis (stage1.py: decode, evaluator, gates) runs afterwards outside the GPU job (README "How to run").
# Output: runs/20261025_m1_act_state/$RUN/{run.log (first line hostname), run.pid, config.json, checks.jsonl,
#         <arm>/predictions.jsonl}; Slurm log runs/20261025_m1_act_state/slurm_<job>.out.
# Markers: C50_READS_DONE at the end; C50_READS FAILED on any error (in-job checks raise CHECK_FAILED).
set -euo pipefail
trap 'echo "C50_READS FAILED (line $LINENO)"' ERR
cd /home/jehc223/Hate-follow-up
RUN=${RUN:-stage1}
DATASETS=${DATASETS:-HateMM HateClipSeg}
ARMS=${ARMS:-base all_local A1 A1_other}
export PATH=/home/jehc223/Hate-follow-up/.cache/envs/HateVLM/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/home/jehc223/Hate-follow-up/.cache/triton CUDA_CACHE_PATH=/home/jehc223/Hate-follow-up/.cache/cuda
export PYTORCH_ALLOC_CONF=expandable_segments:True
hostname; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
python -c "import torch, transformers; assert torch.cuda.is_available(), 'no CUDA in the job'; print(torch.cuda.get_device_name(0), torch.__version__, transformers.__version__)"
# shellcheck disable=SC2086
python -u experiments/20261025_m1_act_state/measure.py --run-name "$RUN" --datasets $DATASETS --arms $ARMS
grep -q "RUN_DONE" "runs/20261025_m1_act_state/$RUN/run.log"
echo C50_READS_DONE "$RUN"
