#!/usr/bin/env bash
#SBATCH --job-name=bl-qwen3-text
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261008_baselines/qwen3_8b_text/slurm_%j.out
# Qwen3-8B on Whisper transcript segments (run_plan.md L2), HateMM, HateClipSeg, DeHate.
set -uo pipefail
cd /home/jehc223/Hate-follow-up
source experiments/20261008_baselines/launch/lab2_env.sh
$PY_VLM experiments/20261008_baselines/qwen3_text/text_llm.py --datasets HateMM HateClipSeg DeHate \
  || echo "FAILED text_llm (see run.log)"
echo "JOB_DONE $(date -Is)"
