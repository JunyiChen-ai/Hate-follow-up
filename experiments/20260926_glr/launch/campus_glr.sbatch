#!/usr/bin/env bash
#SBATCH --job-name=glr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=runs/20260926_glr/slurm_%j.out
# GLR measurement pass on campus. Submit: cd /data/jehc223/Hate-follow-up && mkdir -p runs/20260926_glr && sbatch experiments/20260926_glr/launch/campus_glr.sbatch
set -eo pipefail
source /data/jehc223/miniconda3/bin/activate HateVLM
set -u
cd /data/jehc223/Hate-follow-up
export HF_HOME=/data/jehc223/Hate-follow-up/.cache/hf
export GLR_PYTHON=/data/jehc223/miniconda3/envs/HateVLM/bin/python
echo "host $(hostname) job $SLURM_JOB_ID gpu ${CUDA_VISIBLE_DEVICES:-?}"
bash experiments/20260926_glr/launch/run_glr.sh
echo "SBATCH_DONE glr $(date -Is)"
