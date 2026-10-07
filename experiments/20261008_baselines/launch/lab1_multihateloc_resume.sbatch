#!/bin/bash
# Weakly supervised baselines (README in each method directory). Submit from the repo root of the target machine:
#   cd ~/Hate-follow-up && mkdir -p runs/20261008_baselines/slurm && sbatch experiments/20261008_baselines/launch/lab1_multihateloc_resume.sbatch
# Resumes the MultiHateLoc Optuna studies of Slurm 291 (HateMM stopped at 4/5 completed trials after 10 attempts).
# Tasks run one after another (weaksup_common/run_tasks.sh); a finished seed is kept on resubmission.
#SBATCH --job-name=bl-mhl-lab1
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261008_baselines/slurm/lab1_multihateloc_resume_%j.out
set -uo pipefail
cd "$HOME/Hate-follow-up"
source "$HOME/miniconda3/bin/activate" HateVideo
export HF_HOME="$PWD/.cache/hf" TORCH_HOME="$PWD/.cache/torch"
hostname
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
bash experiments/20261008_baselines/weaksup_common/run_tasks.sh method:multihateloc:hatemm method:multihateloc:hateclipseg
