#!/bin/bash
# Weakly supervised baselines (README in each method directory). Submit from the repo root of the target machine:
#   cd ~/Hate-follow-up && mkdir -p runs/20261008_baselines/slurm && sbatch experiments/20261008_baselines/launch/lab3_dsanet_vadclip.sbatch
# Tasks run one after another (weaksup_common/run_tasks.sh); a finished seed is kept on resubmission.
#SBATCH --job-name=bl-ws-lab3
#SBATCH --partition=local-sc474398
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261008_baselines/slurm/lab3_dsanet_vadclip_%j.out
set -uo pipefail
cd "$HOME/Hate-follow-up"
source "$HOME/miniconda3/bin/activate" HateVideo
export HF_HOME="$PWD/.cache/hf" TORCH_HOME="$PWD/.cache/torch"
hostname
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
bash experiments/20261008_baselines/weaksup_common/run_tasks.sh method:dsanet:hateclipseg method:dsanet:hatemm method:vadclip:hateclipseg method:vadclip:hatemm method:vadclip:dehate
