#!/bin/bash
# Weakly supervised baselines (README in each method directory). Submit from the repo root of the target machine:
#   cd ~/Hate-follow-up && mkdir -p runs/20261008_baselines/slurm && sbatch experiments/20261008_baselines/launch/lab1_mil_mhl_avadclip.sbatch
# Tasks run one after another (weaksup_common/run_tasks.sh); a finished seed is kept on resubmission.
#SBATCH --job-name=bl-ws-lab1
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261008_baselines/slurm/lab1_mil_mhl_avadclip_%j.out
set -uo pipefail
cd "$HOME/Hate-follow-up"
source "$HOME/miniconda3/bin/activate" HateVideo
export HF_HOME="$PWD/.cache/hf" TORCH_HOME="$PWD/.cache/torch"
hostname
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
bash experiments/20261008_baselines/weaksup_common/run_tasks.sh mil:bert:hatemm mil:bert:hateclipseg mil:bert:dehate mil:wav2vec2:hatemm mil:wav2vec2:hateclipseg mil:wav2vec2:dehate mil:clip:hatemm mil:clip:hateclipseg mil:clip:dehate method:multihateloc:hatemm method:multihateloc:hateclipseg method:avadclip:hateclipseg method:avadclip:hatemm method:avadclip:dehate
