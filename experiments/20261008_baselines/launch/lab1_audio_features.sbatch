#!/bin/bash
# wav2vec2-base and Wav2CLIP 1 fps audio features for the weakly supervised baselines (all splits, three corpora).
# uoa-lab1 only (the 16 kHz wavs are in the read-only Retrieval-hate checkout there).
#   cd ~/Hate-follow-up && mkdir -p runs/20261008_baselines/audio_features && \
#   sbatch experiments/20261008_baselines/launch/lab1_audio_features.sbatch
#SBATCH --job-name=bl-audiofeat
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261008_baselines/audio_features/slurm_%j.out
set -euo pipefail
cd "$HOME/Hate-follow-up"
source "$HOME/miniconda3/bin/activate" HateVideo
export HF_HOME="$PWD/.cache/hf" TORCH_HOME="$PWD/.cache/torch"
hostname
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python -u experiments/20261008_baselines/weaksup_common/extract_audio.py --corpora hatemm hateclipseg dehate
echo "DONE $(date -Is)"
