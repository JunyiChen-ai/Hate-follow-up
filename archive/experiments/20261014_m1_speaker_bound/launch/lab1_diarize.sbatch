#!/bin/bash
#SBATCH --job-name=c44_diar
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261014_m1_speaker_bound/slurm_%j.out
# Candidate 44 pre-check input: speaker diarization of the test audio (uoa-lab1). Usage:
#   sbatch experiments/20261014_m1_speaker_bound/launch/lab1_diarize.sbatch            # both corpora
#   sbatch experiments/20261014_m1_speaker_bound/launch/lab1_diarize.sbatch --limit 2  # smoke
set -euo pipefail
cd ~/Hate-follow-up
export PATH=$PWD/.cache/envs/diar/bin:$PATH
OUT=runs/20261014_m1_speaker_bound/diarize
mkdir -p "$OUT"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} commit $(git rev-parse --short HEAD) $(date -Is)" | tee "$OUT/run.log"
echo $$ > "$OUT/run.pid"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$OUT/run.log"
python experiments/20261014_m1_speaker_bound/diarize.py "$@" 2>&1 | tee -a "$OUT/run.log"
echo C44_DIAR_DONE | tee -a "$OUT/run.log"
