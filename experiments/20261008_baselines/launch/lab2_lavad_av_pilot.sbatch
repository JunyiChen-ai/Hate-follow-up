#!/usr/bin/env bash
#SBATCH --job-name=bl-lavad-av-pilot
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261008_baselines/lavad_av/slurm_%j.out
# LAVAD-AV refusal pilot (2026-10-10). Attempt 1 (prompt version "rules") failed the F2 rule on HateMM (Slurm 366:
# Llama-2 refused every centre of 11 of 215 videos). This job scores 17 HateMM videos (the 11 all-refused ones and the
# first 6 other cohort ids in sorted order) with the candidate prompt versions and writes refusal counts only
# (runs/20261008_baselines/lavad_av/pilot/<version>/prompt_stats_HateMM.json). No curve, no evaluation, no label.
# Submit on uoa-lab2: cd ~/Hate-follow-up && sbatch experiments/20261008_baselines/launch/lab2_lavad_av_pilot.sbatch [VERSION ...]
set -uo pipefail
cd /home/jehc223/Hate-follow-up
source experiments/20261008_baselines/launch/lab2_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
IDS=hate_video_102,hate_video_181,hate_video_198,hate_video_205,hate_video_247,hate_video_25,hate_video_284,hate_video_318,hate_video_330,hate_video_88,non_hate_video_314,hate_video_1,hate_video_10,hate_video_101,hate_video_114,hate_video_12,hate_video_132
for V in ${@:-moderator moderator_prefill}; do
  $LAVAD_PYTHON experiments/20261008_baselines/lavad/lavad_av.py score --datasets HateMM --prompt-version $V \
    --pilot-ids $IDS || echo "FAILED pilot $V"
  echo "PILOT_DONE $V $(date -Is)"
done
echo "JOB_DONE $(date -Is)"
