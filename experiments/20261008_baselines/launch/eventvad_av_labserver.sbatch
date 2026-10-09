#!/bin/bash
# EventVAD transcript (audio-visual) variant, scoring stage only, on lab-server (sc448960). The events are the
# visual-only run's (runs/20261008_baselines/eventvad/<DS>/events.jsonl, complete for every listed corpus); the
# scores go to runs/20261008_baselines/eventvad_av/<DS>/. Usage, from the repo root on lab-server:
#   mkdir -p runs/20261008_baselines/eventvad_av
#   sbatch experiments/20261008_baselines/launch/eventvad_av_labserver.sbatch HateMM HateClipSeg DeHate
#   sbatch --export=ALL,LIMIT=2 experiments/20261008_baselines/launch/eventvad_av_labserver.sbatch HateMM   (smoke)
# The scoring stage resumes from event_scores.jsonl, so a job stopped by the 1-day limit is resubmitted as is.
#SBATCH --job-name=evvad-av
#SBATCH --partition=local-sc448960
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --output=runs/20261008_baselines/eventvad_av/slurm_%j.out
set -uo pipefail
[ $# -ge 1 ] || { echo "usage: sbatch $0 <dataset> [<dataset> ...]"; exit 1; }
REPO=$HOME/Hate-follow-up
cd "$REPO"
PY=$REPO/.cache/envs/eventvad/bin/python
export HF_HOME=$REPO/.cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=$REPO/.cache/torch
export OMP_NUM_THREADS=4
EXTRA=${LIMIT:+--limit $LIMIT}
echo "host $(hostname) job ${SLURM_JOB_ID:-} datasets $* start $(date '+%F %T') commit $(git log -1 --format=%h)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
$PY -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || { echo FAILED cuda; exit 1; }
for DS in "$@"; do
  $PY -u experiments/20261008_baselines/eventvad/eventvad_4fps.py score --transcript --dataset "$DS" $EXTRA \
    || { echo "FAILED score --transcript $DS"; exit 1; }
done
echo "EVENTVAD_AV_JOB_DONE $* $(date '+%F %T')"
