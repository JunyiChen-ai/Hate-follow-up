#!/bin/bash
# EventVAD (4 fps protocol) on lab-server (sc448960). Usage, from the repo root on lab-server:
#   sbatch experiments/20261008_baselines/launch/eventvad_labserver.sbatch DeHate
#   sbatch experiments/20261008_baselines/launch/eventvad_labserver.sbatch DeHate HateMM HateClipSeg   (corpora in order)
#   sbatch --export=ALL,LIMIT=3 experiments/20261008_baselines/launch/eventvad_labserver.sbatch DeHate   (smoke)
# Both stages resume from their jsonl files, so a job stopped by the 1-day partition limit is resubmitted as is.
#SBATCH --job-name=evvad
#SBATCH --partition=local-sc448960
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --output=runs/20261008_baselines/eventvad/slurm_%j.out
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
  $PY -u experiments/20261008_baselines/eventvad/eventvad_4fps.py segment --dataset "$DS" $EXTRA || { echo "FAILED segment $DS"; exit 1; }
  $PY -u experiments/20261008_baselines/eventvad/eventvad_4fps.py score --dataset "$DS" $EXTRA || { echo "FAILED score $DS"; exit 1; }
done
echo "EVENTVAD_JOB_DONE $* $(date '+%F %T')"
