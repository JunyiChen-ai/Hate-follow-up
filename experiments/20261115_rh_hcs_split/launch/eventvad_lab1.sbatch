#!/bin/bash
#SBATCH --job-name=rh-evvad
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --output=runs/20261115_rh_hcs_split/slurm_%x_%j.out
# EventVAD on Qwen3-VL-8B on the 79 test videos of the Retrieval-hate HateClipSeg split, uoa-lab1 (README.md):
#   1 the main-table EventVAD segmentation (eventvad_4fps.py segment, released config) for the test videos that have no
#     events yet (those shared with the existing run are carried over by reuse_existing.py eventvad);
#   2 eventvad_q3vl.py infer + readout + raster (launch/lf_run.sh eventvad:HateClipSeg).
# Both stages resume from their jsonl files.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
REPO=$PWD
source experiments/20261115_rh_hcs_split/env.sh
PY=$HOME/venvs/SafetyContradiction/bin/python   # lab1 EventVAD venv (as launch/eventvad_lab1.sbatch of 20261008_baselines)
echo "host $(hostname) job ${SLURM_JOB_ID:-} start $(date '+%F %T') commit $(git log -1 --format=%h)"
( export HF_HOME=$REPO/.cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=$REPO/.cache/torch OMP_NUM_THREADS=4
  $PY -u experiments/20261008_baselines/eventvad/eventvad_4fps.py segment --dataset HateClipSeg ) || { echo "FAILED segment"; exit 1; }
echo "SEGMENT_DONE $(date '+%F %T')"
bash experiments/20261115_rh_hcs_split/launch/lf_run.sh eventvad:HateClipSeg
