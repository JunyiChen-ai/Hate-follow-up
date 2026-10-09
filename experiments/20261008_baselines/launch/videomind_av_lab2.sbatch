#!/bin/bash
# VideoMind-7B transcript (audio-visual) variant (videomind/README.md, "Transcript (audio-visual) variant,
# 2026-10-09") on uoa-lab2 (the VideoMind weights and env are on uoa-lab2 and lab-server; uoa-lab1's copy of the
# weights is gone). Same env (torch 2.8.0+cu128, transformers 4.45.2) and model files as the visual-only run.
# From the repo root on uoa-lab2:
#   sbatch experiments/20261008_baselines/launch/videomind_av_lab2.sbatch                 (HateMM HateClipSeg DeHate)
#   sbatch --export=ALL,DATASETS=HateMM,LIMIT=3 experiments/20261008_baselines/launch/videomind_av_lab2.sbatch   (smoke)
# Resumable: videos already in raw.jsonl without an error are skipped. Rasterising and evaluation run on uoa-lab1
# after rsync (videomind_hate.py raster --av).
#SBATCH --job-name=vmind-av
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --output=runs/20261008_baselines/videomind_av/slurm_%j.out
set -uo pipefail
REPO=$HOME/Hate-follow-up
cd "$REPO"
PY=$REPO/.cache/envs/videomind/bin/python          # torch 2.8.0+cu128, transformers 4.45.2 (pinned by VideoMind)
export HF_HOME=$REPO/.cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$REPO/third_party/VideoMind
MODEL=$(ls -d $HF_HOME/hub/models--yeliudev--VideoMind-7B/snapshots/* | head -1)
BASE=$(ls -d $HF_HOME/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/* | head -1)
DATASETS=${DATASETS:-HateMM HateClipSeg DeHate}
EXTRA=${LIMIT:+--limit $LIMIT}
echo "host $(hostname) job ${SLURM_JOB_ID:-} datasets ${DATASETS//,/ } start $(date '+%F %T') commit $(git log -1 --format=%h)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
for DS in ${DATASETS//,/ }; do
  $PY -u experiments/20261008_baselines/videomind/videomind_hate.py infer --av --dataset "$DS" --model "$MODEL" --base-model "$BASE" $EXTRA || { echo "FAILED $DS"; exit 1; }
  echo "CORPUS_DONE $DS $(date '+%F %T')"
done
echo "VIDEOMIND_AV_JOB_DONE $(date '+%F %T')"
