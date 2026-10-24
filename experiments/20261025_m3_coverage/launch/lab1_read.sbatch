#!/usr/bin/env bash
#SBATCH --job-name=m3-read
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/jehc223/Hate-follow-up
#SBATCH --output=runs/20261025_m3_coverage/slurm_%j.out
# M3 extra-question option reads (read_extra.py) on uoa-lab1. Run from the repo root after mkdir -p runs/20261025_m3_coverage:
#   RUN=<run name> QFILE=experiments/20261025_m3_coverage/questions/<q>.txt OPTIONS="A B C D E F" MODE=both \
#   DATASETS="HateMM HateClipSeg" [LIMIT=2] [VERIFY=1] sbatch experiments/20261025_m3_coverage/launch/lab1_read.sbatch
# Smoke: LIMIT=2 VERIFY=1 (2 videos per corpus, all four compared with a plain call). DeHate: DATASETS=DeHate.
# Output: runs/20261025_m3_coverage/$RUN/{reads.jsonl, config.json, run.log, run.pid, verify.jsonl}.
set -euo pipefail
cd /home/jehc223/Hate-follow-up
: "${RUN:?set RUN}" "${QFILE:?set QFILE}" "${OPTIONS:?set OPTIONS}"
MODE=${MODE:-both}
DATASETS=${DATASETS:-HateMM HateClipSeg}
export PATH=/home/jehc223/Hate-follow-up/.cache/envs/HateVLM/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/home/jehc223/Hate-follow-up/.cache/triton CUDA_CACHE_PATH=/home/jehc223/Hate-follow-up/.cache/cuda
export PYTORCH_ALLOC_CONF=expandable_segments:True
hostname; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA in the job'; print(torch.cuda.get_device_name(0))"
# shellcheck disable=SC2086
python -u experiments/20261025_m3_coverage/read_extra.py --run-name "$RUN" --question-file "$QFILE" --options $OPTIONS \
  --mode "$MODE" --datasets $DATASETS ${LIMIT:+--limit "$LIMIT"} ${VERIFY:+--verify}
grep -q "DONE videos" "runs/20261025_m3_coverage/$RUN/run.log"
echo M3_READ_DONE "$RUN"
