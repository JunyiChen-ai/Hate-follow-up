#!/bin/bash
# Sequential judge_offline submissions: wait for prev job to finish, submit next
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

MODEL="$1"
BATCH="${2:-4}"
GPU_MEM="${3:-0.88}"
MAX_MODEL_LEN="${4:-32768}"
EXTRA="${5:-}"

python src/boundary_rescue/judge_offline.py \
  --model "$MODEL" --all \
  --batch-size "$BATCH" --gpu-mem "$GPU_MEM" \
  --max-model-len "$MAX_MODEL_LEN" $EXTRA
