#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2
python src/boundary_rescue/judge_offline.py \
  --model Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
  --all --batch-size 1 --gpu-mem 0.88 --max-model-len 16384
