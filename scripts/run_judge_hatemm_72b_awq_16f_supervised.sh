#!/bin/bash
#SBATCH -J q72_hm_16f
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -o /data/jehc223/EMNLP2/slurm-%j.out

set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

python scripts/supervise_q72_hatemm_16f.py

jq -s '[length, (map(select(.pred==0 or .pred==1))|length), (map(select(.pred==-1 or (.raw_response? == "")))|length)] | @tsv' \
  -r results/boundary_rescue/HateMM/offline_test_qwen2.5-vl-72b-awq.jsonl
