#!/bin/bash
#SBATCH -J ds_em
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o /data/jehc223/EMNLP2/logs/ds_em_%j.out
#SBATCH -e /data/jehc223/EMNLP2/logs/ds_em_%j.err

set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2
python scripts/ds_em_estimate.py
