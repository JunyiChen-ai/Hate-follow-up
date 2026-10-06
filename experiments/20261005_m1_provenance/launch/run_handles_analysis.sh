#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p runs/20261005_m1_provenance/r1_handles_full_main_analysis
exec > >(tee -a runs/20261005_m1_provenance/r1_handles_full_main_analysis/run.log) 2>&1
hostname
echo "$BASHPID" > runs/20261005_m1_provenance/r1_handles_full_main_analysis/run.pid
.cache/envs/HateVLM/bin/python -u experiments/20261005_m1_provenance/analyze_handles.py --stage prepare
source /home/jehc223/miniconda3/bin/activate HateVideo
python -u experiments/20261005_m1_provenance/analyze_handles.py --stage evaluate --name base
python -u experiments/20261005_m1_provenance/analyze_handles.py --stage evaluate --name optimized
python -u experiments/20261005_m1_provenance/analyze_handles.py --stage report
