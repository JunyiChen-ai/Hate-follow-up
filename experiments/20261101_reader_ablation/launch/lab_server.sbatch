#!/usr/bin/env bash
#SBATCH --job-name=reader-abl
#SBATCH --partition=local-sc448960
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261101_reader_ablation/slurm_%j.out
# Same job on lab-server (account junyi, conda HateVLM):
#   SCOPE=DeHate ARMS="no_stance asr_windows" sbatch experiments/20261101_reader_ablation/launch/lab_server.sbatch
# Needs data/temporal_retrieved_kv/<corpus>, data/asr_whisper_aligned/<corpus>, data/frames_k20, data/asr_whisper_large_v3 and the
# stored adopted reads (README §4) rsynced from uoa-lab1.
set -euo pipefail
cd /home/junyi/Hate-follow-up
source /home/junyi/miniconda3/bin/activate HateVLM
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/junyi/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/home/junyi/Hate-follow-up/.cache/triton CUDA_CACHE_PATH=/home/junyi/Hate-follow-up/.cache/cuda
export PYTORCH_ALLOC_CONF=expandable_segments:True
hostname; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
case ${SCOPE:-smoke} in
  smoke) python -u experiments/20261101_reader_ablation/read_arms.py --smoke;;
  HateMM|HateClipSeg|DeHate) python -u experiments/20261101_reader_ablation/read_arms.py --datasets $SCOPE ${ARMS:+--arms $ARMS};;
  *) echo FAILED_invalid_scope; exit 2;;
esac
echo READER_ABLATION_DONE
