#!/usr/bin/env bash
#SBATCH --job-name=spvl_mllm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=runs/20260910_spvl/slurm_%j.out
# Family / size robustness study on campus (README §11). One model per job, all arms sequentially inside it.
# Submit: cd /data/jehc223/Hate-follow-up && sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-32B-Instruct,MODEL_TAG=q3vl-32b experiments/20260910_spvl/launch/campus_mllm.sbatch
set -eo pipefail
source /data/jehc223/miniconda3/bin/activate HateVLM
set -u
cd /data/jehc223/Hate-follow-up
mkdir -p runs/20260910_spvl  # note: Slurm opens --output before this runs; the directory must exist at submit time
export HF_HOME=/data/jehc223/Hate-follow-up/.cache/hf
export SPVL_PYTHON=/data/jehc223/miniconda3/envs/HateVLM/bin/python
: "${MODEL_ID:?set MODEL_ID}" "${MODEL_TAG:?set MODEL_TAG}"
echo "host $(hostname) job $SLURM_JOB_ID gpu ${CUDA_VISIBLE_DEVICES:-?} model $MODEL_ID tag $MODEL_TAG"
bash experiments/20260910_spvl/launch/run_mllm.sh "$MODEL_ID" "$MODEL_TAG" ${ARMS:-}
echo "SBATCH_DONE $MODEL_TAG $(date -Is)"
