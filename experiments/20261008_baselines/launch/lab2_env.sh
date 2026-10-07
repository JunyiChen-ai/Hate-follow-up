# Sourced by the lab2_*.sbatch jobs (uoa-lab2, partition local-sc474399). Models are read from the existing
# Hugging Face cache in offline mode, so nothing is downloaded or written outside the repository.
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRITON_CACHE_DIR=/home/jehc223/Hate-follow-up/.cache/triton
export MPLCONFIGDIR=/home/jehc223/Hate-follow-up/.cache/mpl
export TORCH_HOME=/home/jehc223/Hate-follow-up/.cache/torch
export PYTHONPATH=/home/jehc223/Hate-follow-up
export PY_VIDEO=$HOME/miniconda3/envs/HateVideo/bin/python   # torch 2.7.1, transformers 4.57.6
export PY_VLM=$HOME/miniconda3/envs/HateVLM/bin/python       # torch 2.11.0, transformers 5.15.1
export EVAL_PYTHON=$PY_VIDEO                                  # canonical evaluator runs in HateVideo
echo "job ${SLURM_JOB_ID:-none} host $(hostname) start $(date -Is) commit $(git -C /home/jehc223/Hate-follow-up rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
$PY_VIDEO -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
