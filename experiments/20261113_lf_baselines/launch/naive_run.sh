#!/bin/bash
# Body of the naive-prompt jobs (README §1). Arguments: <arm>:<dataset> ... ; each pair runs `naive_prompt.py infer`
# in the HateVLM env (resumable). Called by naive_lab1.sbatch / naive_lab2.sbatch inside the Slurm allocation.
# Extra environment: LIMIT=<n> for a smoke run (first n cohort videos); LF_RUN_ROOT=<dir> writes outside the run root.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
C=$PWD/.cache
export LF_RUN_ROOT=${LF_RUN_ROOT:-}
export HF_HOME=$C/hf HF_HUB_OFFLINE=1 TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg VLLM_CACHE_ROOT=$C/vllm \
  TORCHINDUCTOR_CACHE_DIR=$C/torchinductor TRITON_CACHE_DIR=$C/triton MPLCONFIGDIR=$C/mpl VLLM_NO_USAGE_STATS=1 \
  DO_NOT_TRACK=1 PYTHONUNBUFFERED=1
mkdir -p $C/tmp $C/xdg $C/vllm $C/mpl
echo "host $(hostname) job ${SLURM_JOB_ID:-none} pairs $* $(date) commit $(git log -1 --format='%h %cd' --date=short)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
PY=$C/envs/HateVLM/bin/python; [ -x $PY ] || PY=$HOME/miniconda3/envs/HateVLM/bin/python   # lab1 keeps it in .cache/envs, lab2/lab3 in miniconda
rc=0
for pair in "$@"; do
  arm=${pair%%:*}; ds=${pair##*:}
  $PY experiments/20261113_lf_baselines/naive_prompt.py infer --arm $arm --dataset $ds \
    --limit ${LIMIT:-0} || { echo "FAILED $arm $ds"; rc=1; }
done
[ $rc = 0 ] && echo "DONE naive $*"
exit $rc
