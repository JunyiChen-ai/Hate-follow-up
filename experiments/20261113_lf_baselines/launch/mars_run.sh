#!/bin/bash
# Body of the MARS-on-Qwen3-VL-8B jobs (README §2). Arguments: <dataset> ... ; per dataset `mars_q3vl.py infer`
# (vLLM env, text generation) then `readout` (HateVLM env, fp32 label read-out); both resumable.
# Extra environment: LIMIT=<n> for a smoke run; LF_RUN_ROOT=<dir> writes outside the run root.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
C=$PWD/.cache
export LF_RUN_ROOT=${LF_RUN_ROOT:-}
export HF_HOME=$C/hf HF_HUB_OFFLINE=1 TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg VLLM_CACHE_ROOT=$C/vllm \
  TORCHINDUCTOR_CACHE_DIR=$C/torchinductor TRITON_CACHE_DIR=$C/triton MPLCONFIGDIR=$C/mpl VLLM_NO_USAGE_STATS=1 \
  DO_NOT_TRACK=1 PYTHONUNBUFFERED=1
mkdir -p $C/tmp $C/xdg $C/vllm $C/mpl
echo "host $(hostname) job ${SLURM_JOB_ID:-none} datasets $* $(date) commit $(git log -1 --format='%h %cd' --date=short)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
HF=$C/envs/HateVLM/bin/python; [ -x $HF ] || HF=$HOME/miniconda3/envs/HateVLM/bin/python
VL=$C/envs/vllm_q3vl/bin/python
X=experiments/20261113_lf_baselines/mars_q3vl.py
rc=0
for ds in "$@"; do
  $VL $X infer --dataset $ds --limit ${LIMIT:-0} || { echo "FAILED infer $ds"; rc=1; continue; }
  $HF $X readout --dataset $ds || { echo "FAILED readout $ds"; rc=1; }
done
[ $rc = 0 ] && echo "DONE mars $*"
exit $rc
