#!/bin/bash
# Body of the §3 jobs (README §3): published methods on Qwen3-VL-8B. Arguments: <method> <dataset> ...
#   probevad  `probevad_q3vl.py infer` (HateVLM env);            extra environment STRIDE=<seconds> (default 4)
#   corevad   `corevad_q3vl.py infer` (vLLM env) then `readout` (HateVLM env);  UNIT=<seconds> (default 8)
#   cogvadu   `cogvadu_q3vl.py infer` (vLLM env), `readout` (HateVLM env), `embed` (HateVideo env, ImageBind)
#   lavad     `lavad_q3vl.py infer` (vLLM env) then `readout` (HateVLM env)
#   anyanomaly `anyanomaly_q3vl.py infer` (vLLM env: CLIP context + generation) then `readout` (HateVLM env)
#   lela      `lela_q3vl.py infer` (vLLM env) then `readout` (HateVLM env)
# All stages are resumable. LIMIT=<n> for a smoke run (first n cohort videos); LF_RUN_ROOT=<dir> writes outside the
# run root. Called by q3vl_lab{1,2,3}.sbatch inside the Slurm allocation.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
C=$PWD/.cache
export LF_RUN_ROOT=${LF_RUN_ROOT:-}
export HF_HOME=$C/hf HF_HUB_OFFLINE=1 TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg VLLM_CACHE_ROOT=$C/vllm \
  TORCHINDUCTOR_CACHE_DIR=$C/torchinductor TRITON_CACHE_DIR=$C/triton MPLCONFIGDIR=$C/mpl VLLM_NO_USAGE_STATS=1 \
  DO_NOT_TRACK=1 PYTHONUNBUFFERED=1
mkdir -p $C/tmp $C/xdg $C/vllm $C/mpl
method=$1; shift
echo "host $(hostname) job ${SLURM_JOB_ID:-none} method $method datasets $* $(date) commit $(git log -1 --format='%h %cd' --date=short)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
HF=$C/envs/HateVLM/bin/python; [ -x $HF ] || HF=$HOME/miniconda3/envs/HateVLM/bin/python   # lab1 keeps it in .cache/envs, lab2/lab3 in miniconda
VL=$C/envs/vllm_q3vl/bin/python
HV=$HOME/miniconda3/envs/HateVideo/bin/python
X=experiments/20261113_lf_baselines/${method}_q3vl.py
rc=0
for ds in "$@"; do
  case $method in
    probevad)
      $HF $X infer --dataset $ds --stride ${STRIDE:-4} --limit ${LIMIT:-0} || { echo "FAILED infer $ds"; rc=1; } ;;
    corevad)
      $VL $X infer --dataset $ds --unit ${UNIT:-8} --limit ${LIMIT:-0} || { echo "FAILED infer $ds"; rc=1; continue; }
      $HF $X readout --dataset $ds --unit ${UNIT:-8} || { echo "FAILED readout $ds"; rc=1; } ;;
    cogvadu)
      $VL $X infer --dataset $ds --limit ${LIMIT:-0} || { echo "FAILED infer $ds"; rc=1; continue; }
      $HF $X readout --dataset $ds || { echo "FAILED readout $ds"; rc=1; continue; }
      $HV $X embed --dataset $ds || { echo "FAILED embed $ds"; rc=1; } ;;
    lavad|lela|anyanomaly)
      $VL $X infer --dataset $ds --limit ${LIMIT:-0} || { echo "FAILED infer $ds"; rc=1; continue; }
      $HF $X readout --dataset $ds || { echo "FAILED readout $ds"; rc=1; } ;;
    *) echo "FAILED unknown method $method"; exit 2 ;;
  esac
done
[ $rc = 0 ] && echo "DONE $method $*"
exit $rc
