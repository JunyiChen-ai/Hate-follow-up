#!/bin/bash
# Body of the label-free jobs on the Retrieval-hate cohorts (README.md); called by lf_lab1.sbatch / lf_lab3.sbatch
# inside the Slurm allocation. Arguments, run one after the other (each stage is resumable):
#   naive:<arm>:<dataset>    naive_prompt.py infer + raster   (arm vlm = Qwen3-VL-8B, vlm_internvl = InternVL3.5-8B)
#   mars:<dataset>           mars_q3vl.py infer (vLLM) + readout (HF) + raster
#   eventvad:<dataset>       eventvad_q3vl.py infer (vLLM) + readout (HF) + raster; needs $RH_RUNS/eventvad/<dataset>/events.jsonl
# A failed task prints FAILED and the next one still runs.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source experiments/20261115_rh_hcs_split/env.sh
C=$PWD/.cache
export HF_HOME=$C/hf HF_HUB_OFFLINE=1 TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg VLLM_CACHE_ROOT=$C/vllm \
  TORCHINDUCTOR_CACHE_DIR=$C/torchinductor TRITON_CACHE_DIR=$C/triton MPLCONFIGDIR=$C/mpl VLLM_NO_USAGE_STATS=1 \
  DO_NOT_TRACK=1 PYTHONUNBUFFERED=1
mkdir -p $C/tmp $C/xdg $C/vllm $C/mpl
echo "host $(hostname) job ${SLURM_JOB_ID:-none} tasks $* $(date) commit $(git log -1 --format='%h %cd' --date=short)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
HF=$C/envs/HateVLM/bin/python; [ -x $HF ] || HF=$HOME/miniconda3/envs/HateVLM/bin/python
VL=$C/envs/vllm_q3vl/bin/python
CPU=$HOME/miniconda3/envs/HateVideo/bin/python
X=experiments/20261113_lf_baselines
failed=()
for task in "$@"; do
  IFS=: read -r kind a b <<< "$task"
  echo "$(date -Is) TASK START $task"
  case $kind in
  naive)
    $HF $X/naive_prompt.py infer --arm $a --dataset $b && $CPU $X/naive_prompt.py raster --arm $a --dataset $b ;;
  mars)
    $VL $X/mars_q3vl.py infer --dataset $a && $HF $X/mars_q3vl.py readout --dataset $a && $CPU $X/mars_q3vl.py raster --dataset $a ;;
  eventvad)
    $VL $X/eventvad_q3vl.py infer --dataset $a && $HF $X/eventvad_q3vl.py readout --dataset $a && $CPU $X/eventvad_q3vl.py raster --dataset $a ;;
  *) false ;;
  esac
  rc=$?
  if [ $rc -eq 0 ]; then echo "$(date -Is) TASK DONE $task"; else echo "$(date -Is) TASK FAILED $task rc=$rc"; failed+=("$task"); fi
done
if [ ${#failed[@]} -eq 0 ]; then echo "$(date -Is) ALL TASKS DONE"; else echo "$(date -Is) FAILED: ${failed[*]}"; exit 1; fi
