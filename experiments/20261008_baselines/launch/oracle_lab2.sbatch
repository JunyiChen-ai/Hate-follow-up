#!/bin/bash
# Oracle test selection of the weakly supervised baselines (experiments/20261008_baselines/oracle_test_selection/
# README.md): retrain with the identical recipes and write the test cohort's scores after every epoch.
# Submit from the repo root on uoa-lab2:
#   mkdir -p runs/20261008_baselines/oracle_test_selection/slurm
#   sbatch experiments/20261008_baselines/launch/oracle_lab2.sbatch <task> [<task> ...]
# Tasks (oracle_test_selection/run_tasks.sh): mil:<feature>:<corpus>  fixed:<method>:<corpus>  mhl:<corpus>
#   sage:<DS>  clara:<DS>. A finished seed is kept on resubmission.
#SBATCH --job-name=oracle-ts
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --output=runs/20261008_baselines/oracle_test_selection/slurm/%x_%j.out
set -uo pipefail
[ $# -ge 1 ] || { echo "FAILED: no task"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
C=$PWD/.cache
export HF_HOME=$C/hf TORCH_HOME=$C/torch HF_HUB_OFFLINE=1 TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg MPLCONFIGDIR=$C/mpl \
  PYTHONUNBUFFERED=1 OPENCV_LOG_LEVEL=ERROR TOKENIZERS_PARALLELISM=false
mkdir -p $C/tmp $C/xdg $C/mpl
echo "host $(hostname) job $SLURM_JOB_ID tasks $* $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
$HOME/miniconda3/envs/HateVideo/bin/python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" \
  || { echo "FAILED cuda"; exit 1; }
bash experiments/20261008_baselines/oracle_test_selection/run_tasks.sh "$@"
