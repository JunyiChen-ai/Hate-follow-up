#!/bin/bash
#SBATCH --job-name=sage-win8
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --output=runs/20261008_baselines/sage/slurm_%x_%j.out
# SAGE per-window baseline, one corpus per job: train seeds 2025/234/3407 on train (epoch chosen on val), then score
# every 8-s window of the test cohort and evaluate. Usage (in the repo on the target machine, after
#   python experiments/20261008_baselines/sage/sage_run.py prep --dataset <DS>   (CPU, setsid nohup)):
#   sbatch experiments/20261008_baselines/launch/sage_lab2.sbatch <DS>
set -uo pipefail
DS=${1:?dataset}
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=$PWD/.cache/hf HF_HUB_OFFLINE=1 TMPDIR=$PWD/.cache/tmp PYTHONUNBUFFERED=1
PY=$PWD/.cache/envs/hv_cv2/bin/python
echo "host $(hostname) job $SLURM_JOB_ID dataset $DS $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
$PY -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || { echo FAILED cuda; exit 1; }
for s in 2025 234 3407; do
  if grep -q "DONE train" runs/20261008_baselines/sage/$DS/seed$s/run.log 2>/dev/null; then
    echo "seed $s already trained"
  else
    $PY experiments/20261008_baselines/sage/sage_run.py train --dataset $DS --seed $s || { echo "FAILED train $s"; exit 1; }
  fi
done
$PY experiments/20261008_baselines/sage/sage_run.py infer --dataset $DS || { echo "FAILED infer"; exit 1; }
echo "DONE sage $DS $(date)"
