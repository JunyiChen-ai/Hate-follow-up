#!/usr/bin/env bash
#SBATCH --job-name=m1-tree-controls
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261004_m1_tree/slurm_%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
source /home/jehc223/miniconda3/bin/activate HateVLM
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
case "${1:?smoke or main}" in
  smoke) opts=(--smoke);;
  main) opts=();;
  *) exit 2;;
esac
nvidia-smi
python - <<'PY'
import json
from pathlib import Path
summary=json.loads(Path('runs/20261004_m1_tree/r3_full_main_analysis/summary.json').read_text())
assert summary['gates']['performance_pass'] is True, 'R3 full main performance gate has not passed; controls forbidden'
assert summary['datasets']['HateMM']['n_eligible']==84 and summary['datasets']['HateClipSeg']['n_eligible']==99
PY
python -u experiments/20261004_m1_tree/control_extract.py "${opts[@]}"
python -u experiments/20261004_m1_tree/control_measure.py "${opts[@]}"
echo CONTROL_GPU_DONE
