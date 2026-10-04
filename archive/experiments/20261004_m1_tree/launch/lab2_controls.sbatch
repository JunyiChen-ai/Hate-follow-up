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
revision=${2:-r3}
case "$revision" in r3|r4) ;; *) exit 2;; esac
nvidia-smi
python - "$revision" <<'PY'
import json
import sys
from pathlib import Path
summary=json.loads(Path('runs/20261004_m1_tree',sys.argv[1]+'_full_main_analysis/summary.json').read_text())
assert summary['gates']['performance_pass'] is True, 'Complete main performance gate has not passed; controls forbidden'
assert summary['datasets']['HateMM']['n_eligible']==84 and summary['datasets']['HateClipSeg']['n_eligible']==99
PY
python -u experiments/20261004_m1_tree/control_extract.py "${opts[@]}" --revision "$revision"
python -u experiments/20261004_m1_tree/control_measure.py "${opts[@]}" --revision "$revision"
echo CONTROL_GPU_DONE
