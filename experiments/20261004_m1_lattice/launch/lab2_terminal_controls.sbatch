#!/usr/bin/env bash
#SBATCH --job-name=m1-terminal-controls
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261004_m1_lattice/terminal_controls_slurm_%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
source /home/jehc223/miniconda3/bin/activate HateVLM
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# Dispatch guard only: this summary is never an input to a speech/model read.
python - <<'PY'
import json
from pathlib import Path
p=Path('runs/20261004_m1_lattice/r3_full_main_analysis/summary.json')
r=json.loads(p.read_text())
assert r['gates']['performance_pass'] is True, 'complete main gate has not passed'
assert {d:v['n_eligible'] for d,v in r['datasets'].items()}=={'HateMM':84,'HateClipSeg':99}
assert len(Path('runs/20261004_m1_lattice/r3_full_main/optimized/predictions.jsonl').read_text().splitlines())==333
print('COMPLETE_MAIN_GUARD_PASS')
PY
case "${1:?smoke or main}" in
  smoke) opts=(--smoke);;
  main) opts=();;
  *) exit 2;;
esac
nvidia-smi
python -u experiments/20261004_m1_lattice/terminal_control_measure.py "${opts[@]}"
