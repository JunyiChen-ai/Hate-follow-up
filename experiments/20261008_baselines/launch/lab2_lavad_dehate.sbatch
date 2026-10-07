#!/usr/bin/env bash
#SBATCH --job-name=bl-lavad-dehate
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261008_baselines/lavad/slurm_%j.out
# LAVAD, DeHate: the 1151 cohort videos (the port's caption and summary indices are per video, so the output of a
# video does not depend on which other videos are processed). Resumable: resubmit to continue.
set -uo pipefail
cd /home/jehc223/Hate-follow-up
source experiments/20261008_baselines/launch/lab2_env.sh
$LAVAD_PYTHON -c "import transformers; assert transformers.__version__ == '4.49.0', transformers.__version__" || { echo "FAILED LAVAD env"; exit 1; }
IDS=runs/20261008_baselines/lavad/ids_DeHate.txt
$PY_VIDEO -c "
import sys; sys.path.insert(0, 'experiments/20261008_baselines'); import exact_cohort as ec
open('$IDS', 'w').write('\n'.join(ec.cohort('DeHate')) + '\n')"
bash experiments/20261008_baselines/lavad/run_chain.sh DeHate "$IDS" || { echo "FAILED chain"; exit 1; }
$PY_VIDEO experiments/20261008_baselines/lavad/finalize.py --dataset DeHate || echo "FAILED finalize (see run.log)"
echo "JOB_DONE $(date -Is)"
