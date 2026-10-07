#!/usr/bin/env bash
#SBATCH --job-name=bl-lavad-hcs-all
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261008_baselines/lavad/slurm_%j.out
# LAVAD, HateClipSeg, all 118 cohort videos with the port. Used because the 6-video rerun did not reproduce the
# campaign's curves within float tolerance (HateClipSeg_rerun_check.json), so old and new curves are not mixed
# (run_plan.md L1b). Videos already done by the 6-video rerun are skipped (every stage is idempotent).
set -uo pipefail
cd /home/jehc223/Hate-follow-up
source experiments/20261008_baselines/launch/lab2_env.sh
IDS=runs/20261008_baselines/lavad/ids_HateClipSeg.txt
$PY_VIDEO -c "
import sys; sys.path.insert(0, 'experiments/20261008_baselines'); import exact_cohort as ec
open('$IDS', 'w').write('\n'.join(ec.cohort('HateClipSeg')) + '\n')"
bash experiments/20261008_baselines/lavad/run_chain.sh HateClipSeg "$IDS" || { echo "FAILED chain"; exit 1; }
$PY_VIDEO experiments/20261008_baselines/lavad/finalize.py --dataset HateClipSeg --all-rerun || echo "FAILED finalize (see run.log)"
echo "JOB_DONE $(date -Is)"
