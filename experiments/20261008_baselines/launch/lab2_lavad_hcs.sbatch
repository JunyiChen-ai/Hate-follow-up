#!/usr/bin/env bash
#SBATCH --job-name=bl-lavad-hcs
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261008_baselines/lavad/slurm_%j.out
# LAVAD, HateClipSeg: rerun bit_ZaY0S1anrdep (12 unscored frames in the campaign curve) plus 5 overlap videos for
# the reproduction check (run_plan.md L1b), then complete HateClipSeg.
set -uo pipefail
cd /home/jehc223/Hate-follow-up
source experiments/20261008_baselines/launch/lab2_env.sh
IDS=runs/20261008_baselines/lavad/ids_HateClipSeg_rerun.txt
$PY_VIDEO -c "
import sys; sys.path.insert(0, 'experiments/20261008_baselines'); import exact_cohort as ec
ids = ['bit_ZaY0S1anrdep'] + [v for v in ec.cohort('HateClipSeg') if v != 'bit_ZaY0S1anrdep'][:5]
open('$IDS', 'w').write('\n'.join(ids) + '\n')"
bash experiments/20261008_baselines/lavad/run_chain.sh HateClipSeg "$IDS" || { echo "FAILED chain"; exit 1; }
$PY_VIDEO experiments/20261008_baselines/lavad/compare_rerun.py --ids-file "$IDS" || { echo "FAILED compare"; exit 1; }
$PY_VIDEO experiments/20261008_baselines/lavad/finalize.py --dataset HateClipSeg || echo "FAILED finalize (see run.log)"
echo "JOB_DONE $(date -Is)"
