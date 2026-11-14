#!/bin/bash
#SBATCH --job-name=rh-lf
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --output=runs/20261115_rh_hcs_split/slurm_%x_%j.out
# Label-free baselines on the Retrieval-hate cohorts, uoa-lab1 (README.md). Usage:
#   sbatch experiments/20261115_rh_hcs_split/launch/lf_lab1.sbatch naive:vlm:HateClipSeg mars:HateClipSeg ...
bash experiments/20261115_rh_hcs_split/launch/lf_run.sh "$@"
