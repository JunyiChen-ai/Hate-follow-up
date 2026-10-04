#!/usr/bin/env bash
#SBATCH --job-name=m1-quote-r2graph
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261005_m1_quote_graph/slurm_%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
bash experiments/20261005_m1_quote_graph/launch/run_lab_r2.sh "$@"
