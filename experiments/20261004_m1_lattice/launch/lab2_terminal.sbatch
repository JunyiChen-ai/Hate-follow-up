#!/usr/bin/env bash
#SBATCH --job-name=m1-terminal-lattice
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261004_m1_lattice/terminal_slurm_%j.out
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
# The unchanged actual R1 ASR cache is validated before every video; no GT or prior scores.
python -u experiments/20261004_m1_lattice/terminal_measure.py "${opts[@]}"
echo ACQUISITION_DONE
