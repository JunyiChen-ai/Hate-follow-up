#!/bin/bash
# Rebuttal E4b — SAGE preprocessing (CPU only, no GPU).
# Builds SAGE-format splits, flat 16-frame dirs, and 16k HateMM audio under
# results/rebuttal/E4_sage/. Idempotent / resumable (skips existing frames+wav).
# Submit all:  sbatch --cpus-per-task=8 --mem=32G scripts/rebuttal_e4_sage_prep.sh
# One dataset: sbatch --cpus-per-task=8 --mem=32G scripts/rebuttal_e4_sage_prep.sh ImpliHateVid
cd /data/jehc223/EMNLP3
source ~/.bashrc                    # /etc/bashrc reads unset vars — must precede `set -u`
conda activate SafetyContradiction
set -euo pipefail                  # strict flags for the actual work
echo "=== E4b PREP start $(date) host=$(hostname) datasets=[${*:-ALL}] ==="
python scripts/rebuttal_e4_sage_prep.py "$@"
echo "=== E4b PREP done $(date) ==="
