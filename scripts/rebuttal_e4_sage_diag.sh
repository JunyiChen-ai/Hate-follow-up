#!/bin/bash
# Rebuttal E4b — expert/gate diagnostic (1 GPU, short ~10-15 min).
# Reloads each dataset's best_model.pth and dumps per-expert accuracy + mean
# gate weight on the test set, to check whether any expert (esp. ZH text) is
# dead/noisy. Read-only w.r.t. training artifacts (writes diag.json only).
# Usage: sbatch --gres=gpu:1 --cpus-per-task=8 --mem=48G scripts/rebuttal_e4_sage_diag.sh
cd /data/jehc223/EMNLP2
source ~/.bashrc                    # /etc/bashrc reads unset vars — must precede `set -u`
conda activate SafetyContradiction
set -euo pipefail
export HF_HUB_OFFLINE=1
CFG=external_repos/SAGE/config/config_ours.yaml
OUT=results/rebuttal/E4_sage
echo "=== E4b DIAG start $(date) host=$(hostname) ==="
# datasets to diagnose; override by passing args, e.g. `... diag.sh impli`
DATASETS="${*:-mhclip_bl mhclip_yt hatemm impli}"
for ds in $DATASETS; do
  python scripts/rebuttal_e4_sage_run.py --mode diag --config "$CFG" \
      --dataset "$ds" --seed 0 --out_dir "$OUT"
done
echo "=== E4b DIAG done $(date) ==="
