#!/bin/bash
#SBATCH --job-name=rh-hcs-weaksup
#SBATCH --partition=local-sc474398
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --output=runs/20261115_rh_hcs_split/slurm_%x_%j.out
# Weakly supervised baselines retrained on the Retrieval-hate HateClipSeg split (README.md), uoa-lab1 or uoa-lab3 (weaksup_lab3.sbatch: CLARA stage, whose encoder weights are on uoa-lab3), one job, stages
# one after the other (each skips what is finished, so a resubmission resumes):
#   mil      BERT / wav2vec2 / CLIP + MIL and AVadCLIP (weaksup_common/run_tasks.sh), seeds 2025 / 234 / 3407
#   sage     SAGE train (3 seeds) + window scores   (after: sage_run.py prep, CPU)
#   clara    CLARA embeddings + training (rationale mode both, 3 seeds)   (after: clara_prep.py and OCR, CPU)
# Usage: sbatch experiments/20261115_rh_hcs_split/launch/weaksup_lab1.sbatch [mil] [sage] [clara]
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source experiments/20261115_rh_hcs_split/env.sh
C=$PWD/.cache
export HF_HOME=$C/hf HF_HUB_OFFLINE=1 TORCH_HOME=$C/torch TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg MPLCONFIGDIR=$C/mpl \
  PYTHONUNBUFFERED=1 OPENCV_LOG_LEVEL=ERROR
mkdir -p $C/tmp $C/xdg $C/mpl
DS=HateClipSeg
echo "host $(hostname) job $SLURM_JOB_ID stages ${*:-mil sage clara} $(date) commit $(git log -1 --format='%h %cd' --date=short)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
rc=0
for stage in ${@:-mil sage clara}; do
  case $stage in
  mil)
    ( source "$HOME/miniconda3/bin/activate" HateVideo
      bash experiments/20261008_baselines/weaksup_common/run_tasks.sh mil:bert:hateclipseg mil:wav2vec2:hateclipseg \
        mil:clip:hateclipseg method:avadclip:hateclipseg ) || { echo "FAILED mil"; rc=1; } ;;
  sage)
    PY=$C/envs/hv_cv2/bin/python
    ok=1
    for s in 2025 234 3407; do
      if grep -q "DONE train" $RH_RUNS/sage/$DS/seed$s/run.log 2>/dev/null; then echo "sage seed $s already trained"; continue; fi
      $PY experiments/20261008_baselines/sage/sage_run.py train --dataset $DS --seed $s || { echo "FAILED sage train $s"; ok=0; break; }
    done
    [ $ok = 1 ] && { $PY experiments/20261008_baselines/sage/sage_run.py infer --dataset $DS || ok=0; }
    [ $ok = 1 ] && echo "DONE sage $DS $(date)" || { echo "FAILED sage"; rc=1; } ;;
  clara)
    PY=$C/envs/hv_cv2/bin/python
    X=experiments/20261008_baselines/clara
    ok=1
    grep -q "DONE prep" $RH_RUNS/clara/$DS/prep/run.log || { echo "FAILED clara: run clara_prep.py first"; ok=0; }
    [ $ok = 1 ] && { $PY $X/clara_ocr.py --dataset $DS --check || { echo "FAILED clara: OCR incomplete"; ok=0; }; }
    [ $ok = 1 ] && { $PY $X/clara_embed.py --dataset $DS || { echo "FAILED clara embed"; ok=0; }; }
    if [ $ok = 1 ]; then
      for s in 2025 234 3407; do
        if grep -q "DONE clara" $RH_RUNS/clara/$DS/seed$s/run.log 2>/dev/null; then echo "clara seed $s done"; continue; fi
        $PY $X/clara_train.py --dataset $DS --seed $s --rationale-mode both || { echo "FAILED clara train $s"; ok=0; break; }
      done
    fi
    [ $ok = 1 ] && echo "DONE clara $DS $(date)" || { echo "FAILED clara"; rc=1; } ;;
  *) echo "FAILED unknown stage $stage"; rc=1 ;;
  esac
done
[ $rc = 0 ] && echo "JOB_DONE $(date)" || echo "JOB_FAILED $(date)"
exit $rc
