#!/bin/bash
# Oracle test selection tasks, one after another inside one Slurm job (launch/oracle_lab*.sbatch). Run from the repo
# root. A finished seed is kept, so a resubmission resumes. Exit status 1 if any task failed (the others still run).
#   mil:<bert|wav2vec2|clip>:<hatemm|hateclipseg|dehate>     retrain.py mil
#   fixed:<vadclip|dsanet|avadclip>:<corpus>                  retrain.py fixed
#   mhl:<corpus>                                              retrain.py mhl
#   sage:<HateMM|HateClipSeg|DeHate>   sage_run.py train --save-every-epoch (3 seeds) then sage_epochs.py
#                                      (needs the test window frames of sage_run.py prep)
#   clara:<DS>[:cpu]                   clara_train.py --save-every-epoch (3 seeds; needs data/clara_emb/<DS>)
set -uo pipefail
X=experiments/20261008_baselines/oracle_test_selection
O=runs/20261008_baselines/oracle_test_selection
PY=$HOME/miniconda3/envs/HateVideo/bin/python
PYCV=$PWD/.cache/envs/hv_cv2/bin/python

sage() {
  local DS=$1 s
  for s in 2025 234 3407; do
    if grep -q "DONE train" $O/sage/$DS/seed$s/run.log 2>/dev/null; then echo "sage $DS seed $s trained"; continue; fi
    rm -rf $O/sage/$DS/seed$s
    DETWIN_RUNS=$PWD/$O $PYCV experiments/20261008_baselines/sage/sage_run.py train --dataset $DS --seed $s \
      --save-every-epoch || return 1
  done
  grep -q "DONE score_epochs" $O/sage/$DS/score_epochs/run.log 2>/dev/null && return 0
  $PYCV $X/sage_epochs.py --dataset $DS
}

clara() {
  local DS=$1 dev=${2:-gpu} s extra=""
  [ "$dev" = cpu ] && extra="--cpu"
  for s in 2025 234 3407; do
    if grep -q "DONE clara" $O/clara/$DS/seed$s/run.log 2>/dev/null; then echo "clara $DS seed $s done"; continue; fi
    rm -rf $O/clara/$DS/seed$s
    DETWIN_RUNS=$PWD/$O $PYCV experiments/20261008_baselines/clara/clara_train.py --dataset $DS --seed $s \
      --save-every-epoch $extra || return 1
  done
}

rc=0
for t in "$@"; do
  IFS=: read -r fam a b <<< "$t"
  echo "$(date '+%F %T') START $t"
  case $fam in
    mil)   $PY $X/retrain.py mil --feature "$a" --corpus "$b" ;;
    fixed) $PY $X/retrain.py fixed --method "$a" --corpus "$b" ;;
    mhl)   $PY $X/retrain.py mhl --corpus "$a" ;;
    sage)  sage "$a" ;;
    clara) clara "$a" "$b" ;;
    *)     echo "unknown task $t"; false ;;
  esac
  st=$?
  if [ $st -eq 0 ]; then echo "$(date '+%F %T') DONE $t"; else echo "$(date '+%F %T') FAILED $t rc=$st"; rc=1; fi
done
echo "$(date '+%F %T') ALL TASKS FINISHED rc=$rc"
exit $rc
