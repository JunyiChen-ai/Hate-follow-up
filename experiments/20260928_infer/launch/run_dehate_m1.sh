#!/usr/bin/env bash
# README §11: two M1 ablations on DeHate (joint branch; branches that see each other), then the r6 time level on each
# read set and the paired-bootstrap analysis against final_m2. GPU + CPU on one lab machine (not uoa-lab1).
# Needs: HateVLM env (reads), HateVideo env (time level), data/manifests/DeHate_test.jsonl, data/frames_k20/DeHate,
# data/asr_whisper_large_v3/DeHate, data/gt_4fps/DeHate.npz, runs/20260926_twolevel/final_dehate/final_m2.
set -euo pipefail
cd "$(dirname "$0")/../../.."
RPY="${READS_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
OUT=runs/20260928_infer/dehate
mkdir -p $OUT
echo "host $(hostname) date $(date -Is) commit $(git rev-parse --short HEAD)" | tee -a $OUT/launch_m1.log
ln -sfn ../../20260926_twolevel/final_dehate/final_m2 $OUT/final_m2
run_reads() {  # name, flags...
  local name=$1; shift
  local d=$OUT/reads_$name; mkdir -p $d
  "$RPY" experiments/20260922_til/til_measure.py --exp-id 20260928_infer --run-name dehate/reads_$name --window-offset 0 \
    --datasets DeHate --manifest data/manifests/DeHate_test.jsonl "$@" >> $d/launch.log 2>&1
  grep -q "DONE videos" $d/launch.log || { echo "RUN_FAILED reads_$name" | tee -a $OUT/launch_m1.log; exit 1; }
  echo "READS_DONE $name $(date -Is)" | tee -a $OUT/launch_m1.log
}
run_reads joint --branches joint
run_reads seq --branches dual --isolation sequential
S="experiments/20260926_twolevel/twolevel_r2.py --noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --out-root $OUT --datasets DeHate --arm m2"
"$PY" $S --run $OUT/reads_joint --tag m1_joint
"$PY" $S --run $OUT/reads_seq --tag m1_seq
"$PY" experiments/20260927_dvd/analyze_dvd.py --root $OUT --arms final_m2 m1_joint m1_seq --datasets DeHate --out $OUT/analysis_m1
"$PY" experiments/20260928_infer/reads_agreement.py --base runs/20260927_dehate_external/reads_gridA \
  --others $OUT/reads_joint $OUT/reads_seq --out $OUT/analysis_m1/reads_agreement.txt
echo ALL_DONE | tee -a $OUT/launch_m1.log
