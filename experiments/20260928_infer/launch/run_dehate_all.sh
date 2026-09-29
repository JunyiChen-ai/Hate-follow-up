#!/usr/bin/env bash
# README §12: every leave-one-out ablation of r6_bma on DeHate. GPU reads for the four reading-module ablations
# (no frames, no transcript context, no stance turn, ASR-segment windows), then the CPU time-level ablations on the
# isolated dual reads, then the paired-bootstrap tables. One lab machine (not uoa-lab1).
# Needs: HateVLM env (reads), HateVideo env (time level), data/manifests/DeHate_test.jsonl, data/frames_k20/DeHate,
# data/asr_whisper_large_v3/DeHate, data/gt_4fps/DeHate.npz, runs/20260927_dehate_external/reads_gridA,
# runs/20260926_twolevel/final_dehate/final_m2, and the §11 runs m1_joint / m1_seq in the same output root.
set -euo pipefail
cd "$(dirname "$0")/../../.."
RPY="${READS_PYTHON:-$HOME/miniconda3/envs/HateVLM/bin/python}"
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
OUT=runs/20260928_infer/dehate
mkdir -p $OUT
echo "host $(hostname) date $(date -Is) commit $(git rev-parse --short HEAD)" | tee -a $OUT/launch_all.log
ln -sfn ../../20260926_twolevel/final_dehate/final_m2 $OUT/final_m2
ln -sfn ../../20260927_dehate_external/reads_gridA $OUT/reads_gridA
run_reads() {  # name, flags...   (til_measure.py resumes from its own predictions.jsonl)
  local name=$1; shift
  local d=$OUT/reads_$name; mkdir -p $d
  if grep -q "DONE videos" $d/launch.log 2>/dev/null; then echo "READS_SKIP $name" | tee -a $OUT/launch_all.log; return; fi
  "$RPY" experiments/20260922_til/til_measure.py --exp-id 20260928_infer --run-name dehate/reads_$name --window-offset 0 \
    --datasets DeHate --manifest data/manifests/DeHate_test.jsonl "$@" >> $d/launch.log 2>&1
  grep -q "DONE videos" $d/launch.log || { echo "RUN_FAILED reads_$name" | tee -a $OUT/launch_all.log; exit 1; }
  echo "READS_DONE $name $(date -Is)" | tee -a $OUT/launch_all.log
}
run_reads noframes --frames 0
run_reads noctx --no-transcript-context
run_reads nostance --stance none
run_reads asrwin --windows asr
T=experiments/20260926_twolevel/twolevel_r2.py
C="--out-root $OUT --datasets DeHate --arm m2"
TL="--duration bma --bma-prior length --min-windows 2 --bma-grid 6"
S="$T --noleak --transform nscore --key calib $TL $C"
R="--run $OUT/reads_gridA"
# time-level ablations on the isolated dual reads (same flags as experiments/20260926_twolevel/launch/run_final.sh)
"$PY" $S $R --nocoupling --tag abl_nocoupling
"$PY" $T --noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 0.5 --bma-grid 6 $C $R --tag abl_k1
"$PY" $S $R --sharedchain --tag abl_sharedchain
"$PY" $T --noleak --transform none --key calib $TL $C $R --tag abl_rawscale
"$PY" $T --noleak --transform nscore --key raw $TL $C $R --tag abl_rawkey
"$PY" $T --noleak --transform nscore --key none $TL $C $R --tag abl_nokey
"$PY" $S $R --norank --tag abl_norank
"$PY" $T --noleak --transform nscore --key calib --k 4 $C $R --tag abl_fixed80          # r3_m2: mean length fixed at 80 s
"$PY" $T --noleak --k 4 $C $R --tag abl_r2nl                                            # r2_noleak: raw reads, EM, fixed 80 s, raw key
"$PY" $T --evidence linear --k 4 --fusion or $C $R --tag abl_linstd                     # diag_lin_or_k4: reads / corpus std instead of EM
# reading-module ablations under the r6 time level
for a in noframes noctx nostance; do "$PY" $S --run $OUT/reads_$a --tag abl_$a; done
# ASR-segment windows cannot enter the time level (windows longer than two cells): SPVL composition on both read sets
for a in fixed asrwin; do
  src=$OUT/reads_gridA; [ $a = asrwin ] && src=$OUT/reads_asrwin
  mkdir -p $OUT/compose_$a $OUT/spvl_$a; cp $src/predictions.jsonl $OUT/compose_$a/predictions.jsonl
  "$PY" experiments/20260910_spvl/compose.py --run-dir $OUT/compose_$a --intercept zv_plus_mean --residual rank --datasets DeHate
  ln -sfn ../compose_$a/predictions_izv_plus_mean_rrank.jsonl $OUT/spvl_$a/predictions.jsonl
done
A="experiments/20260927_dvd/analyze_dvd.py --root $OUT --datasets DeHate"
"$PY" $A --arms final_m2 abl_nocoupling abl_k1 abl_sharedchain abl_rawscale abl_rawkey abl_nokey abl_norank abl_fixed80 \
  abl_noframes abl_noctx abl_nostance m1_joint m1_seq --out $OUT/analysis_all
"$PY" $A --arms abl_r2nl abl_linstd --out $OUT/analysis_linstd
"$PY" $A --arms spvl_fixed spvl_asrwin --out $OUT/analysis_asrwin
"$PY" experiments/20260928_infer/reads_agreement.py --base $OUT/reads_gridA \
  --others $OUT/reads_noframes $OUT/reads_noctx $OUT/reads_nostance --out $OUT/analysis_all/reads_agreement.txt
echo ALL_DONE | tee -a $OUT/launch_all.log
