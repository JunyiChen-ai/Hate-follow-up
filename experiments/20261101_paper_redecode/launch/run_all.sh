#!/usr/bin/env bash
# Paper re-decode of the cached reads with the current method (README of this directory). CPU only, uoa-lab1.
# Usage: setsid nohup bash experiments/20261101_paper_redecode/launch/run_all.sh <stage> > runs/20261101_paper_redecode/launch_<stage>.out 2>&1 &
#   stage repro    : the method on the adopted reads (must equal the stored decodes), one corpus per process, timed
#   stage decode   : every other arm (ablations, smoothing, modality, intervals, read sets, key scale), in parallel
#   stage analysis : paired bootstrap over videos (analyze_dvd.py; within_boot.py = copy of analyze.py's within bootstrap)
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${ANALYSIS_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
NP="${NP:-8}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=runs/20261101_paper_redecode
mkdir -p $OUT/main $OUT/dehate
STAGE=$1
echo "host $(hostname) date $(date -Is) commit $(git rev-parse --short HEAD) stage $STAGE" | tee -a $OUT/launch.log
T=experiments/20260926_twolevel/twolevel_r2.py
G=experiments/20260922_til/til_infer.py
TL="--duration bma --bma-prior length --min-windows 2 --bma-grid 6"
M="--noleak --transform nscore --key raw $TL"            # the current method: r6_bma, raw key
# read sets (identity checked by identity_check.py, README §2)
MAIN_BOTH=runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned
MAIN_OLD=runs/20260926_glr/base_gridA
MAIN_FRAMES=runs/20261007_m1_streamingtom/placement_controls_main/adjacent_local
MAIN_WORDS=runs/20261014_m1_audio_bound/vl_aligned/vl_r6_aligned
DH=runs/20261031_window_evidence/DeHate
DH_BOTH=$DH/adopted; DH_OLD=$DH/r6_replay; DH_FRAMES=$DH/frames; DH_WORDS=$DH/words
MAIN_GATE="--gate-visual runs/20260928_infer/main/reads_noctx --gate-speech runs/20260928_infer/main/reads_noframes"
DH_GATE="--gate-visual runs/20260928_infer/dehate/reads_noctx --gate-speech runs/20260928_infer/dehate/reads_noframes"

jobs_for() {  # set name, adopted reads, old, frames, words, gate flags, dataset flags
  local s=$1 R=$2 RO=$3 RF=$4 RW=$5 GF=$6 DS=$7
  local C="$DS --out-root $OUT/$s"
  local P="$PY $T"
  j() { local tag=$1; shift; echo "$* --tag $tag > $OUT/$s/$tag.out 2>&1 || echo JOB_FAILED $s $tag"; }
  # task 1: decoder and integrator ablations
  j full           $P $M --arm full --run $R $C
  j nocoupling     $P $M --arm m2 --nocoupling --run $R $C
  j k1             $P --noleak --transform nscore --key raw --duration bma --bma-prior length --min-windows 0.5 --bma-grid 6 --arm m2 --run $R $C
  j fixed80        $P $M --arm m2 --bma-fixed 80 --run $R $C
  j fixed80_k4     $P --noleak --transform nscore --key raw --k 4 --arm m2 --run $R $C          # old DeHate flags (r3_m2); must equal fixed80
  j sharedchain    $P $M --arm m2 --sharedchain --run $R $C
  j rawscale       $P --noleak --transform none --key raw $TL --arm m2 --run $R $C
  j r2nl           $P --noleak --k 4 --arm m2 --run $R $C                                       # EM, raw reads, fixed 80 s, raw key (old base of the evidence pair)
  j linstd         $P --evidence linear --k 4 --fusion or --arm m2 --run $R $C                  # reads / corpus s.d., fixed 80 s (old diag_lin_or_k4)
  j lin_bma_raw    $P --evidence linear --transform none --key raw $TL --arm m2 --run $R $C     # extra: reads / s.d. under the current chain (pair: rawscale)
  j lin_bma_nscore $P --noleak --evidence linear --transform nscore --key raw $TL --arm m2 --run $R $C  # extra: normal scores / s.d. (pair: m2)
  j nokey          $P --noleak --transform nscore --key none $TL --arm m2 --run $R $C
  j norank         $P $M --arm m2 --norank --run $R $C
  # task 3: modality rows
  j speechonly     $P $M --arm m2 --chains z_speech --run $R $C
  j visualonly     $P $M --arm m2 --chains z_visual --run $R $C
  j gate           $P $M --arm m2 --gate verdicts $GF --run $R $C
  # task 5: read sets
  j rs_old         $P $M --arm m2 --run $RO $C
  j rs_frames      $P $M --arm m2 --run $RF $C
  j rs_words       $P $M --arm m2 --run $RW $C
  # task 6: key scale
  for sc in 100 8 4 2 1 0.5 0.25 0.125; do
    j s$sc         $P --noleak --transform nscore --key scaled --key-scale $sc $TL --arm m2 --run $R $C
  done
  # task 2: Gaussian smoothing of the same reads (raw key, as final_gauss*)
  for sg in 2 4 8 16; do
    j gauss$sg     $PY $G --runs $R --model gauss --sigma $sg --fusion max $C
  done
}

if [ "$STAGE" = repro ]; then
  for spec in "main HateMM" "main HateClipSeg" "dehate DeHate"; do
    set -- $spec; s=$1; ds=$2; R=$MAIN_BOTH; [ $s = dehate ] && R=$DH_BOTH
    /usr/bin/time -v -o $OUT/$s/time_$ds.txt $PY $T $M --arm m2 --run $R --datasets $ds --out-root $OUT/$s --tag m2_$ds \
      > $OUT/$s/m2_$ds.out 2>&1 || echo JOB_FAILED $s m2_$ds
  done
  $PY $T $M --arm m2 --run $MAIN_BOTH --datasets HateMM HateClipSeg --out-root $OUT/main --tag m2 --selftest \
    > $OUT/main/m2.out 2>&1 || echo JOB_FAILED main m2
  ln -sfn m2_DeHate $OUT/dehate/m2
elif [ "$STAGE" = decode ] || [ "$STAGE" = jobs ]; then
  { jobs_for main $MAIN_BOTH $MAIN_OLD $MAIN_FRAMES $MAIN_WORDS "$MAIN_GATE" "--datasets HateMM HateClipSeg"
    jobs_for dehate $DH_BOTH $DH_OLD $DH_FRAMES $DH_WORDS "$DH_GATE" "--datasets DeHate"; } > $OUT/jobs_decode.txt
  # DeHate jobs first (longest), then the main ones
  [ "$STAGE" = jobs ] && exit 0
  { grep "/dehate/" $OUT/jobs_decode.txt; grep -v "/dehate/" $OUT/jobs_decode.txt; } | xargs -P $NP -d '\n' -I{} bash -c '{}'
elif [ "$STAGE" = extra ]; then
  # the key-scale sweep on the old (r6) reads under the current decoder, for comparison with the adopted reads
  for sc in 100 8 4 2 1 0.5 0.25 0.125; do
    for spec in "main $MAIN_OLD HateMM HateClipSeg" "dehate $DH_OLD DeHate"; do
      set -- $spec; s=$1; R=$2; shift 2
      echo "$PY $T --noleak --transform nscore --key scaled --key-scale $sc $TL --arm m2 --run $R --datasets $* --out-root $OUT/$s --tag old_s$sc > $OUT/$s/old_s$sc.out 2>&1 || echo JOB_FAILED $s old_s$sc"
    done
  done > $OUT/jobs_extra.txt
  xargs -P $NP -d '\n' -I{} bash -c '{}' < $OUT/jobs_extra.txt
elif [ "$STAGE" = analysis ]; then
  A=experiments/20260927_dvd/analyze_dvd.py
  W=experiments/20261101_paper_redecode/within_boot.py
  for s in main dehate; do
    DS="--datasets HateMM HateClipSeg"; [ $s = dehate ] && DS="--datasets DeHate"
    $PY $A --root $OUT/$s $DS --arms m2 nocoupling k1 fixed80 sharedchain rawscale lin_bma_nscore nokey norank \
      --out $OUT/$s/analysis_ablation
    $PY $A --root $OUT/$s $DS --arms m2 speechonly visualonly gate --out $OUT/$s/analysis_modality
    $PY $A --root $OUT/$s $DS --arms rs_old rs_frames rs_words m2 --out $OUT/$s/analysis_readsets
    $PY $W --root $OUT/$s $DS --base m2 --arms nocoupling k1 fixed80 sharedchain rawscale lin_bma_nscore norank \
      gauss2 gauss4 gauss8 gauss16 speechonly visualonly gate rs_old rs_frames rs_words --out $OUT/$s/analysis_within
    $PY $W --root $OUT/$s $DS --base r2nl --arms linstd --out $OUT/$s/analysis_within_evidence_fixed80
    $PY $W --root $OUT/$s $DS --base rawscale --arms lin_bma_raw --out $OUT/$s/analysis_within_evidence_bma_raw
  done
fi
echo "STAGE_DONE $STAGE $(date -Is)" | tee -a $OUT/launch.log
