# Environment of the reruns on the Retrieval-hate HateClipSeg split (README.md). Source from the repository root:
#   source experiments/20261115_rh_hcs_split/env.sh
# Every variable is an override read by the existing baseline code; unset, that code behaves as before.
RH_DATA=$PWD/data/rh_hcs_split
RH_RUNS=$PWD/runs/20261115_rh_hcs_split
export WEAKSUP_ALT_DIR=$RH_DATA/weaksup WEAKSUP_RUNS=$RH_RUNS
export DETWIN_ALT_COHORT=1 DETWIN_SPLIT_DIR=$RH_DATA/video_splits DETWIN_RUNS=$RH_RUNS SAGE_DATA=$RH_DATA CLARA_DATA=$RH_DATA
export LF_ALT_COHORT_DIR=$RH_DATA/cohort LF_BASELINES_RUNS=$RH_RUNS LF_RUN_ROOT=$RH_RUNS/lf
mkdir -p $RH_RUNS
