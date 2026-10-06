#!/usr/bin/env bash
set -euo pipefail
cd /home/jehc223/Hate-follow-up
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/home/jehc223/Hate-follow-up/.cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
args=()
case ${SCOPE:-main} in
  smoke) args+=(--smoke);;
  main) ;;
  *) echo FAILED_invalid_scope; exit 2;;
esac
out="runs/20261006_m1_rekv/controls_${SCOPE:-main}_analysis"
mkdir -p "$out"
exec >>"$out/run.log" 2>&1
hostname
printf '%s\n' "$$" >"$out/run.pid"
.cache/envs/HateVLM/bin/python -u experiments/20261006_m1_rekv/controls_analyze.py --stage prepare "${args[@]}"
if [[ ${SCOPE:-main} == smoke ]]; then exit 0; fi
source /home/jehc223/miniconda3/bin/activate HateVideo
for arm in R0 L0 C0 L1 C1 L2 D0 T0 H0; do
  python -u experiments/20261006_m1_rekv/controls_analyze.py --stage evaluate --name "$arm"
done
python -u experiments/20261006_m1_rekv/controls_analyze.py --stage report
