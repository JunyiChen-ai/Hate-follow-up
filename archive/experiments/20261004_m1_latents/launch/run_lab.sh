#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$PWD/.cache/hf"
phase="${1:-smoke}"
arm="${2:-full}"
args=(--arm "$arm")
if [ "$phase" = smoke ]; then args+=(--smoke); elif [ "$phase" != main ]; then exit 2; fi
case "$(hostname -s)" in
  sc474399|sc474398) py=/home/jehc223/miniconda3/envs/HateVLM/bin/python ;;
  sc448960) py=/home/junyi/miniconda3/envs/HateVLM/bin/python ;;
  *) echo 'Use a reviewed target environment' >&2; exit 2 ;;
esac
exec "$py" -u experiments/20261004_m1_latents/measure.py "${args[@]}"
