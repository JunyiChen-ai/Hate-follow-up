#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PWD/.cache/hf"
phase="${1:-main}"
revision="${2:-r1}"
args=()
if [ "$phase" = smoke ]; then args+=(--smoke); elif [ "$phase" != main ]; then exit 2; fi
if [ "$revision" = r2 ]; then args+=(--future-keys text); elif [ "$revision" != r1 ]; then exit 2; fi
case "$(hostname -s)" in
  sc474399) py=/home/jehc223/miniconda3/envs/HateVLM/bin/python ;;
  sc448960) py=/home/junyi/miniconda3/envs/HateVLM/bin/python ;;
  *) echo 'Use a reviewed target environment' >&2; exit 2 ;;
esac
exec "$py" -u archive/experiments/20261003_m1_integrator/measure.py --run-name "${revision}_$phase" "${args[@]}"
