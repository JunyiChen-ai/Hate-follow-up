#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
arm="${1:?control arm required}"
version="${2:-r1}"
case "$arm" in uniform|distance|fixed4|mismatch|verdict_replay) ;; *) exit 2 ;; esac
case "$version" in r1|r2|r3|r4) ;; *) exit 2 ;; esac
if [[ "$arm" = verdict_replay && "$version" != r4 ]]; then exit 2; fi
name=controls
if [[ "$version" != r1 ]]; then name="${version}_controls"; fi
out="runs/20261003_m1_explorer/$name/$arm"
mkdir -p "$out"
exec > "$out/analysis_run.log" 2>&1
echo "host $(hostname)"
echo $$ > "$out/analysis_run.pid"
trap 'echo CONTROL_ANALYSIS_FAILED' ERR
py=/home/jehc223/miniconda3/envs/HateVideo/bin/python
runner=experiments/20261003_m1_explorer/analyze_controls.py
for stage in prepare evaluate report; do
    "$py" -u "$runner" --stage "$stage" --arm "$arm" --version "$version"
done
