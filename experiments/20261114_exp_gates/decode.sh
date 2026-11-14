#!/usr/bin/env bash
# Decode one read set with the two-level decoder (CPU). Usage: decode.sh <main|dehate> <tag> <reads_dir> [extra twolevel_r2.py flags]
# Base flags = the current method (r6_bma): --noleak --transform nscore --key raw --duration bma --bma-prior length --min-windows 2 --bma-grid 6
set -euo pipefail
cd "$(dirname "$0")/../.."
part=$1; tag=$2; reads=$3; shift 3
root=${EG_ROOT:-runs/20261114_exp_gates/decoder_scan}/$part
mkdir -p "$root"
if [ "$part" = main ]; then ds="HateMM HateClipSeg"; else ds="DeHate"; fi
base="--noleak --transform nscore --key raw --duration bma --bma-prior length --min-windows 2 --bma-grid 6"
if [ "${EG_NOBASE:-0}" = 1 ]; then base=""; fi
/home/jehc223/miniconda3/envs/HateVideo/bin/python experiments/20260926_twolevel/twolevel_r2.py $base --arm m2 "$@" \
  --run "$reads" --datasets $ds --out-root "$root" --tag "$tag" > "$root/$tag.out" 2>&1 || echo "JOB_FAILED $part $tag" >> "$root/FAILED.txt"
