#!/bin/bash
# Direct (non-Slurm) runner for the duplex kill-test probe.
# Prerequisites: HVD_DATA_ROOT set, vllm==0.11.0 installed, one GPU visible.
# See docs/duplex/NEW_MACHINE_RUNBOOK.md.
set -e
cd "$(dirname "$0")/../.."

: "${HVD_DATA_ROOT:?set HVD_DATA_ROOT to the directory that contains ImpliHateVid/}"

echo "=== duplex probe: 2B, ImpliHateVid train, 5 readers, frames_16 mode === $(date)"
python src/duplex/score_duplex_probe.py \
  --dataset ImpliHateVid --split train \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --batch-size 16 --gpu-mem 0.90 --no-video

echo "=== duplex probe: 8B, ImpliHateVid train, 5 readers, frames_16 mode === $(date)"
python src/duplex/score_duplex_probe.py \
  --dataset ImpliHateVid --split train \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --batch-size 8 --gpu-mem 0.90 --no-video

echo "=== analysis ==="
mkdir -p docs/duplex/reports
python src/duplex/analyze_duplex_probe.py --model-slug qwen3-vl-2b-instruct \
  --json-out docs/duplex/reports/killtest_2b_train_frames.json
python src/duplex/analyze_duplex_probe.py --model-slug qwen3-vl-8b-instruct \
  --json-out docs/duplex/reports/killtest_8b_train_frames.json

echo "=== DONE === $(date)"
