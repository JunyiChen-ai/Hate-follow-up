#!/usr/bin/env bash
#SBATCH --job-name=bl-winonly-dehate
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261008_baselines/qwen25vl_winonly/slurm_%j.out
# Qwen2.5-VL-7B winonly arm of experiments/20260910_spvl/spvl.py on DeHate (run_plan.md L8): same script and flags
# as launch/run_mllm.sh `winonly`, then compose.py (--intercept spvl --residual rank), then the exact-cohort finalise.
set -uo pipefail
cd /home/jehc223/Hate-follow-up
source experiments/20261008_baselines/launch/lab2_env.sh
RUN=runs/20261008_baselines/qwen25vl_winonly/DeHate_spvl
$PY_VLM experiments/20260910_spvl/spvl.py --exp-id 20261008_baselines --run-name qwen25vl_winonly/DeHate_spvl \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --method-name q25vl-7b__spvl_f0_noctx_fixed8_block_stnone_brjoint_qrules_fsk20_isocache \
  --isolation cache --frames 20 --windows fixed --window-seconds 8 \
  --frames 0 --no-transcript-context --branches joint --window-question rules \
  --datasets DeHate --manifest data/manifests/DeHate_test.jsonl
grep -q "DONE videos" $RUN/run.log || { echo "FAILED spvl (no DONE line)"; exit 1; }
$PY_VLM experiments/20260910_spvl/compose.py --run-dir $RUN --intercept spvl --residual rank --datasets DeHate \
  || { echo "FAILED compose"; exit 1; }
$PY_VLM experiments/20261008_baselines/qwen25vl_winonly/finalize.py --datasets DeHate || echo "FAILED finalize"
echo "JOB_DONE $(date -Is)"
