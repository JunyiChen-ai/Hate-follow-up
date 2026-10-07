#!/usr/bin/env bash
#SBATCH --job-name=bl-qwen-text-winonly
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=runs/20261008_baselines/qwen3_8b_text/slurm_%j.out
# Campus copy of lab2_text_winonly.sbatch (lab Slurm 303; moved because the lab account may run only 2 GPU jobs):
#   1. Qwen3-8B on Whisper transcript segments, HateMM, HateClipSeg, DeHate (= lab2_qwen3_text.sbatch);
#   2. Qwen2.5-VL-7B winonly arm of spvl.py on DeHate, compose.py, finalize (= lab2_winonly_dehate.sbatch).
# Same scripts, flags and models; only the repository path and the environment (campus_env.sh) differ.
# Submit on uoa-campus2: cd /data/jehc223/Hate-follow-up && sbatch experiments/20261008_baselines/launch/campus_qwen.sbatch
set -uo pipefail
cd /data/jehc223/Hate-follow-up
set +u; source /data/jehc223/miniconda3/bin/activate HateVLM; set -u   # conda activate scripts read unset variables
source experiments/20261008_baselines/launch/campus_env.sh

$PY_VLM experiments/20261008_baselines/qwen3_text/text_llm.py --datasets HateMM HateClipSeg DeHate \
  || echo "FAILED text_llm (see run.log)"
echo "TEXT_DONE $(date -Is)"

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
echo "JOB_DONE_ALL $(date -Is)"
