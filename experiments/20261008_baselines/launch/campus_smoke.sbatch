#!/usr/bin/env bash
#SBATCH --job-name=bl-campus-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=runs/20261008_baselines/_smoke_campus/slurm_%j.out
# Smoke test of campus_qwen.sbatch and campus_lavad.sbatch on uoa-campus2 (same scripts, envs and models), 2 videos
# per workload: Qwen3-8B text on the first 2 cohort videos of each corpus (no evaluation), Qwen2.5-VL winonly on the
# first 2 DeHate manifest videos, LAVAD on HateClipSeg bit_0nXuyV2rypaf (also in the lab2 rerun check, Slurm 294) and
# DeHate 4PmH5EgjyduF (25 s). Outputs the scripts write to fixed paths (qwen3_8b_text/, lavad/work, lavad/curves,
# data/blip2_captions_1fps) are moved into runs/20261008_baselines/_smoke_campus/ after the job, so the real jobs
# start from empty output directories.
set -uo pipefail
cd /data/jehc223/Hate-follow-up
set +u; source /data/jehc223/miniconda3/bin/activate HateVLM; set -u   # conda activate scripts read unset variables
source experiments/20261008_baselines/launch/campus_env.sh
SM=runs/20261008_baselines/_smoke_campus

$PY_VLM experiments/20261008_baselines/qwen3_text/text_llm.py --datasets HateMM HateClipSeg DeHate --limit 2 \
  && echo "SMOKE_OK text_llm" || echo "FAILED smoke text_llm"
$PY_VLM experiments/20260910_spvl/spvl.py --exp-id 20261008_baselines --run-name _smoke_campus/qwen25vl_winonly_DeHate \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --method-name q25vl-7b__spvl_f0_noctx_fixed8_block_stnone_brjoint_qrules_fsk20_isocache \
  --isolation cache --frames 20 --windows fixed --window-seconds 8 \
  --frames 0 --no-transcript-context --branches joint --window-question rules \
  --datasets DeHate --manifest data/manifests/DeHate_test.jsonl --limit 2
grep -q "DONE videos" $SM/qwen25vl_winonly_DeHate/run.log && echo "SMOKE_OK spvl" || echo "FAILED smoke spvl"
$PY_VLM experiments/20260910_spvl/compose.py --run-dir $SM/qwen25vl_winonly_DeHate --intercept spvl --residual rank \
  --datasets DeHate && echo "SMOKE_OK compose" || echo "SMOKE_WARN compose/evaluator on 2 videos"

set +u; source /data/jehc223/miniconda3/bin/activate /data/jehc223/Hate-follow-up/.cache/envs/lavad_tf449; set -u   # conda activate scripts read unset variables
$LAVAD_PYTHON -c "import transformers, tokenizers, torch; print('lavad env', transformers.__version__, tokenizers.__version__, torch.__version__)"
echo bit_0nXuyV2rypaf > $SM/lavad_ids_HateClipSeg.txt
echo 4PmH5EgjyduF > $SM/lavad_ids_DeHate.txt
for DS in HateClipSeg DeHate; do
  bash experiments/20261008_baselines/lavad/run_chain.sh $DS $SM/lavad_ids_$DS.txt \
    && echo "SMOKE_OK lavad $DS" || echo "FAILED smoke lavad $DS"
done
echo "SMOKE_DONE $(date -Is)"
