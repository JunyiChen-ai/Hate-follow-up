#!/usr/bin/env bash
#SBATCH --job-name=prism-av
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=runs/20261008_baselines/prism_av/slurm_%j.out
# PRISM transcript (audio-visual) variant (prism/README.md, "Transcript (audio-visual) variant, 2026-10-09") on a
# campus server: per corpus, window embeddings of frames + window transcript in one input item (prism_hate.py
# extract --av), then the text axes and 4 fps curves (score --av). Same driver, model files, package versions and
# batch as the visual-only run on uoa-lab3 (env: launch/campus_prism_env.sh); only the GPU differs.
# Evaluation runs on uoa-lab1 after rsync (prism_hate.py eval --av).
# Submit on the campus server: cd /data/jehc223/Hate-follow-up && sbatch experiments/20261008_baselines/launch/campus_prism_av.sbatch [CORPUS ...]
# Extraction skips finished videos, so resubmitting continues an interrupted run.
set -uo pipefail
R=/data/jehc223/Hate-follow-up
cd "$R"
export HF_HOME=$R/.cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME=$R/.cache/xdg CUDA_CACHE_PATH=$R/.cache/nv TRITON_CACHE_DIR=$R/.cache/triton \
  MPLCONFIGDIR=$R/.cache/mpl TORCH_HOME=$R/.cache/torch TMPDIR=$R/.cache/tmp PYTHONUNBUFFERED=1
mkdir -p "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$TORCH_HOME" "$TMPDIR"
PY=$R/.cache/envs/prism/bin/python              # torch 2.8.0+cu128, transformers 4.57.1, qwen-vl-utils 0.0.14
# ffmpeg for lf_common.transcode_h264 (videos decord cannot open; 2 HateClipSeg webm files): uoa-campus3 has no system
# ffmpeg, so a separate env holds conda-forge ffmpeg 6.1 (launch/campus_prism_env.sh); uoa-lab3 used /usr/bin/ffmpeg 6.1.1.
export PATH=$R/.cache/envs/ffmpeg/bin:$PATH
command -v ffmpeg >/dev/null || { echo "FAILED no ffmpeg"; exit 1; }
MODEL=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/* | head -1)
echo "host $(hostname) job ${SLURM_JOB_ID:-} start $(date '+%F %T') commit $(git log -1 --format=%h) model $MODEL"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" || true
$PY -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || { echo "FAILED cuda"; exit 1; }
DATASETS=${@:-HateMM HateClipSeg DeHate}
EXTRA=${LIMIT:+--limit $LIMIT}
for DS in $DATASETS; do
  $PY -u experiments/20261008_baselines/prism/prism_hate.py check --av --dataset "$DS" --model "$MODEL" --limit 1 \
    || { echo "FAILED check $DS"; exit 1; }
  $PY -u experiments/20261008_baselines/prism/prism_hate.py extract --av --dataset "$DS" --model "$MODEL" $EXTRA \
    || { echo "FAILED extract $DS"; exit 1; }
  [ -n "${LIMIT:-}" ] && continue
  $PY -u experiments/20261008_baselines/prism/prism_hate.py score --av --dataset "$DS" --model "$MODEL" \
    || { echo "FAILED score $DS"; exit 1; }
  echo "CORPUS_DONE $DS $(date '+%F %T')"
done
echo "PRISM_AV_JOB_DONE $(date '+%F %T')"
