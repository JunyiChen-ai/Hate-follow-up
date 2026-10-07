#!/bin/bash
#SBATCH --job-name=clara-win8
#SBATCH --partition=local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --output=runs/20261008_baselines/clara/slurm_%x_%j.out
# CLARA per-window baseline, one corpus per job, after the CPU step
#   python experiments/20261008_baselines/clara/clara_prep.py --dataset <DS>   (setsid nohup; clips + frames)
# Stages (each skips what is already done, so a resubmission resumes):
#   1 video-level rationale (vLLM, Qwen3-VL-8B)   2 OCR of the remaining clip folders on GPU (CPU fallback)
#   3 embeddings   4 training: rationale mode both and none (w/o Rationale check) x seeds 2025/234/3407,
#     window scores, coverage check, evaluator.
# Usage: sbatch experiments/20261008_baselines/launch/clara_lab1.sbatch <DS>
set -uo pipefail
DS=${1:?dataset}
cd "$SLURM_SUBMIT_DIR"
C=$PWD/.cache
export HF_HOME=$C/hf HF_HUB_OFFLINE=1 TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg VLLM_CACHE_ROOT=$C/vllm \
  TORCHINDUCTOR_CACHE_DIR=$C/torchinductor TRITON_CACHE_DIR=$C/triton PADDLE_PDX_CACHE_HOME=$C/paddlex \
  PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True MPLCONFIGDIR=$C/mpl VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  PYTHONUNBUFFERED=1 OPENCV_LOG_LEVEL=ERROR
mkdir -p $C/tmp $C/xdg $C/vllm $C/mpl
R=runs/20261008_baselines/clara/$DS
X=experiments/20261008_baselines/clara
echo "host $(hostname) job $SLURM_JOB_ID dataset $DS $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
grep -q "DONE prep" $R/prep/run.log || { echo "FAILED: run clara_prep.py first"; exit 1; }
if ! grep -q "DONE rationale" $R/rationale/run.log 2>/dev/null; then
  $C/envs/vllm_q3vl/bin/python $X/clara_rationale.py --dataset $DS || { echo "FAILED rationale"; exit 1; }
fi
$C/envs/paddleocr/bin/python $X/clara_ocr.py --dataset $DS --device gpu:0 --reverse --shard 0 --nshards 1 \
  || { echo "OCR on GPU failed; CPU fallback"; OMP_NUM_THREADS=8 $C/envs/paddleocr/bin/python $X/clara_ocr.py --dataset $DS --device cpu --reverse || { echo "FAILED ocr"; exit 1; }; }
$C/envs/hv_cv2/bin/python $X/clara_embed.py --dataset $DS || { echo "FAILED embed"; exit 1; }
for mode in both none; do
  for s in 2025 234 3407; do
    d=runs/20261008_baselines/clara/$DS/seed$s; [ $mode = none ] && d=runs/20261008_baselines/clara_norationale/$DS/seed$s
    if grep -q "DONE clara" $d/run.log 2>/dev/null; then echo "$mode seed $s done"; continue; fi
    $C/envs/hv_cv2/bin/python $X/clara_train.py --dataset $DS --seed $s --rationale-mode $mode || { echo "FAILED train $mode $s"; exit 1; }
  done
done
echo "DONE clara $DS $(date)"
