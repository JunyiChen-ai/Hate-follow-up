#!/usr/bin/env bash
set -euo pipefail

cd /data/jehc223/EMNLP3
mkdir -p logs/meme_variant/slurm

submit_subset() {
  local job_name="$1"
  shift
  local datasets=("$@")

  sbatch --job-name="$job_name" --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 \
    --output=logs/meme_variant/slurm/%x_%j.out \
    --error=logs/meme_variant/slurm/%x_%j.err \
    --export=ALL,MEME_DATASETS="${datasets[*]}" <<'SLURM'
#!/usr/bin/env bash
set -euo pipefail
cd /data/jehc223/EMNLP3
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction

for ds in $MEME_DATASETS; do
  python -m src.our_method.meme_variant.validate_outputs --stage processed --dataset "$ds"

  # Keep train/test in separate Python processes. Re-initializing vLLM inside
  # one process can hang after the first split on this cluster.
  python -m src.our_method.meme_variant.score_stage1 --dataset "$ds" --split train --model-slug 2b --batch-size 8
  python -m src.our_method.meme_variant.score_stage1 --dataset "$ds" --split test --model-slug 2b --batch-size 8
  python -m src.our_method.meme_variant.validate_outputs --stage stage1 --dataset "$ds" --model-slug 2b

  python -m src.our_method.meme_variant.threshold_and_band --dataset "$ds" --model-slug 2b --criterion gmm
  python -m src.our_method.meme_variant.validate_outputs --stage boundary --dataset "$ds"
done

for ds in $MEME_DATASETS; do
  python -m src.our_method.meme_variant.judge_mllm --dataset "$ds" --split test --model google/gemma-3-27b-it --model-tag gemma-3-27b-it --batch-size 2 --no-mm-kwargs
  python -m src.our_method.meme_variant.validate_outputs --stage judges --dataset "$ds" --judge-tags gemma-3-27b-it
done

for ds in $MEME_DATASETS; do
  python -m src.our_method.meme_variant.judge_mllm --dataset "$ds" --split test --model Qwen/Qwen2.5-VL-32B-Instruct-AWQ --model-tag qwen2.5-vl-32b-awq --batch-size 2 --quantization awq
  python -m src.our_method.meme_variant.validate_outputs --stage judges --dataset "$ds" --judge-tags gemma-3-27b-it,qwen2.5-vl-32b-awq
done

for ds in $MEME_DATASETS; do
  python -m src.our_method.meme_variant.judge_mllm --dataset "$ds" --split test --model Qwen/Qwen2.5-VL-72B-Instruct-AWQ --model-tag qwen2.5-vl-72b-awq --batch-size 1 --quantization awq
  python -m src.our_method.meme_variant.validate_outputs --stage judges --dataset "$ds"
done

for ds in $MEME_DATASETS; do
  python -m src.our_method.meme_variant.eval_sequential --dataset "$ds"
  python -m src.our_method.meme_variant.validate_outputs --stage final --dataset "$ds"
done
SLURM
}

# Two independent one-GPU jobs. There is intentionally no dependency between them.
submit_subset meme_seq_a FHM MAMI
submit_subset meme_seq_b ToxiCN_MM
