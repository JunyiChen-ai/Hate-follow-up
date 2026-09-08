#!/usr/bin/env bash
set -euo pipefail

cd /data/jehc223/EMNLP3

python -m src.our_method.meme_variant.data_utils build --dataset all
python -m src.our_method.meme_variant.validate_outputs --stage processed --dataset all

python -m src.our_method.meme_variant.score_stage1 --dataset all --split all --limit 10 --batch-size 4 --model-slug 2b
python -m src.our_method.meme_variant.validate_outputs --stage stage1 --dataset all --model-slug 2b

python -m src.our_method.meme_variant.threshold_and_band --dataset all --model-slug 2b --criterion gmm
python -m src.our_method.meme_variant.validate_outputs --stage boundary --dataset all

python -m src.our_method.meme_variant.judge_mllm --dataset all --split test --limit 10 --model google/gemma-3-27b-it --model-tag gemma-3-27b-it --batch-size 2 --no-mm-kwargs
python -m src.our_method.meme_variant.judge_mllm --dataset all --split test --limit 10 --model Qwen/Qwen2.5-VL-32B-Instruct-AWQ --model-tag qwen2.5-vl-32b-awq --batch-size 2 --quantization awq
python -m src.our_method.meme_variant.judge_mllm --dataset all --split test --limit 10 --model Qwen/Qwen2.5-VL-72B-Instruct-AWQ --model-tag qwen2.5-vl-72b-awq --batch-size 1 --quantization awq
python -m src.our_method.meme_variant.validate_outputs --stage judges --dataset all

python -m src.our_method.meme_variant.eval_sequential --dataset all
python -m src.our_method.meme_variant.validate_outputs --stage final --dataset all

