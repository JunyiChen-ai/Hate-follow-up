#!/bin/bash
# Environments for the SAGE / CLARA window baselines (2026-10-08), built inside the repo on each lab machine.
# Usage (repo root, CPU only, setsid nohup): bash experiments/20261008_baselines/clara/setup_envs.sh
# All three venvs use the HateVideo conda python 3.11 (torch 2.7.1+cu128 in the base env):
#   .cache/envs/hv_cv2     --system-site-packages HateVideo + opencv-python-headless (SAGE, CLARA prep/embed/train)
#   .cache/envs/vllm_q3vl  vllm 0.11.0 (torch 2.8.0+cu128) + transformers 4.57.6 (CLARA rationale, Qwen3-VL-8B)
#   .cache/envs/paddleocr  paddlepaddle-gpu 3.2.0 (cu129 wheel) + paddleocr 3.4.0 (CLARA OCR)
# The authors pin torch 2.6+cu124 (SAGE requirements.txt), which has no RTX 5090 (sm_120) kernels; cu128 builds
# are used instead. Pip downloads go to .cache/pip and are purged at the end.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
BASEPY=$HOME/miniconda3/envs/HateVideo/bin/python
export PIP_CACHE_DIR=$PWD/.cache/pip TMPDIR=$PWD/.cache/tmp
mkdir -p .cache/tmp .cache/pip
L=runs/20261008_baselines/_setup_$(hostname)
mkdir -p "$L"
echo "host $(hostname) $(date)" > "$L/setup_envs.log"
if [ ! -x .cache/envs/hv_cv2/bin/python ]; then
  $BASEPY -m venv --system-site-packages .cache/envs/hv_cv2
  .cache/envs/hv_cv2/bin/pip install --no-deps opencv-python-headless==4.12.0.88 >> "$L/setup_envs.log" 2>&1
fi
if [ ! -x .cache/envs/vllm_q3vl/bin/vllm ]; then
  $BASEPY -m venv .cache/envs/vllm_q3vl
  .cache/envs/vllm_q3vl/bin/pip install -U pip >> "$L/setup_envs.log" 2>&1
  .cache/envs/vllm_q3vl/bin/pip install vllm==0.11.0 opencv-python-headless==4.12.0.88 >> "$L/setup_envs.log" 2>&1
  .cache/envs/vllm_q3vl/bin/pip install transformers==4.57.6 >> "$L/setup_envs.log" 2>&1
fi
if [ ! -d .cache/envs/paddleocr/lib/python3.11/site-packages/paddleocr ]; then
  $BASEPY -m venv .cache/envs/paddleocr
  .cache/envs/paddleocr/bin/pip install -U pip >> "$L/setup_envs.log" 2>&1
  .cache/envs/paddleocr/bin/pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/ >> "$L/setup_envs.log" 2>&1
  .cache/envs/paddleocr/bin/pip install paddleocr==3.4.0 opencv-python-headless==4.12.0.88 >> "$L/setup_envs.log" 2>&1
fi
.cache/envs/hv_cv2/bin/python -c "import cv2, torch, transformers; print('hv_cv2', cv2.__version__, torch.__version__, transformers.__version__)" >> "$L/setup_envs.log" 2>&1
.cache/envs/vllm_q3vl/bin/python -c "import vllm, torch, transformers; print('vllm_q3vl', vllm.__version__, torch.__version__, transformers.__version__)" >> "$L/setup_envs.log" 2>&1
.cache/envs/paddleocr/bin/python -c "import paddle, paddleocr; print('paddleocr', paddle.__version__, paddleocr.__version__)" >> "$L/setup_envs.log" 2>&1
.cache/envs/hv_cv2/bin/pip cache purge >> "$L/setup_envs.log" 2>&1
echo "DONE setup $(date)" >> "$L/setup_envs.log"
