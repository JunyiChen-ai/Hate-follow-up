#!/usr/bin/env bash
# Build the conda env `HateVLM` (torch 2.11 cu128 + transformers 5.15.1) used by experiments/20260910_spvl.
# Usage: CONDA_ROOT=~/miniconda3 bash scripts/setup_HateVLM.sh   (log: runs/_setup_<host>/HateVLM_install.log)
# Same package list as the 2026-09-10 uoa-lab3 build. Driver must be >= 570 for cu128 wheels.
set -x
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
source "$CONDA_ROOT/bin/activate"
conda env remove -y -n HateVLM 2>/dev/null
conda create -y -n HateVLM python=3.12
conda activate HateVLM
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.15.1 qwen-vl-utils==0.0.14 accelerate==1.14.0 pillow==12.3.0 numpy==2.5.2 scipy==1.18.1 scikit-learn==1.9.0 safetensors==0.8.0 huggingface_hub==1.28.0
python -c "import torch,transformers;print(torch.__version__, torch.cuda.is_available(), transformers.__version__); import transformers.models.qwen3_vl as q; print('qwen3_vl ok')"
echo SETUP_DONE
