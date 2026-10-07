#!/usr/bin/env bash
# Builds the LAVAD env `.cache/envs/lavad_tf449` on a campus server (2026-10-08).
#
# On uoa-lab2 this env is `python -m venv --system-site-packages` on the HateVideo conda env (Python 3.11.8,
# torch 2.7.1+cu128) with only transformers 4.49.0 and tokenizers 0.21.4 installed into it
# (runs/_setup_uoa-lab2/lavad_tf449_venv.log). The campus server used (uoa-campus2) has no HateVideo env, so this
# script builds a standalone Python 3.11.8 env that holds the same versions of every package the LAVAD port
# (BLIP-2, Llama-2 NF4 via bitsandbytes, ImageBind) and the evaluator import: `campus_lavad_env_constraints.txt`
# is lab2's HateVideo package list with transformers 4.49.0 / tokenizers 0.21.4, used as pip constraints.
# Everything (conda packages, pip cache, temp files, dotfiles) stays inside the repository.
#
#   cd /data/jehc223/Hate-follow-up && setsid nohup bash experiments/20261008_baselines/launch/campus_lavad_env.sh \
#       uoa-campus2 > runs/_setup_uoa-campus2/lavad_tf449_env.log 2>&1 < /dev/null &
set -euo pipefail
MACHINE=${1:?machine alias, e.g. uoa-campus2}
R=/data/jehc223/Hate-follow-up
cd "$R"
export HOME=$R/.cache/setup_home CONDA_PKGS_DIRS=$R/.cache/conda_pkgs PIP_CACHE_DIR=$R/.cache/pip TMPDIR=$R/.cache/tmp
mkdir -p "$HOME" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
E=$R/.cache/envs/lavad_tf449
C=$R/experiments/20261008_baselines/launch/campus_lavad_env_constraints.txt
echo "host $(hostname) start $(date -Is) commit $(git rev-parse --short HEAD)"
[ -x "$E/bin/python" ] || /data/jehc223/miniconda3/bin/conda create -y -p "$E" -c conda-forge --override-channels python=3.11.8
"$E/bin/python" -m pip install -c "$C" --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio
"$E/bin/python" -m pip install -c "$C" --extra-index-url https://download.pytorch.org/whl/cu128 \
  transformers tokenizers accelerate bitsandbytes safetensors huggingface_hub sentencepiece protobuf \
  timm ftfy regex iopath fvcore pytorchvideo einops numpy scipy scikit-learn pillow av
"$E/bin/python" -m pip list --format=freeze > "$R/runs/_setup_$MACHINE/lavad_tf449_pip_freeze.txt"
"$E/bin/python" -m pip cache purge || true
rm -rf "$CONDA_PKGS_DIRS"   # the env's files are hard links or copies; the package cache is not needed afterwards
echo "ENV_DONE $(date -Is)"
