#!/usr/bin/env bash
# Builds the PRISM env `.cache/envs/prism` on a campus server (2026-10-09), the counterpart of uoa-lab3's env
# (runs/_setup_uoa-lab3/prism_env.sh): Python 3.12, torch 2.8.0+cu128, torchvision 0.23.0, transformers 4.57.1,
# qwen-vl-utils 0.0.14, decord 0.6.0. `campus_prism_env_constraints.txt` is `pip freeze` of the uoa-lab3 env, used as
# pip constraints, so every package has the lab3 version. The model files are not downloaded here: the uoa-lab3
# snapshot of Qwen/Qwen3-VL-Embedding-2B is copied with rsync (prism/README.md), so both runs read the same files.
# Everything (conda packages, pip cache, temp files, dotfiles) stays inside the repository.
#
#   cd /data/jehc223/Hate-follow-up && mkdir -p runs/_setup_uoa-campus3 && setsid nohup bash \
#     experiments/20261008_baselines/launch/campus_prism_env.sh uoa-campus3 > runs/_setup_uoa-campus3/prism_env.log 2>&1 < /dev/null &
set -euo pipefail
MACHINE=${1:?machine alias, e.g. uoa-campus3}
R=/data/jehc223/Hate-follow-up
cd "$R"
export HOME=$R/.cache/setup_home CONDA_PKGS_DIRS=$R/.cache/conda_pkgs PIP_CACHE_DIR=$R/.cache/pip TMPDIR=$R/.cache/tmp
mkdir -p "$HOME" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
E=$R/.cache/envs/prism
C=$R/experiments/20261008_baselines/launch/campus_prism_env_constraints.txt
CONDA=${CONDA:-/data/jehc223/home/miniconda3/bin/conda}      # campus3: the conda under /data/jehc223/home
echo "machine $MACHINE host $(hostname) start $(date -Is) commit $(git rev-parse --short HEAD) conda $CONDA"
[ -x "$E/bin/python" ] || "$CONDA" create -y -p "$E" -c conda-forge --override-channels python=3.12
"$E/bin/python" -m pip install -c "$C" --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple torch torchvision
"$E/bin/python" -m pip install -c "$C" transformers accelerate qwen-vl-utils decord av scipy numpy pillow huggingface_hub
"$E/bin/python" -c "import torch, transformers, qwen_vl_utils, decord, scipy; print(torch.__version__, transformers.__version__, decord.__version__)"
"$E/bin/python" -m pip freeze > runs/_setup_$MACHINE/prism_env_freeze.txt
diff <(grep -v -E '^(torch|torchvision)==' "$C") <(grep -v -E '^(torch|torchvision)==' runs/_setup_$MACHINE/prism_env_freeze.txt) \
  && echo "package versions identical to uoa-lab3 (torch/torchvision checked by the import line above)" \
  || echo "WARNING: package versions differ from uoa-lab3 (diff above)"
# ffmpeg for lf_common.transcode_h264 (no system ffmpeg on uoa-campus3; added 2026-10-10 after two HateClipSeg webm
# files failed with FileNotFoundError: 'ffmpeg'). Separate env so the PRISM env stays as built.
F=$R/.cache/envs/ffmpeg
[ -x "$F/bin/ffmpeg" ] || "$CONDA" create -y -p "$F" -c conda-forge --override-channels "ffmpeg=6.1"
"$F/bin/ffmpeg" -version | head -1
echo ENV_DONE $(date -Is)
