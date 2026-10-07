# Sourced by the campus_*.sbatch jobs (uoa-campus2 = foscsmlprd02, A100 80G; 2026-10-08). Campus counterpart of
# lab2_env.sh: same Python packages, models read in offline mode from the repository's Hugging Face cache
# (`.cache/hf`, filled with `hf download` or symlinks; see the method READMEs). Every cache and temp file stays in the
# repository (the campus $HOME has a 100M quota).
R=/data/jehc223/Hate-follow-up
export HF_HOME=$R/.cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME=$R/.cache/xdg
export CUDA_CACHE_PATH=$R/.cache/nv
export TRITON_CACHE_DIR=$R/.cache/triton
export MPLCONFIGDIR=$R/.cache/mpl
export TORCH_HOME=$R/.cache/torch
export TMPDIR=$R/.cache/tmp
mkdir -p "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$TORCH_HOME" "$TMPDIR"
export PYTHONPATH=$R
# Qwen jobs: campus2's HateVLM env (torch 2.11.0+cu128, transformers 5.15.1; same package versions as lab2's HateVLM
# apart from pip and packaging).
export PY_VLM=/data/jehc223/miniconda3/envs/HateVLM/bin/python
# LAVAD: Python 3.11.8 env with lab2's HateVideo package versions + transformers 4.49.0 / tokenizers 0.21.4
# (campus_lavad_env.sh). It also stands in for lab2's HateVideo env in the cohort-id, finalize and evaluator steps
# (same numpy 1.26.4 and scikit-learn 1.5.2).
export LAVAD_PYTHON=$R/.cache/envs/lavad_tf449/bin/python
export PY_VIDEO=$LAVAD_PYTHON
export EVAL_PYTHON=$LAVAD_PYTHON
echo "job ${SLURM_JOB_ID:-none} host $(hostname) start $(date -Is) commit $(git -C $R rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" || true
$PY_VLM -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
