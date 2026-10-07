#!/usr/bin/env bash
#SBATCH --job-name=bl-lavad-hcs-dehate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=runs/20261008_baselines/lavad/slurm_%j.out
# Campus copy of lab2_lavad_hcs_full.sbatch followed by lab2_lavad_dehate.sbatch, in one job, one after the other:
# LAVAD on all 118 HateClipSeg cohort videos (full rerun), then on the 1151 DeHate cohort videos. Same port, prompt,
# models and precision (BLIP-2 opt-6.7b-coco fp16, Llama-2-13b-chat-hf NF4 with bf16 compute through the llama_hf shim,
# ImageBind-Huge), transformers 4.49.0 / tokenizers 0.21.4 (campus_lavad_env.sh); only the repository path, the
# environment activation and the GPU (A100 80G instead of an RTX 5090) differ. Every stage skips finished videos, so
# resubmitting continues an interrupted run.
# Submit on uoa-campus2: cd /data/jehc223/Hate-follow-up && sbatch experiments/20261008_baselines/launch/campus_lavad.sbatch [CORPUS ...]
# 2026-10-08: split into one job per corpus (HateClipSeg, DeHate) so the two corpora run in parallel; each corpus
# still runs whole in one job.
set -uo pipefail
cd /data/jehc223/Hate-follow-up
set +u; source /data/jehc223/miniconda3/bin/activate /data/jehc223/Hate-follow-up/.cache/envs/lavad_tf449; set -u   # conda activate scripts read unset variables
source experiments/20261008_baselines/launch/campus_env.sh
$LAVAD_PYTHON -c "import transformers, tokenizers; assert (transformers.__version__, tokenizers.__version__) == ('4.49.0', '0.21.4'), (transformers.__version__, tokenizers.__version__)" \
  || { echo "FAILED LAVAD env"; exit 1; }
for DS in ${@:-HateClipSeg DeHate}; do   # corpora from the command line, default both in order
  IDS=runs/20261008_baselines/lavad/ids_$DS.txt
  $PY_VIDEO -c "
import sys; sys.path.insert(0, 'experiments/20261008_baselines'); import exact_cohort as ec
open('$IDS', 'w').write('\n'.join(ec.cohort('$DS')) + '\n')" || { echo "FAILED ids $DS"; exit 1; }
  bash experiments/20261008_baselines/lavad/run_chain.sh $DS "$IDS" || { echo "FAILED chain $DS"; exit 1; }
  if [ "$DS" = HateClipSeg ]; then
    $PY_VIDEO experiments/20261008_baselines/lavad/finalize.py --dataset HateClipSeg --all-rerun || echo "FAILED finalize HateClipSeg (see run.log)"
  else
    $PY_VIDEO experiments/20261008_baselines/lavad/finalize.py --dataset DeHate || echo "FAILED finalize DeHate (see run.log)"
  fi
  echo "CORPUS_DONE $DS $(date -Is)"
done
echo "JOB_DONE $(date -Is)"
