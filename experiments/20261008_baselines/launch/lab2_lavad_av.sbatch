#!/usr/bin/env bash
#SBATCH --job-name=bl-lavad-av
#SBATCH --partition=local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/20261008_baselines/lavad_av/slurm_%j.out
# LAVAD-AV on uoa-lab2 (RTX 5090), one corpus per job (default HateMM): scoring stage with the hate rules + speech
# transcript (lavad_av.py), curves, exact-cohort finalisation. Same env as the lab2 LAVAD runs (lab2_env.sh:
# .cache/envs/lavad_tf449, transformers 4.49.0). HateMM runs here because its visual-only row is the campaign's
# uoa-lab2 RTX 5090 run; the campus job (campus_av.sbatch, Slurm 24651) skips it through
# runs/20261008_baselines/lavad_av/elsewhere_HateMM.txt on uoa-campus2. Resubmitting continues an interrupted run.
# HateClipSeg (2026-10-11): the campus job restarted after its hold with DeHate only, so HateClipSeg runs here too,
# reading the visual-only campus2 files from a copy (second argument, lavad_av.py --visual-run).
# Submit on uoa-lab2: cd ~/Hate-follow-up && mkdir -p runs/20261008_baselines/lavad_av && \
#   sbatch experiments/20261008_baselines/launch/lab2_lavad_av.sbatch [CORPUS [VISUAL_RUN_COPY]]
set -uo pipefail
cd /home/jehc223/Hate-follow-up
source experiments/20261008_baselines/launch/lab2_env.sh
$LAVAD_PYTHON -c "import transformers, tokenizers; assert (transformers.__version__, tokenizers.__version__) == ('4.49.0', '0.21.4'), (transformers.__version__, tokenizers.__version__)" \
  || { echo "FAILED LAVAD env"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=experiments/20261008_baselines/lavad/lavad_av.py
DS=${1:-HateMM}
VR=${2:-}
if [ -n "$VR" ]; then
  $LAVAD_PYTHON $L check-reuse --datasets $DS --visual-run $VR || { echo "FAILED check-reuse $DS"; exit 1; }
fi
$LAVAD_PYTHON $L score --datasets $DS --batch-size 48 ${VR:+--visual-run $VR} || { echo "FAILED score $DS"; exit 1; }
$LAVAD_PYTHON $L curves --datasets $DS ${VR:+--visual-run $VR} || { echo "FAILED curves $DS"; exit 1; }
$LAVAD_PYTHON $L finalize --datasets $DS ${VR:+--visual-run $VR} || echo "FAILED finalize $DS (see run.log)"
echo "CORPUS_DONE $DS $(date -Is)"
echo "JOB_DONE $(date -Is)"
