#!/usr/bin/env bash
#SBATCH --job-name=bl-av-lavad-ib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=runs/20261008_baselines/lavad_av/slurm_%j.out
# Audio-visual (transcript) variants of the label-free baselines (user decision 2026-10-09), GPU parts, uoa-campus2:
#   imagebind    ZS-ImageBind-AV vision channel: verify the port of the campaign's 4 fps image channel on 5 HateMM
#                cohort videos, then extract it for the 1341 DeHate test videos (staged temporarily in
#                runs/20261008_baselines/zs_imagebind_av/_staging_videos/DeHate/, deleted after the job)
#                -> runs/20261008_baselines/zs_imagebind_av/features/imagebind_image_4fps/DeHate/ (copied back to
#                uoa-lab1 data/imagebind_image_4fps/DeHate/, then removed here)
#   <corpus>     LAVAD-AV on HateClipSeg / HateMM / DeHate: reuse check (HateClipSeg, DeHate), new scoring stage
#                (Llama-2-13b-chat NF4, hate rules + speech transcript), curves, exact-cohort finalisation
#                -> runs/20261008_baselines/lavad_av/<corpus>/
# Same env as the visual-only LAVAD campus run (campus_lavad.sbatch: .cache/envs/lavad_tf449, transformers 4.49.0).
# Every step skips finished videos, so resubmitting continues an interrupted run.
# Submit on uoa-campus2:
#   cd /data/jehc223/Hate-follow-up && mkdir -p runs/20261008_baselines/lavad_av && \
#   sbatch experiments/20261008_baselines/launch/campus_av.sbatch [imagebind] [HateClipSeg] [HateMM] [DeHate]
set -uo pipefail
cd /data/jehc223/Hate-follow-up
set +u; source /data/jehc223/miniconda3/bin/activate /data/jehc223/Hate-follow-up/.cache/envs/lavad_tf449; set -u
source experiments/20261008_baselines/launch/campus_env.sh
$LAVAD_PYTHON -c "import transformers, tokenizers; assert (transformers.__version__, tokenizers.__version__) == ('4.49.0', '0.21.4'), (transformers.__version__, tokenizers.__version__)" \
  || { echo "FAILED LAVAD env"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
IB=experiments/20261008_baselines/zs_imagebind_audio/zs_imagebind_av.py
L=experiments/20261008_baselines/lavad/lavad_av.py
for S in ${@:-imagebind HateClipSeg HateMM DeHate}; do
  echo "=== [$(date -Is)] step $S"
  case $S in
    imagebind)
      $LAVAD_PYTHON $IB verify-image --video-root /data/jehc223/HateMM/video || echo "FAILED verify-image (see run.log)"
      $LAVAD_PYTHON $IB extract-image --video-root runs/20261008_baselines/zs_imagebind_av/_staging_videos/DeHate \
        --out-dir runs/20261008_baselines/zs_imagebind_av/features/imagebind_image_4fps/DeHate --decoders 6 \
        || echo "FAILED extract-image (see run.log)"
      echo "IMAGEBIND_DONE $(date -Is)" ;;
    HateMM|HateClipSeg|DeHate)
      if [ "$S" != HateMM ]; then   # the HateMM reuse check needs the campaign curves on uoa-lab1 (done there)
        $LAVAD_PYTHON $L check-reuse --datasets $S || { echo "FAILED check-reuse $S"; continue; }
      fi
      $LAVAD_PYTHON $L score --datasets $S --batch-size 48 || { echo "FAILED score $S"; continue; }
      $LAVAD_PYTHON $L curves --datasets $S || { echo "FAILED curves $S"; continue; }
      $LAVAD_PYTHON $L finalize --datasets $S || echo "FAILED finalize $S (see run.log)"
      echo "CORPUS_DONE $S $(date -Is)" ;;
    *) echo "FAILED unknown step $S" ;;
  esac
done
echo "JOB_DONE $(date -Is)"
