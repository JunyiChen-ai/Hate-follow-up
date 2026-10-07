# MultiHateLoc (reimplemented), weakly supervised, 2026-10-08

MultiHateLoc has no official code. The row is **our reimplementation**, `scripts/reproduction_baselines/multihateloc/`
(design notes in its `DESIGN.md`), run unchanged through `../weaksup_common/port.py`, which points it at
`data/weaksup_1fps/`. Shared splits, labels, 4 fps rule and evaluation: `../weaksup_common/README.md`.

Mechanism hypothesis: none (reference baseline; video-level labels of the target corpus's train split).

## Setup

| | |
|---|---|
| input (1 fps) | visual `vit_b16_imagenet_1fps` (768, ImageNet ViT-B/16, the encoder the paper cites), audio `vggish_1s` (128, VGGish on 1 s windows), text `bert_sentence_1fps` (768, BERT CLS of the Whisper segment covering each second). 1 fps matches VGGish's 1 s audio features, so no interpolation is needed |
| output | per-second `score_fused` (fused branch), headline; ×4 to the 4 fps grid |
| protocol | the one of the existing DeHate run (Retrieval-hate `experiments/20260926_dehate_external/launch/baseline.sh`): Optuna TPE, 5 completed trials, sampler seed 234, each trial at seed 234; objective = the trial's selected validation video AP; search space and batch ≥ 64 guard of `scripts/reproduction_baselines/tune_official_val.py`; the winner retrained at seeds 2025 / 234 / 3407, epoch selected on validation video AP (val split, video-level labels) |
| environment | conda HateVideo, torch 2.7.1 + cu128, RTX 5090 (Slurm) |

**HateClipSeg.** The earlier HCS MultiHateLoc number in the main table is leaked (its checkpoint was trained on a list
holding 95 of our 118 test videos; run_plan §2.1). It is retrained here on p11 (train 237 / val 39). p11 val has 5
normal videos, so the validation-AP objective and epoch choice are noisy on HCS.

**HateMM.** Retrained with the same protocol (replaces the earlier DMS-branch, single-seed row).

**DeHate: reused** (`../weaksup_common/reuse_dehate.py`). The run of 2026-09-26 (Retrieval-hate, uoa-lab2,
`runs/20260926_dehate_external/baselines/{tuning,final}/multihateloc/dehate/`) passed every check, recorded in
`runs/20261008_baselines/multihateloc/DeHate/reuse_check.json`:
- 5 completed trials (6 more pruned by the batch guard); best.json value = the winning trial's validation AP (.5390);
- every trial and seed trained on 4680 train and selected on 668 val videos (= our splits; (train ∪ val) ∩ cohort = ∅),
  `select = val_ap`; seeds 234 / 2025 / 3407 use the winner's configuration;
- test scores cover all 1151 cohort videos with a finite value on every GT frame after ×4 and last-value padding
  (29 videos padded, 823 frames, at most 237 frames: the 1 fps curve ends at the audio length).
Records copied read-only to `runs/20261008_baselines/multihateloc/DeHate/source_lab2/`.

Splits: HateMM 744 / 109 / 215, HateClipSeg (p11) 237 / 39 / 118, DeHate 4680 / 668 / 1151;
(train ∪ val) ∩ test = ∅ (`runs/20261008_baselines/weaksup_inputs/split_check.json`).

## How to run

```bash
# uoa-lab1, inside Slurm: launch/lab1_mil_mhl_avadclip.sbatch
python experiments/20261008_baselines/weaksup_common/run_method.py --method multihateloc --corpus {hatemm,hateclipseg}
# DeHate reuse (CPU)
python experiments/20261008_baselines/weaksup_common/reuse_dehate.py
```

Outputs: `runs/20261008_baselines/multihateloc/<Dataset>/seed<k>/`, `.../<Dataset>/tuning/`, `.../<Dataset>/summary.json`.

## Results

RESULTS_PLACEHOLDER
