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

Seed mean ± sd (n − 1) over seeds 2025 / 234 / 3407, transcribed from `runs/20261008_baselines/multihateloc/<Dataset>/summary.json` (built from each seed's `metrics.json`, canonical evaluator). Branch `score_fused`.

| corpus | ROC | PR | within | per seed (ROC / PR / within; 2025, 234, 3407) | host | tuning (val video AP of the winner) |
|---|---:|---:|---:|---|---|---|
| HateMM | 0.7441 ± 0.0134 | 0.4946 ± 0.0138 | 0.6228 ± 0.0128 | 0.7537/0.4957/0.6081; 0.7499/0.5080/0.6312; 0.7288/0.4803/0.6292 | sc474397 | trial 9, 0.8741 (5 complete, 6 pruned) |
| HateClipSeg | 0.5185 ± 0.0152 | 0.4787 ± 0.0100 | 0.4983 ± 0.0063 | 0.5168/0.4823/0.5052; 0.5345/0.4864/0.4965; 0.5043/0.4673/0.4930 | sc474397 | trial 10, 0.9947 (5 complete, 6 pruned) |
| DeHate | 0.6102 ± 0.0090 | 0.1289 ± 0.0040 | 0.5420 ± 0.0161 | 0.6050/0.1316/0.5439; 0.6206/0.1244/0.5570; 0.6050/0.1308/0.5250 | sc474397 (re-evaluated); trained on uoa-lab2 2026-09-26 | trial 10, 0.5390 (5 complete, 6 pruned) |

Runs: HateMM and HateClipSeg on uoa-lab1 (sc474397). Slurm 291 completed 4 of 5 trials on each corpus within its
attempt budget (about a third of the suggestions are pruned by the batch ≥ 64 guard) and stopped; Slurm 302 resumed
the same Optuna studies (sampler re-seeded 234 + 10 = 244, as in the DeHate run's resume) and finished the fifth
trial and the three seeds. Selected epochs (seeds 2025 / 234 / 3407): HateMM 21 / 8 / 8, HateClipSeg 11 / 18 / 9.
HateClipSeg's validation AP is near its ceiling for every trial (.98–.995; 34 of 39 val videos are positive), so the
HCS hyper-parameter and epoch choices carry little information. Coverage: every cohort video scored on every GT frame.

Earlier rows replaced: the leaked HCS number (`runs/20260829_omsl_v6/multihateloc_frozen_current4fps_v1_metrics.json`)
and the HateMM DMS-branch single-seed row must not be used.

## Change for oracle test selection (2026-10-09)

`scripts/reproduction_baselines/multihateloc/train.py` accepts `--save-every-epoch` (default off): it also writes `epoch_states/eNNN.pt` after every epoch. Saving draws no random number. For the DeHate retraining, the DeHate ImageNet-ViT and VGGish rows were copied into `data/weaksup_1fps/` (addendum in its PROVENANCE.md). Used only by `../oracle_test_selection/` (checkpoint and branch chosen on TEST labels, an upper bound for the baselines; see that README). The rows above are unchanged.
