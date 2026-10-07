# DSANet (AAAI 2026), weakly supervised, 2026-10-08

Upstream: `third_party/DSANet` (@ eb335b2). Port used unchanged: `scripts/reproduction_baselines/dsanet/` (patches in
`scripts/reproduction_baselines/PATCHES.md`), run through `../weaksup_common/port.py`, which points it at
`data/weaksup_1fps/`. Shared splits, labels, 4 fps rule and evaluation: `../weaksup_common/README.md`.

Mechanism hypothesis: none (reference baseline; video-level labels of the target corpus's train split).

## Setup

| | |
|---|---|
| input | `clip_b16_1fps`: CLIP ViT-B/16 image embedding of the frame at t = i s, one row per second (upstream: one CLIP row per 16-frame snippet) |
| classes / prompts | 2: "normal content", "hateful content" (`dsanet/descriptions.py` DESCRIPTIONS_HATE) |
| hyper-parameters | the upstream XD-Violence preset (`src/xd_option.py`) on all three corpora: embed 512, width 512, 1 head, 1 layer, visual-length 256, attn-window 64, prompt 10/10, decoder_depth 8, normal_selection_ratio .8, DNP on, 16 prototypes, text_adapt_until 1, t_w .6, loss2_weight 5, temp 1, 10 epochs, batch 96, lr 1e-5, warm-up 100; classes 7 → 2 |
| selection | none: the last of the 10 epochs is kept; the validation split is not read |
| score | `score_mlp` = sigmoid(logits1 / temp), headline (the branch the earlier DeHate run reported; with two classes upstream's refined score equals it); `score_align` = 1 − softmax(logits2)[:, 0] reported as `dsanet_align` |
| seeds | 2025 / 234 / 3407 |
| environment | conda HateVideo, torch 2.7.1 + cu128, RTX 5090 (Slurm). DSANet's pinned torch 2.0.1 / cu117 does not support the 5090 |

**What the original script did and what changed.** Upstream `xd_train.py` evaluates the test set during training,
saves the checkpoint with the best TEST frame AP and reloads it every epoch. Here (coordinator instruction
2026-10-08) the test set is never opened during training, no checkpoint is reloaded, and the last epoch of the fixed
10-epoch XD schedule is kept on every corpus (`--select last`, `port.py --no-val`). Other port changes (binary
description table, stale text-feature cache fix D4, removed `ipdb` import, 1 fps rows without ×16 upsampling) are in
PATCHES.md (V1–V7, D1–D4, T1, O1). No further code change was needed for torch 2.7.

**DeHate.** The existing DeHate DSANet run (Retrieval-hate on uoa-lab1/lab2, 2026-09-26) passed the split and label
checks (5-trial Optuna and epoch selection on validation video AP; train 4680 / val 668 = ours) but is not reused:
its hyper-parameters were tuned, not the XD preset used here on HateMM and HateClipSeg. Its re-evaluation is kept
for reference only in `runs/20261008_baselines/_superseded/dsanet_DeHate_reused_optuna/` (seed mean ROC .6325 /
PR .1207 / within .4873, equal to `runs/20260927_dehate_external/weaksup/`). DeHate is rerun with the XD preset.

Splits: HateMM 744 / 109 / 215, HateClipSeg (p11) 237 / 39 / 118, DeHate 4680 / 668 / 1151 (val unused);
(train ∪ val) ∩ test = ∅ (`runs/20261008_baselines/weaksup_inputs/split_check.json`).

## How to run

```bash
# uoa-lab3, inside Slurm: launch/lab3_dsanet_vadclip.sbatch
python experiments/20261008_baselines/weaksup_common/run_method.py --method dsanet --corpus {hatemm,hateclipseg,dehate}
```

Outputs: `runs/20261008_baselines/dsanet/<Dataset>/seed<k>/` and `.../<Dataset>/summary.json`.

## Results

RESULTS_PLACEHOLDER
