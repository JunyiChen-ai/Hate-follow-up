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

Seed mean ± sd (n − 1) over seeds 2025 / 234 / 3407, transcribed from `runs/20261008_baselines/dsanet/<Dataset>/summary.json`, which is built from each seed's `metrics.json` (canonical evaluator). Pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC.

| row | corpus | ROC | PR | within | per seed (ROC / PR / within; 2025, 234, 3407) | host |
|---|---|---:|---:|---:|---|---|
| DSANet (score_mlp, headline) | HateMM | 0.7005 ± 0.0142 | 0.4136 ± 0.0546 | 0.5290 ± 0.0427 | 0.6844/0.3754/0.4976; 0.7115/0.4762/0.5776; 0.7055/0.3893/0.5118 | sc474398 |
| DSANet (score_mlp, headline) | HateClipSeg | 0.5080 ± 0.0178 | 0.4622 ± 0.0119 | 0.5024 ± 0.0188 | 0.5261/0.4703/0.5125; 0.5073/0.4679/0.5140; 0.4906/0.4485/0.4808 | sc474398 |
| DSANet (score_mlp, headline) | DeHate | 0.6404 ± 0.0151 | 0.1363 ± 0.0268 | 0.4776 ± 0.0126 | 0.6574/0.1662/0.4737; 0.6350/0.1145/0.4916; 0.6287/0.1282/0.4674 | sc474398 |
| DSANet score_align | HateMM | 0.6862 ± 0.0105 | 0.4479 ± 0.0182 | 0.5600 ± 0.0055 | 0.6873/0.4311/0.5539; 0.6752/0.4672/0.5646; 0.6962/0.4452/0.5614 | sc474398 |
| DSANet score_align | HateClipSeg | 0.5349 ± 0.0269 | 0.5053 ± 0.0098 | 0.5074 ± 0.0391 | 0.5646/0.5166/0.5455; 0.5279/0.4995/0.4674; 0.5121/0.4998/0.5093 | sc474398 |
| DSANet score_align | DeHate | 0.6655 ± 0.0117 | 0.1398 ± 0.0098 | 0.5229 ± 0.0096 | 0.6766/0.1467/0.5245; 0.6666/0.1442/0.5316; 0.6532/0.1286/0.5126 | sc474398 |

Host: uoa-lab3 (sc474398), Slurm 290, 2026-10-08 (results rsynced to uoa-lab1 without the model checkpoints, which stay on uoa-lab3).

## Change for oracle test selection (2026-10-09)

`scripts/reproduction_baselines/dsanet/train.py` accepts `--save-every-epoch` (default off): it also writes `model_eNN.pth` after every completed epoch. Saving draws no random number, so training is unchanged. Used only by `../oracle_test_selection/` (checkpoint and branch chosen on TEST labels, an upper bound for the baselines; see that README). The rows above are unchanged.
