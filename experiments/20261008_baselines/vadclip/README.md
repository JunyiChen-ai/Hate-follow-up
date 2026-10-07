# VadCLIP (AAAI 2024), weakly supervised, 2026-10-08

Upstream: `third_party/VadCLIP` (github.com/nwpu-zxr/VadCLIP @ c41067f). Port used unchanged:
`scripts/reproduction_baselines/vadclip/` (patches in `scripts/reproduction_baselines/PATCHES.md`), run through
`../weaksup_common/port.py`, which points it at `data/weaksup_1fps/`. Shared splits, labels, 4 fps rule and
evaluation: `../weaksup_common/README.md`.

Mechanism hypothesis: none (reference baseline; video-level labels of the target corpus's train split).

## Setup

| | |
|---|---|
| input | `clip_b16_1fps`: CLIP ViT-B/16 image embedding of the frame at t = i s, one row per second (upstream: one CLIP row per 16-frame XD snippet) |
| classes / prompts | 2: "normal content", "hateful content" (`hate_common.PROMPT_TEXT`) |
| hyper-parameters | the upstream XD-Violence preset (`src/xd_option.py`) on all three corpora: embed 512, width 512, 1 head, 1 layer, visual-length 256, attn-window 64, prompt 10/10, 10 epochs, batch 96, lr 1e-5, MultiStepLR [3, 6, 10] × 0.1, loss3 weight 1e-4; classes 7 → 2 |
| selection | none: the last of the 10 epochs is kept; the validation split is not read |
| score | `score_align` = 1 − softmax(logits2)[:, 0] (upstream's AP branch), headline; `score_mlp` = sigmoid(logits1) reported as `vadclip_mlp` |
| seeds | 2025 / 234 / 3407 |
| environment | conda HateVideo, torch 2.7.1 + cu128, RTX 5090 (Slurm) |

**What the original script did and what changed.** Upstream `xd_train.py` calls `test()` on the test set after every
epoch, saves the checkpoint with the best TEST frame AP and reloads it at the end of every epoch; the reported model
is that test-selected checkpoint. Here (coordinator instruction 2026-10-08) the test set is never opened during
training, no checkpoint is reloaded, and the last epoch of the fixed 10-epoch XD schedule is kept on every corpus
(`--select last`, `port.py --no-val`). Other port changes (binary class prompts, 1 fps rows with no ×16
upsampling, orthogonality normaliser /(classes − 1), logging) are listed in PATCHES.md (V1–V7, T1, O1). No code change
was needed for torch 2.7.

Splits: HateMM 744 / 109 / 215, HateClipSeg (p11) 237 / 39 / 118, DeHate 4680 / 668 / 1151 (val unused);
(train ∪ val) ∩ test = ∅ (`runs/20261008_baselines/weaksup_inputs/split_check.json`).

## How to run

```bash
# uoa-lab3, inside Slurm: launch/lab3_dsanet_vadclip.sbatch
python experiments/20261008_baselines/weaksup_common/run_method.py --method vadclip --corpus {hatemm,hateclipseg,dehate}
```

Outputs: `runs/20261008_baselines/vadclip/<Dataset>/seed<k>/` and `.../<Dataset>/summary.json`.

## Results

Seed mean ± sd (n − 1) over seeds 2025 / 234 / 3407, transcribed from `runs/20261008_baselines/vadclip/<Dataset>/summary.json`, which is built from each seed's `metrics.json` (canonical evaluator). Pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC.

| row | corpus | ROC | PR | within | per seed (ROC / PR / within; 2025, 234, 3407) | host |
|---|---|---:|---:|---:|---|---|
| VadCLIP (score_align, headline) | HateMM | 0.6111 ± 0.0434 | 0.3588 ± 0.0435 | 0.4783 ± 0.0365 | 0.5784/0.3110/0.4697; 0.5947/0.3694/0.5184; 0.6604/0.3959/0.4469 | sc474398 |
| VadCLIP (score_align, headline) | HateClipSeg | 0.5308 ± 0.0246 | 0.4825 ± 0.0256 | 0.5124 ± 0.0181 | 0.5510/0.4923/0.5274; 0.5380/0.5016/0.5174; 0.5033/0.4534/0.4923 | sc474398 |
| VadCLIP (score_align, headline) | DeHate | 0.6035 ± 0.0046 | 0.1091 ± 0.0049 | 0.5120 ± 0.0109 | 0.6041/0.1111/0.5000; 0.6077/0.1128/0.5213; 0.5986/0.1036/0.5147 | sc474398 |
| VadCLIP score_mlp | HateMM | 0.6822 ± 0.0226 | 0.3998 ± 0.0562 | 0.4640 ± 0.0356 | 0.6578/0.3522/0.4263; 0.6865/0.4619/0.4971; 0.7024/0.3854/0.4686 | sc474398 |
| VadCLIP score_mlp | HateClipSeg | 0.5001 ± 0.0114 | 0.4580 ± 0.0163 | 0.5126 ± 0.0209 | 0.4932/0.4485/0.5053; 0.5133/0.4768/0.5361; 0.4939/0.4486/0.4963 | sc474398 |
| VadCLIP score_mlp | DeHate | 0.6276 ± 0.0150 | 0.1167 ± 0.0122 | 0.4619 ± 0.0047 | 0.6396/0.1301/0.4566; 0.6108/0.1061/0.4644; 0.6326/0.1139/0.4648 | sc474398 |

Host: uoa-lab3 (sc474398), Slurm 290, 2026-10-08 (results rsynced to uoa-lab1 without the model checkpoints, which stay on uoa-lab3 under the same paths). The VadCLIP rows have the largest seed spread on HateMM (sd .04): with a fixed last epoch and no selection, seeds vary more.
