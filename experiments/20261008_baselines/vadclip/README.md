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

RESULTS_PLACEHOLDER
