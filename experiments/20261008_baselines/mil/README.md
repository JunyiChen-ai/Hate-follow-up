# BERT + MIL, wav2vec2 + MIL, CLIP + MIL (weakly supervised, 2026-10-08)

Three single-modality rows with one MIL head (run_plan.md §3 W1–W3; user decision D1 = top-k BCE). The rows differ
only in the input feature. Shared splits, labels, 4 fps rule and evaluation: `../weaksup_common/README.md`.

Mechanism hypothesis: none (reference baselines; video-level labels of the target corpus's train split).

## Setup (`train_mil.py`)

| | |
|---|---|
| instances | one row per second (1 fps). Text: `bert_sentence_1fps` (bert-base CLS of the Whisper segment covering the second, zeros in silence). Audio: `wav2vec2_base_1s` (facebook/wav2vec2-base, pre-trained, last hidden layer, mean over each second). Visual: `clip_b16_1fps` (CLIP ViT-B/16 image embedding of the frame at t = i s) |
| head | Sultani et al. MLP d → 512 → 32 → 1, ReLU, dropout 0.6, sigmoid (same for all three) |
| loss | top-k MIL BCE (Wu et al., ECCV 2020): video score = mean of the top floor(T/16)+1 instance scores; BCE vs the video label |
| training | Adam, lr 1e-4, weight decay 5e-4, 64 videos per batch (padded, masked), 50 epochs; no hyper-parameter search |
| selection | epoch with the best validation video AP (top-k mean vs val video labels; val split only) |
| output | per-second instance score of each cohort video (`score_mil`) → ×4, last-value pad to ceil(4D) |
| seeds | 2025 / 234 / 3407 |

Splits: HateMM 744 / 109 / 215, HateClipSeg (p11) 237 / 39 / 118, DeHate 4680 / 668 / 1151; (train ∪ val) ∩ test = ∅
(`runs/20261008_baselines/weaksup_inputs/split_check.json`). HCS val has 5 normal videos, so its epoch choice is noisy.

## How to run

```bash
# uoa-lab1, inside Slurm (launch/lab1_mil_mhl_avadclip.sbatch runs all nine corpus x feature tasks)
python experiments/20261008_baselines/mil/train_mil.py --feature {bert,wav2vec2,clip} --corpus {hatemm,hateclipseg,dehate}
```

Outputs: `runs/20261008_baselines/mil_<feature>/<Dataset>/seed<k>/{run.log,config.json,scores.jsonl,predictions.jsonl,coverage.json,metrics.json,train_meta.json}` and `.../<Dataset>/summary.json`.

## Results

RESULTS_PLACEHOLDER
