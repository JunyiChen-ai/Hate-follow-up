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

Seed mean ± sd (n − 1) over seeds 2025 / 234 / 3407, transcribed from `runs/20261008_baselines/mil_bert/<Dataset>/summary.json`, which is built from each seed's `metrics.json` (canonical evaluator). Pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC.

| row | corpus | ROC | PR | within | per seed (ROC / PR / within; 2025, 234, 3407) | host |
|---|---|---:|---:|---:|---|---|
| BERT + MIL | HateMM | 0.6362 ± 0.0309 | 0.3835 ± 0.0440 | 0.5253 ± 0.0148 | 0.6712/0.4343/0.5419; 0.6125/0.3582/0.5205; 0.6250/0.3582/0.5134 | sc474397 |
| BERT + MIL | HateClipSeg | 0.5442 ± 0.0085 | 0.5138 ± 0.0070 | 0.5044 ± 0.0108 | 0.5377/0.5058/0.4935; 0.5411/0.5164/0.5046; 0.5538/0.5191/0.5151 | sc474397 |
| BERT + MIL | DeHate | 0.6456 ± 0.0036 | 0.1730 ± 0.0035 | 0.5526 ± 0.0027 | 0.6428/0.1769/0.5526; 0.6496/0.1719/0.5554; 0.6444/0.1701/0.5500 | sc474397 |

Seed mean ± sd (n − 1) over seeds 2025 / 234 / 3407, transcribed from `runs/20261008_baselines/mil_wav2vec2/<Dataset>/summary.json`, which is built from each seed's `metrics.json` (canonical evaluator). Pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC.

| row | corpus | ROC | PR | within | per seed (ROC / PR / within; 2025, 234, 3407) | host |
|---|---|---:|---:|---:|---|---|
| wav2vec2 + MIL | HateMM | 0.6919 ± 0.0090 | 0.4422 ± 0.0046 | 0.5570 ± 0.0057 | 0.6959/0.4444/0.5602; 0.6983/0.4452/0.5604; 0.6816/0.4369/0.5505 | sc474397 |
| wav2vec2 + MIL | HateClipSeg | 0.4973 ± 0.0017 | 0.4718 ± 0.0036 | 0.4775 ± 0.0011 | 0.4956/0.4678/0.4779; 0.4973/0.4746/0.4762; 0.4990/0.4730/0.4783 | sc474397 |
| wav2vec2 + MIL | DeHate | 0.5418 ± 0.0002 | 0.0821 ± 0.0003 | 0.5293 ± 0.0094 | 0.5420/0.0820/0.5285; 0.5419/0.0825/0.5390; 0.5417/0.0819/0.5202 | sc474397 |

Seed mean ± sd (n − 1) over seeds 2025 / 234 / 3407, transcribed from `runs/20261008_baselines/mil_clip/<Dataset>/summary.json`, which is built from each seed's `metrics.json` (canonical evaluator). Pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC.

| row | corpus | ROC | PR | within | per seed (ROC / PR / within; 2025, 234, 3407) | host |
|---|---|---:|---:|---:|---|---|
| CLIP + MIL | HateMM | 0.7191 ± 0.0036 | 0.4585 ± 0.0057 | 0.5229 ± 0.0027 | 0.7152/0.4571/0.5222; 0.7222/0.4537/0.5259; 0.7199/0.4648/0.5207 | sc474397 |
| CLIP + MIL | HateClipSeg | 0.5583 ± 0.0035 | 0.5182 ± 0.0035 | 0.5249 ± 0.0035 | 0.5547/0.5143/0.5241; 0.5616/0.5189/0.5218; 0.5585/0.5212/0.5287 | sc474397 |
| CLIP + MIL | DeHate | 0.6418 ± 0.0032 | 0.1312 ± 0.0012 | 0.5202 ± 0.0047 | 0.6395/0.1307/0.5250; 0.6455/0.1304/0.5156; 0.6404/0.1325/0.5202 | sc474397 |

Hosts: uoa-lab1 (sc474397), Slurm 291, 2026-10-08. Selected epochs (seeds 2025 / 234 / 3407), from each seed's `train_meta.json`: BERT HateMM 50/2/7, HCS 22/32/49, DeHate 21/29/34; wav2vec2 HateMM 50/50/50, HCS 42/49/38, DeHate 45/45/50; CLIP HateMM 38/31/50, HCS 50/50/46, DeHate 23/24/26. wav2vec2 on HateMM always takes the last epoch, so 50 epochs at lr 1e-4 may under-train it; the recipe is kept fixed across rows by design.

Coverage: every cohort video scored with a finite value on every GT frame (`coverage.json`); tail padding at most 3 frames on HateMM, 28 on HCS, 237 on DeHate (one DeHate video whose feature rows end at the audio length).
