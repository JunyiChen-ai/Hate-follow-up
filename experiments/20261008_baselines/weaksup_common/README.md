# Weakly supervised baselines: shared setup (2026-10-08)

Shared code and rules for the weakly supervised rows of `run_plan.md` §3 (user approval 2026-10-08): BERT / wav2vec2 /
CLIP + MIL (`../mil/`), VadCLIP (`../vadclip/`), DSANet (`../dsanet/`), AVadCLIP (`../avadclip/`) and MultiHateLoc
(`../multihateloc/`). Each method directory has its own README with setup, host and results.

Mechanism hypothesis: none. These are reference baselines trained with video-level labels of the target corpus.

## 1. Splits and labels (`prepare_inputs.py` → `data/weaksup_1fps/`, provenance there)

| corpus | train | val | test (exact cohort) | (train ∪ val) ∩ test | train ∩ val |
|---|---:|---:|---:|---:|---:|
| HateMM | 744 (298 hateful) | 109 (43) | 215 | 0 | 0 |
| HateClipSeg (p11) | 237 (207) | 39 (34) | 118 | 0 | 0 |
| DeHate | 4680 (1484) | 668 (212) | 1151 | 0 | 0 |

Checked by id comparison; record `runs/20261008_baselines/weaksup_inputs/split_check.json`. HCS labels = offensive
union (dimensions 1..5), the HCS GT definition; 0 disagreements with p11's `gold_segments.json` on train+val.
Only train and val video labels are stored. The ports' dataset classes need a label slot for test videos; it is
filled with −1 and never used. Frame labels are never read by training or selection; the test GT is read only by
the evaluator and, for its frame count per video, by the coverage check.

## 2. Features (1 fps; one row = one second)

- `clip_b16_1fps` (512), `bert_sentence_1fps` (768), `vggish_1s` (128), `vit_b16_imagenet_1fps` (768): copies of the
  2026-08 reproduction caches (`data/weaksup_1fps/PROVENANCE.md`).
- `data/wav2vec2_base_1s` (768) and `data/wav2clip_1s` (512): extracted here by `extract_audio.py` (Slurm 285,
  uoa-lab1, 10 min for 7961 videos) from the 16 kHz wavs of Retrieval-hate `data/AV2A_wav/`. Rows are aligned to the
  CLIP rows of the same video (row i = audio [i, i+1) s, padded with zeros or cut to T s). Two HateMM val videos
  (`non_hate_video_559`, `non_hate_video_585`) have no audio stream and get digital silence (run_plan §1.3 F1).

## 3. Training protocols (`run_method.py`)

- **fixed** (VadCLIP, DSANet, AVadCLIP; coordinator instruction 2026-10-08). Each upstream `xd_train*.py` tests on
  the test set after every epoch and keeps the checkpoint with the best TEST frame AP. That is removed. One
  documented hyper-parameter set for all three corpora: the method's XD-Violence preset (`src/xd_option.py`,
  classes 7 → 2), its fixed 10 epochs, the LAST epoch kept. The validation split is not read.
- **tuned** (MultiHateLoc). The protocol of the existing DeHate MultiHateLoc run: 5-trial Optuna (TPE, sampler seed
  234) on validation video AP, then each seed retrained with the epoch selected on validation video AP.
- **MIL heads**: no search; epoch selected on validation video AP (`../mil/README.md`).

Seeds 2025 / 234 / 3407 for every method; one `metrics.json` per seed; the table reports mean ± sd (n − 1).

## 4. Scoring and evaluation (`common.py`)

1 fps scores → 4 fps grid: each value repeated 4 times, then last-value padding or cutting to `ceil(4 · duration)`
(durations from `data/omsl_v6_inputs/manifests/all_test.jsonl`, `data/manifests/DeHate_test.jsonl`; run_plan §1.2).
Before evaluation every cohort video must have a finite score on every GT frame, and no other id may remain; the
evaluator's `n_videos_overlap` and frame pool must equal the cohort's (HateMM 215 / 116,975 frames, HCS 118 /
113,002, DeHate 1151 / 441,345). Evaluation only by `python -m src.eval.evaluate_four_datasets --datasets <corpus>`.
No fallback was needed: every cohort video has every feature.

## 5. Environment

conda `HateVideo` (torch 2.7.1 + cu128) on the RTX 5090 lab nodes, inside Slurm jobs. The upstream pinned
environments (e.g. DSANet torch 2.0.1 / cu117) do not support the 5090. No upstream code change was needed for
torch 2.7 beyond what the ports already document (`scripts/reproduction_baselines/PATCHES.md`); AVadCLIP's own code
runs unchanged (its `DistanceAdj` hard-codes `cuda`, so it needs a GPU even for a smoke test).

## 6. Files

| file | what |
|---|---|
| `common.py` | splits, labels, durations, 1 → 4 fps rule, coverage check, evaluator call, run.log |
| `prepare_inputs.py` | builds `data/weaksup_1fps/` and the split check (CPU) |
| `extract_audio.py` | wav2vec2-base and Wav2CLIP 1 s features (GPU) |
| `port.py` | runs a `scripts/reproduction_baselines` port (VadCLIP, DSANet, MultiHateLoc) unchanged, with its data paths, labels, splits and gold placeholders pointed at `data/weaksup_1fps/`; `--no-val` drops the validation split |
| `run_method.py` | fixed / tuned protocol, 3 seeds, test scoring, coverage check, evaluation, `summary.json` |
| `reuse_dehate.py` | checks and re-evaluates the existing DeHate MultiHateLoc run |
| `run_tasks.sh` | runs a list of tasks inside one Slurm job |

Outputs: `runs/20261008_baselines/<method>/<Dataset>/seed<k>/` (`run.log` first line = host, `config.json`,
`scores.jsonl` 1 fps, `predictions.jsonl`, `coverage.json`, `metrics.json`) and `.../<Dataset>/summary.json`.

## 7. Results (all rows done, 2026-10-08)

Seed mean ± sd (n − 1), seeds 2025 / 234 / 3407; pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro
ROC-AUC. Transcribed from `runs/20261008_baselines/weaksup_table.md` (`collect.py`), which reads each
`runs/20261008_baselines/<method>/<Dataset>/summary.json`, itself built from the per-seed `metrics.json`. Headline
branches are the unlabelled rows; "score_mlp" / "score_align" rows are the declared secondary branches. All rows use
video-level labels of the target corpus's train split; DeHate is external validation only.

| method | HateMM ROC | PR | within | HateClipSeg ROC | PR | within | DeHate ROC | PR | within |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BERT + MIL (text) | 0.6362 ± 0.0309 | 0.3835 ± 0.0440 | 0.5253 ± 0.0148 | 0.5442 ± 0.0085 | 0.5138 ± 0.0070 | 0.5044 ± 0.0108 | 0.6456 ± 0.0036 | 0.1730 ± 0.0035 | 0.5526 ± 0.0027 |
| wav2vec2 + MIL (audio) | 0.6919 ± 0.0090 | 0.4422 ± 0.0046 | 0.5570 ± 0.0057 | 0.4973 ± 0.0017 | 0.4718 ± 0.0036 | 0.4775 ± 0.0011 | 0.5418 ± 0.0002 | 0.0821 ± 0.0003 | 0.5293 ± 0.0094 |
| CLIP + MIL (visual) | 0.7191 ± 0.0036 | 0.4585 ± 0.0057 | 0.5229 ± 0.0027 | 0.5583 ± 0.0035 | 0.5182 ± 0.0035 | 0.5249 ± 0.0035 | 0.6418 ± 0.0032 | 0.1312 ± 0.0012 | 0.5202 ± 0.0047 |
| VadCLIP | 0.6111 ± 0.0434 | 0.3588 ± 0.0435 | 0.4783 ± 0.0365 | 0.5308 ± 0.0246 | 0.4825 ± 0.0256 | 0.5124 ± 0.0181 | 0.6035 ± 0.0046 | 0.1091 ± 0.0049 | 0.5120 ± 0.0109 |
| VadCLIP, score_mlp | 0.6822 ± 0.0226 | 0.3998 ± 0.0562 | 0.4640 ± 0.0356 | 0.5001 ± 0.0114 | 0.4580 ± 0.0163 | 0.5126 ± 0.0209 | 0.6276 ± 0.0150 | 0.1167 ± 0.0122 | 0.4619 ± 0.0047 |
| DSANet | 0.7005 ± 0.0142 | 0.4136 ± 0.0546 | 0.5290 ± 0.0427 | 0.5080 ± 0.0178 | 0.4622 ± 0.0119 | 0.5024 ± 0.0188 | 0.6404 ± 0.0151 | 0.1363 ± 0.0268 | 0.4776 ± 0.0126 |
| DSANet, score_align | 0.6862 ± 0.0105 | 0.4479 ± 0.0182 | 0.5600 ± 0.0055 | 0.5349 ± 0.0269 | 0.5053 ± 0.0098 | 0.5074 ± 0.0391 | 0.6655 ± 0.0117 | 0.1398 ± 0.0098 | 0.5229 ± 0.0096 |
| AVadCLIP (audio-visual) | 0.6369 ± 0.0225 | 0.3503 ± 0.0398 | 0.5077 ± 0.0358 | 0.4996 ± 0.0096 | 0.4596 ± 0.0091 | 0.5291 ± 0.0184 | 0.6104 ± 0.0168 | 0.1112 ± 0.0017 | 0.5343 ± 0.0314 |
| AVadCLIP, score_mlp | 0.6772 ± 0.0214 | 0.4345 ± 0.0359 | 0.4764 ± 0.0268 | 0.5574 ± 0.0229 | 0.5163 ± 0.0274 | 0.5326 ± 0.0065 | 0.6419 ± 0.0345 | 0.1385 ± 0.0231 | 0.5057 ± 0.0172 |
| MultiHateLoc (reimpl.) | 0.7441 ± 0.0134 | 0.4946 ± 0.0138 | 0.6228 ± 0.0128 | 0.5185 ± 0.0152 | 0.4787 ± 0.0100 | 0.4983 ± 0.0063 | 0.6102 ± 0.0090 | 0.1289 ± 0.0040 | 0.5420 ± 0.0161 |

Hosts: MIL and AVadCLIP uoa-lab1 (Slurm 291), MultiHateLoc HateMM/HCS uoa-lab1 (291 + 302), DSANet and VadCLIP
uoa-lab3 (Slurm 290; checkpoints left on uoa-lab3), MultiHateLoc DeHate re-evaluated from the 2026-09-26 run.
