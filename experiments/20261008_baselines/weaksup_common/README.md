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
