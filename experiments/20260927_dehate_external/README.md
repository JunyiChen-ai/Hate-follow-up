# DeHate external validation (2026-09-27)

Hosts are written in the first line of each `runs/20260927_dehate_external/<run>/run.log`.

## 1. Why

The user asked on 2026-09-27 to run our method and a few strong baselines on DeHate. That request is the consent
CLAUDE.md requires before a new corpus is used. DeHate is **external validation only**:
- it never gates a method;
- it never enters the main table;
- no design choice is changed because of it.

It is the first corpus our method has not been developed on. HateMM and HateClipSeg numbers are development-selected
(rule 10); the numbers here are not, as long as nothing below changes after the first DeHate result is seen.

DeHate terms of use (Microsoft application form): non-commercial research, no redistribution of the dataset or of
derived data. Everything derived from it stays in `data/` and `runs/`, which git ignores.

## 2. Declared before any DeHate number of ours was computed

**Methods, frozen at commit `864819c` (code) as they are on HateMM and HateClipSeg:**

| name | what | command |
|---|---|---|
| `spvl_r2` | SPVL-r2 composition, no duration prior | `til_infer.py --model none --dwell 0` |
| `current` | SPVL-r2 + duration prior (the reported method) | `til_infer.py --model average --fusion max --dwell 80` |
| `r3_m2` | candidate redesign (`experiments/20260926_twolevel/` §14) | `twolevel_r2.py --noleak --transform nscore --key calib --k 4 --arm m2` |

All three are computed from one set of reads: `experiments/20260922_til/til_measure.py` (SPVL-r2 grid A:
verdict, stance, isolated visual and speech branches per fixed 8 s window), Qwen/Qwen3-VL-8B-Instruct, 20 uniform
frames, Whisper large-v3 transcripts. These are the same settings as `runs/20260926_glr/base_gridA`.

**Baselines.**
- Zero-label: ZS-ImageBind (`scripts/reproduction_baselines/zs_imagebind.py`, the baseline LAVAD defines; unchanged).
- T3AL: depends on a user decision (its pipeline lives in the Retrieval-hate repository and would have to be ported).
- Weakly supervised references, all trained with DeHate **train-split video labels**. They come from the Retrieval-hate
  DeHate run (uoa-lab2, `~/Retrieval-hate/runs/20260926_dehate_external/baselines/final/<method>/dehate/seed_<s>/`):
  - the methods are Fed-WSVAD with 3 clients, MultiHateLoc, DSANet and MACIL-SD;
  - each was tuned on DeHate validation and trained at seeds 234 / 2025 / 3407;
  - their 1 fps test scores are repeated 4 times per second onto the 4 fps grid;
  - each is padded with its last value to `ceil(duration * 4)` frames;
  - each is scored per seed by the shared evaluator; the table reports the seed mean.

**Cohort and gold (`scripts/dehate/prepare_dehate_4fps.py`).**
- Official test split: 1341 videos.
- Every method runs on all 1341 videos. The method reads no labels, so no video is dropped by label before inference.
- Gold: the 4 fps rasterization of `data/gt_4fps` (frame i at t = i / 4 is positive when start <= t < end;
  `floor(duration * 4)` frames), with the duration taken from the video container.
- Spans are parsed as in the Retrieval-hate DeHate amendment (`docs/duplex/FRAME_EVAL_PROTOCOL_DEHATE.md` there):
  - every `(a, b)` pair counts, plus the one row written `[a, b]`;
  - spans with `end <= start` are dropped.
- A hateful video with no usable span (text-only hate in the title or description) is excluded from the gold
  (rule (b)). Expected: 1151 scored videos (234 hateful).

**Metrics.** The shared evaluator (`src/eval/evaluate_four_datasets.py`, unchanged) reports:
- pooled ROC, pooled PR and within-video macro ROC;
- interval F1 for `r3_m2`, the only method here that outputs intervals.

Paired bootstrap over videos (4000 resamples, seed 0) for:
- `r3_m2` − `current` (the no-drop question on an unseen corpus);
- `current` and `r3_m2` against the strongest baseline on each metric.

**What is concluded.**
- (1) Whether our method is above every baseline on DeHate, on each of the three metrics.
- (2) Whether `r3_m2` is not below `current` beyond the noise floor (pooled .005, within .01).

No threshold, constant or design is chosen on DeHate.

## 3. Inputs (derived caches, provenance in each directory)

| cache | source |
|---|---|
| `~/data/DeHate/test/*.mp4` on uoa-lab1 | rsync of uoa-lab2 `~/data/DeHate/test/` (1341 files) |
| `data/manifests/DeHate_test.jsonl` | `prepare_dehate_4fps.py manifest` (ffprobe container duration) |
| `data/asr_whisper_large_v3/DeHate/timestamped_chunks.jsonl` | test rows of Retrieval-hate `results/reproduction/asr/dehate_all/` (Whisper large-v3, same format and producer family as the HateClipSeg file) |
| `data/frames_k20/DeHate/` | `experiments/20260910_spvl/prep_frames.py`, unchanged |
| `data/gt_4fps/DeHate.npz` | `prepare_dehate_4fps.py gt` |

## 4. How to run

```
# uoa-lab1, HateVideo env (CPU)
python scripts/dehate/prepare_dehate_4fps.py manifest
python scripts/dehate/prepare_dehate_4fps.py asr
python scripts/dehate/prepare_dehate_4fps.py gt
python experiments/20260910_spvl/prep_frames.py --manifest data/manifests/DeHate_test.jsonl --datasets DeHate
# reads (lab machine with the HateVLM env)
setsid nohup bash experiments/20260927_dehate_external/launch/run_reads.sh > runs/20260927_dehate_external/launch_reads.out 2>&1 &
# ZS-ImageBind (lab machine with the DeHate videos and the ImageBind weights)
setsid nohup bash experiments/20260927_dehate_external/launch/run_zsib.sh > runs/20260927_dehate_external/launch_zsib.out 2>&1 &
# uoa-lab1, after the reads are back (CPU)
bash experiments/20260927_dehate_external/launch/run_compose.sh
python experiments/20260927_dehate_external/convert_weaksup.py
python experiments/20260927_dehate_external/summarize.py
```

## 5. Test-read log (rule 10)

- 2026-09-27, before any run: the DeHate label file (split and label counts, the span string format) and the
  Retrieval-hate DeHate README (its protocol and 1 fps results). No per-video prediction or error was read.
  Nothing in the method was changed.

## 6. Runs

(filled in as they run)

## 7. Results

(filled in after the runs)
