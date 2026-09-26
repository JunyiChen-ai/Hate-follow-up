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
  - Branch: the one the Retrieval-hate run reports as its headline number (its README §5.1). These are Fed-WSVAD
    `score_align`, MultiHateLoc `score_fused`, DSANet `score_mlp` and MACIL-SD `score_av`
    (`convert_weaksup.py`; the choice was added before any number of ours existed).

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

**Correction (2026-09-27 09:24, after the `r3_m2` frame numbers were seen).** The `m2` arm writes no intervals. On
HateMM and HateClipSeg the interval F1 of round 3 came from the `r3_full` arm:
- same reads and time level;
- composition `log P(V=1|K) + log P(hate at t | V=1)`, intervals where the product is at least .5;
- `runs/20260926_twolevel/analysis_r3/table.txt`.

`r3_full` is therefore added, unchanged, as the interval-output arm. It was a fixed arm of round 3, not a choice made
on DeHate.

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
- 2026-09-27, after the results of §7: a video-level diagnostic. It read the gold video label, our verdict and key
  from `runs/20260927_dehate_external/{r3_m2,weaksup}/predictions.jsonl`, and the platform column. Findings are in
  §7. Nothing in the method was changed.

## 6. Runs

**Inputs (uoa-lab1, 2026-09-27).**
- Videos: copied from uoa-lab2, 1341 files, same total size.
- Manifest: 1341 rows, all with a duration and a video stream.
- Transcripts: 1341 rows, 0 error rows.
- Gold: 1151 videos, 234 hateful, base rate .0760, 222 videos with both classes, 190 excluded by rule (b). This is the
  same cohort as the Retrieval-hate 1 fps run.
- Frames: 1339 complete, 2 with 19 of 20 frames (`runs/20260927_dehate_external/prep_frames.log`).

**Reads (`reads_gridA`).**
- Host: lab-server (sc448960, the host of `base_gridA`), HateVLM env, commit f7d1c5b, from 08:54.
- Smoke test on 2 videos: 0 errors.
- Plumbing check: cache vs plain forward differs by .18 logits on a verdict of 3.8, bf16 rounding. On `base_gridA` it
  was .03 on 17.6. The script has no stop threshold for it.
- The smoke output was deleted before the full run.

**ZS-ImageBind (`zs_imagebind`).**
- Host: uoa-lab2 (sc474399, where the videos live), HateVideo env, commit f7d1c5b, from 08:49.
- Weights and the ImageBind code were copied from uoa-lab1.
- A 2-video smoke test passed and was deleted.
- Speed is about 8 s per video.

**Weakly supervised references (`weaksup`).**
- Test scores copied from uoa-lab2 to `runs/20260927_dehate_external/weaksup_src/`: 12 files, 1151 videos each.
- Converted and evaluated by `convert_weaksup.py`.
- Padding: the median video needs no padding (−1 frame). One video needs 237 frames (59 s): its 1 fps curve ends at
  the audio length.
- The 4 fps numbers reproduce the 1 fps originals within about .002 (e.g. Fed-WSVAD seed 234: .6922 / .1925 at 4 fps
  vs .6925 / .1918 at 1 fps).

## 7. Results

Source: `runs/20260927_dehate_external/summary/table.txt` and `summary.json` (from `summarize.py`), which read each
method's evaluator output.
- Grid and cohort: test split, 4 fps, 1151 scored videos (234 hateful), frame base rate .076.
- Weakly supervised rows are seed means, with the sd in brackets.
- ZS-ImageBind was still running when this was written.

| method | pooled ROC | pooled PR | within |
|---|---|---|---|
| SPVL-r2 | .6993 | .1570 | .6406 |
| SPVL-r2 + duration prior (current) | .6996 | .1570 | .6364 |
| r3_m2 (candidate) | .7009 | .1578 | .6539 |
| r3_full (candidate, interval output; F1@.3 / .5 / .7 = .174 / .127 / .104) | .7028 | .1626 | .6539 |
| Fed-WSVAD, 3 clients (video labels) | .7007 (.011) | .1752 (.017) | .5055 (.009) |
| MultiHateLoc (video labels) | .6102 (.007) | .1289 (.003) | .5420 (.013) |
| DSANet (video labels) | .6325 (.010) | .1207 (.009) | .4873 (.014) |
| MACIL-SD (video labels) | .5620 (.010) | .0881 (.002) | .5272 (.011) |

Paired bootstrap over videos (4000, seed 0), 95 % intervals:

| comparison | ROC | PR | within |
|---|---|---|---|
| r3_m2 − current | +.0013 [+.0002, +.0024] | +.0008 [−.0008, +.0022] | +.0175 [−.0187, +.0539] |
| current − strongest baseline | −.0012 [−.061, +.058] (Fed-WSVAD) | −.0182 [−.079, +.037] (Fed-WSVAD) | +.0945 [+.042, +.147] (MultiHateLoc) |
| r3_m2 − strongest baseline | +.0001 [−.060, +.059] | −.0174 [−.078, +.038] | +.1119 [+.062, +.159] |

**Answers to the two declared questions.**
- (1) Within-video ROC: every variant of ours is above every baseline. The margin is +.09 to +.11 over the best one,
  MultiHateLoc, and the interval excludes 0.
- (1) Pooled ROC: we are level with Fed-WSVAD, the strongest baseline, which is trained on 4680 DeHate training
  videos with labels.
- (1) Pooled PR: we are below Fed-WSVAD by .017–.018. The interval includes 0. We are above the other three
  weakly supervised baselines.
- (2) `r3_m2` is not below `current` on any metric. Within is +.0175, above the .01 floor, but its interval includes
  0. The no-drop result from HateMM and HateClipSeg holds on a corpus it was not developed on.

**Why pooled PR and interval F1 are lower than on HateMM (diagnostic, test labels read, method unchanged).**
- Video ranking: our calibrated key ranks videos better than Fed-WSVAD's max-pooled score by AUC (.727 vs .699).
  It is worse by AP (.361 vs .381).
- The MLLM verdict is positive for 84 % of hateful videos, but also for 57 % of non-hateful ones (BitChute 63 %,
  TikTok 42 %). Over all 1341 test videos it is positive for 60 %; the EM video prior is .62.
- Confident false-positive videos at the top of the ranking lower pooled PR. They also produce intervals: at IoU .3,
  precision is .115 and recall .364.
- Likely cause, not yet checked video by video: DeHate's non-hateful side holds offensive, conspiratorial or political
  videos (mostly BitChute) that the policy prompt judges violating. That would be a label-definition gap like
  HateClipSeg's. It is not a localization failure: the within-video ordering is the strongest result here.
