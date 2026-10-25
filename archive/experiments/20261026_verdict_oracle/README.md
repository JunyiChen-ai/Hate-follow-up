**Archived 2026-10-26 with candidate 51 (item 41), which this analysis motivated; see `archive/experiments/20261026_knowledge_verdict/README.md` §7.**

# Verdict oracle: how much a more accurate video verdict would help frame-level metrics

2026-10-26, uoa-lab1 (sc474397), CPU only. Asked by the user: "video-level 判断得更准了，对我们当前的 frame-level ranking
是否有帮助？如果把判断错的 video 反转过来，再去 carry 那些窗口，结果会不会好很多？"

This is an oracle diagnostic. It reads test GT (rule 10) and is not a method. Every number here is development
analysis.

## What was read
- r6_bma predictions: `runs/20260926_twolevel/final_rawkey/predictions.jsonl` (HateMM, HateClipSeg) and
  `runs/20260927_dehate_external/r6_rawkey/predictions.jsonl` (DeHate).
- The verdict z_video of each video: `runs/20260926_glr/base_gridA/predictions.jsonl`,
  `runs/20260927_dehate_external/reads_gridA/predictions.jsonl`.
- Test GT `data/gt_4fps/{HateMM,HateClipSeg,DeHate}.npz`. A video counts as hateful if it has any positive 4 fps frame.

## Method (`flip_cpu.py`)
The r6 frame score is K + centred rank, with K = z_video + mean window z (checked on 10 videos: the mean of
score_curve equals z_video + mean window z). A video is "wrong" when the sign of z_video disagrees with its GT.

| arm | change |
|---|---|
| base | none; reproduces the reference metrics exactly |
| flip_all | each wrong video's K uses −z_video instead of z_video (same confidence, right sign) |
| flip_fp / flip_fn | only the false-positive / only the false-negative videos |
| flip_p25 / flip_p50 | a random 25 % / 50 % of the wrong videos (5 seeds, mean) |
| perfect_split | every GT-negative video shifted below every GT-positive video (headroom A1) |

Window reads, M2 and the within-video rank are not touched. Metrics come from `src/eval/evaluate_four_datasets.py`.

## Results
Wrong videos: HateMM 67 of 215 (64 false positives, 3 false negatives); HateClipSeg 15 of 118 (10 / 5); DeHate 556 of
1151 (518 / 38). Output: `runs/20261026_verdict_oracle/cpu_flip_{main,dehate}/{metrics.json,summary.json}`.

Pooled ROC / pooled PR / within:

| arm | HateMM | HateClipSeg | DeHate |
|---|---|---|---|
| base (r6) | .8955 / .6883 / .7508 | .7137 / .6666 / .6373 | .6994 / .1568 / .6431 |
| fix 25 % of wrong videos | .9019 / .6967 / .7508 | .7210 / .6702 / .6373 | .7390 / .1781 / .6431 |
| fix 50 % | .9083 / .7042 / .7508 | .7378 / .6778 / .6373 | .7913 / .2090 / .6431 |
| fix all (flip_all) | .9245 / .7288 / .7508 | .7462 / .6812 / .6373 | .9010 / .3309 / .6431 |
| fix false positives only | .9214 / .7274 | .7462 / .6813 | .8306 / .3066 |
| fix false negatives only | .8970 / .6887 | .7129 / .6663 | .7257 / .1602 |
| perfect_split | .9246 / .7289 / .7508 | .7478 / .6818 / .6373 | .9119 / .3404 / .6431 |

Seed sd of the 25 % / 50 % arms: at most .0045 ROC and .0102 PR.

## Reading
1. A more accurate verdict raises only the pooled metrics. Within does not move, because K is constant within a video.
2. Almost all of the pooled gain comes from fixing false positives. Most wrong videos are false positives, and those
   videos have no positive frames, so they never enter the within average.
3. Re-reading the windows under the corrected verdict ("carry the windows") can change within only through the
   false-negative videos that have both positive and negative frames: 3 of 84 (HateMM), 5 of 99 (HateClipSeg),
   37 of 222 (DeHate). Even if all of them reached within 1.0, the macro within would gain at most +.020 / +.017 /
   +.065. If they reached the corpus mean, the gain would be +.012 / +.005 / +.023. The pooled room left beyond
   flip_all is the gap to perfect_split: +.0001 / +.0016 / +.011 ROC.
4. The forced-verdict window re-read itself was not run (no reader supports forcing the verdict answer yet).
