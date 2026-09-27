# Headroom of the open concerns under the current protocol (2026-09-28)

Analysis only; it reads the gold. It is not a method and is never part of one.

## Question

The user asked (2026-09-28) whether K3, K4, K5 and K7 can be solved with the protocol, the gate and the corpora
unchanged. Each concern is answered with two numbers:
- **Metric headroom.** How far the metric could move if that level were perfect.
- **Reachable from the current reads.** How much a model fitted on the gold reaches from the same reads, using
  5-fold cross-validation by video.

If even a model fitted on the gold does not beat the label-free method, the reads are the limit, not the way the
method combines them.

## How

`headroom.py`, output `runs/20260928_headroom/table.txt` and `summary.json`. Inputs:
- the current method `runs/20260926_twolevel/final_m2` (r6_bma) and its reads `runs/20260926_glr/base_gridA`;
- DeHate `final_dehate/final_m2` and `reads_gridA`;
- the DVD reads (target, endorsement, attack) `runs/20260927_dvd/reads_*`;
- the older reads with and without one frame per window: `runs/20260910_spvl/full2_dual_evid_stance`,
  `full3_dual_evid_stance_w8`.

**A. Video level (K3).** The method's within term is kept.
- **A1.** Every non-hateful video goes below every hateful one; hateful videos keep their scores.
- **A2.** A1, and hateful videos are also ordered by their true hate share.
- **A3.** A video score fitted on the hate share from the current reads, the method's outputs and the DVD reads.
  - Features: verdict; per modality the mean, median, max, quartiles, spread and positive share of the window reads;
    the key; the mean cell posterior; the DVD reads.
  - Models: ridge, or small boosted trees. The best model and scale are kept, so the bound is generous.

**B. Within level (K4, K5).** A model fitted on each 4 s cell's hate share, over videos with both classes. It sees
only within-video features:
- each modality's read minus the video mean;
- its rank in the video;
- the neighbouring windows;
- the position in the video;
- optionally, the method's own cell log-odds and their neighbours.

The best of three settings is kept: boosted trees, small boosted trees, ridge.

**C. Visual branch (K7).** B with visual reads only or speech reads only, and on the older reads with and without a
frame per window.

## Results (pooled ROC / pooled PR, or within)

**A. Video level:**

| | HateMM | HateClipSeg | DeHate |
|---|---|---|---|
| current method | .8971 / .6942 | .7168 / .6711 | .7011 / .1582 |
| A1 perfect hateful / non-hateful split | .9261 / .7353 | .7510 / .6865 | .9126 / .3434 |
| A2 plus hateful videos ordered by true hate share | .9791 / .9343 | .8923 / .8790 | .9800 / .8388 |
| A3 fitted on the gold from the current reads | .8691 / .6331 (video AUC .883, method .927) | .7120 / .6566 (.766, method .803) | .6963 / .2168 (.745, method .727) |

**B. Within level.** Fitted minus the label-free method, with a paired bootstrap over videos:

| fitted on | HateMM (method .7508) | HateClipSeg (method .6373) |
|---|---|---|
| reads, both modalities | .7337, −.017 [−.058, +.028] | .5967, −.041 [−.072, −.007] |
| reads plus the method's outputs | .7659, +.015 [−.015, +.049] | .6214, −.016 [−.034, +.003] |
| subset where hate covers < 25 % (reads plus method) | +.008 [−.043, +.050] | −.026 [−.061, +.005] |

**C. Visual branch.** Fitted within:

| | HateMM | HateClipSeg |
|---|---|---|
| visual reads only / speech reads only | .7100 / .7571 | .5410 / .5791 |
| older reads, visual only: 20 shared frames / plus a frame per window | .7179 / .6841 | .5563 / .5353 |
| older reads, both modalities: 20 shared frames / plus a frame per window | .7439 / .7278 | .5870 / .5964 |

## Conclusions

1. **K3 has metric headroom, but the current reads do not reach it.**
   - A perfect split of hateful and non-hateful videos would raise HateMM by +.029 / +.041 and HateClipSeg by +.034 /
     +.015. That is above the gate. The larger headroom is ordering hateful videos by how much of them is hate.
   - This corrects a statement made on 2026-09-27: it is not true that no video-level change could pass the gate.
   - On HateMM and HateClipSeg, a video score fitted on the gold from the same reads, including the DVD target,
     endorsement and attack reads, is below the label-free key. It separates videos worse (video AUC .883 against
     .927, .766 against .803).
   - The information needed to separate the remaining non-hateful videos is not in these reads. On DeHate it is,
     partly: the fitted PR is .217 against .158.
2. **K4 and K5: within is at the limit of the reads.**
   - With the gold, the same reads plus the method's own outputs give at most +.015 on HateMM, with an interval that
     includes 0, and nothing on HateClipSeg. The short-hate subset does not gain either.
   - A better label-free combination of these reads cannot be expected to pass the .01 gate on both corpora.
3. **K7: the visual reads carry less within-video information than speech reads.** Visual only is .045 / .038 below
   speech only. One frame per window does not add fitted information to the visual reads: HateMM .718 → .684,
   HateClipSeg .556 → .535.
4. **Caveat.** The fitted models are trained on 84–215 videos per corpus, so they give a lower bound on what the gold
   could teach, not an exact ceiling. They consistently fail to beat the label-free method on the same inputs. So the
   data show no reachable headroom from the current reads.
5. **Consequence.** Under the unchanged protocol, these concerns need new information: new reads or new inputs, not a
   new way to combine the existing ones. The 8-MLLM study already shows that a larger reader of the same family does
   not change this. Under r6, Qwen3-VL-32B is +.009 / −.003 within against the 8B reads.

## Test-read log

2026-09-28: the gold of all three corpora (hate share per video and per 4 s cell), used only for the bounds above. No
method changed.
