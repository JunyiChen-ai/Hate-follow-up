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

## Is it worth trying: combining several MLLM readers (2026-09-28)

The headroom above needs information that one reader lacks. The only such source already on disk is the other seven
MLLMs' reads from the family study. They use the older ASR loader, so every comparison below stays inside those reads.

`ensemble_reads.py` builds combined reads without labels:
- each model's reads become normal scores within the corpus, per modality, and so does each model's verdict;
- they are averaged over the models, window by window.

The time level is `r6_bma`. Tables: `runs/20260928_headroom/analysis_ensemble/table.txt` and
`analysis_verdict8/table.txt`. The reference is Qwen3-VL-8B alone through the same path (`r6_single8`). Differences
carry a paired bootstrap 95 % interval.

| reads | HateMM ROC / PR / within | HateClipSeg ROC / PR / within | reading cost against the 8B alone (rough) |
|---|---|---|---|
| 8B alone | .8899 / .6776 / .7636 | .7191 / .6587 / .6372 | 1× |
| all 8 models | +.006 / +.003 / **+.029 [+.001, +.061]** | +.007 / **+.023 [−.002, +.051]** / −.007 | about 10× |
| 4 Qwen3-VL sizes (2B, 4B, 8B, 32B) | +.004 / +.004 / +.030 [−.001, +.062] | −.003 / +.011 / −.001 | about 6× |
| 3 small Qwen3-VL (2B, 4B, 8B) | +.003 / +.001 / +.030 | **−.018 / −.000 / −.017** | about 2× |
| verdict of all 8, windows from the 8B only | +.002 / −.008 / +.000 | **−.100 / −.070** / −.001 | about 2–3× |

Video AUC: all 8 models .912 → .933 (HateMM) and .818 → .870 (HateClipSeg).

Reading:
- Several readers carry information that one reader lacks.
  - Within: +.03 on HateMM, borderline. HateClipSeg does not move.
  - Video level: HateClipSeg PR +.02.
- The cheap variants fail on HateClipSeg. Averaging only the verdict breaks the calibrated key: 23 % of positive
  videos fall below .5.
- The combinations that help cost about 6–10× the current reading.
- Rule 14(e) of `RESEARCH_ITERATION_RULES.md` excludes an ensemble from a claimed method. Adopting a reader average
  therefore needs a rule change by the user. This was missed when this section was written. It was noted on
  2026-09-28, after the Codex consultation below.

## Codex consultation and a check of its first proposal (2026-09-28)

The user asked whether the performance is close to what this paradigm can reach, and asked for Codex's view on making
the mechanism more interesting. The full record is in `docs/reviews/20260928_codex_mechanism_review.md`.

**Codex's view:**
- The current scalar reads are close to their practical limit. It is not shown that the paradigm is at its ceiling.
- It made two proposals:
  1. **Coupling by speech continuity (M3).** Time coupling is driven by speech continuity instead of clock time:
     strong across a boundary where the speech continues, weak across a break. No extra MLLM calls.
  2. **A small dense student (M1–M3).** The student is trained from the current method's outputs, with temporal
     consistency. It needs no MLLM calls at deployment and 2–8 GPU hours to train.
- It asked for a check that separates the 8 s reading grid from the accuracy of the reads.

**Check.** Script `continuity_check.py`, CPU; output `runs/20260928_headroom/continuity/table.txt`.
- Label: a window is hateful if its gold hate share is ≥ .5.
- A change is a boundary between neighbouring 8 s windows where that label differs. Only videos with both labels count.
- Continuity measures:
  - whether a Whisper segment runs across the boundary;
  - the cosine of the two window transcripts under all-MiniLM-L6-v2.
- Change prediction uses logistic regression with 5-fold CV by video. The interval is a bootstrap over videos.

| | HateMM | HateClipSeg |
|---|---|---|
| boundaries (videos) | 1452 (75) | 2814 (96) |
| change rate: Whisper segment runs across / speech on both sides without one / speech missing on a side | .115 / .128 / .246 | .190 / .224 / .196 |
| change rate by transcript cosine quartile, lowest to highest | .141 / .111 / .117 / .096 | .198 / .219 / .207 / .163 |
| predicting a change, AUC: reads / reads + continuity / continuity only | .602 / .604 / .562 | .545 / .554 / .515 |
| added by continuity on top of the reads | +.002 [−.008, +.010] | +.008 [−.005, +.021] |
| within: method / true share per 8 s window / true share per 4 s cell | .751 / .954 / .983 | .637 / .986 / .996 |
| same, videos where hate covers < 25 % (19 each) | .694 / .957 / .982 | .680 / .992 / .997 |

**Reading:**
1. **Proposal 1 lacks the signal it needs.**
   - Whether the speech continues across a boundary barely changes whether the label changes.
   - Even a model fitted on the gold gains nothing significant from continuity on top of the reads.
   - A coupling rule built on it cannot be expected to pass the gate or to show a ≥ .01 ablation.
   - Caveat: the Whisper segments here are coarse (up to 100 s, no word timestamps). Finer pauses and speaker turns
     were not tested.
2. **The 8 s reading grid is not the limit.**
   - If every window carried its true hate share, within would be .954 / .986, and the same for short hate.
   - The gap to the method (.20 / .35) comes from the accuracy of the reads themselves.
3. **Proposal 2 was not tested.** Three things weigh against it:
   - Label-free methods built on features in this repository scored far below the MLLM reads (T3AL, HateMM .6091 /
     .3096 / .5068).
   - Self-training (SDL, NGA) failed.
   - The method would no longer be training-free.

## Codex consultation, round 2 (2026-09-28)

The user asked to send the checks above back to Codex for another round. The record is in
`docs/reviews/20260928_codex_mechanism_review_round2.md`.

**Codex's view:**
- It accepts that coupling by speech continuity is dead. It withdraws the dense student, because nothing in it
  corrects the teacher's confusion between mentioning a group and attacking it.
- It made one proposal only, a small pilot of **layer-dependent access to context in M1**:
  - the early layers of each window branch see the full prefix, as now;
  - the last layers see only the policy and the window's own question, transcript and frames.
  - It needs no extra calls.
  - Its premise: late-layer access to the verdict and to other windows' evidence swamps the window's own evidence.
- Bottom line: nothing has a high probability of meeting all constraints. The one constraint it would change is to
  allow a small amount of segment-level supervision (mention against attack) to adapt the reader. That gives up
  label-free.

**Check of the premise on cached reads.** Script `context_mix_check.py`, CPU; output
`runs/20260928_headroom/context_mix/table.txt`.
- Reads: the Qwen3-VL-8B reads of the family study (older ASR loader).
- Metric: window-level within, against the gold label (hate share ≥ .5).
- Each read made with less context is tested two ways:
  - alone;
  - mixed into the full-context read as normal scores: full + λ × reduced, with λ = .25, .5, 1.
- Intervals: bootstrap over videos.

Full-context window-level within: HateMM .7628, HateClipSeg .6247.

| less-context read | HateMM: alone vs full | HateMM: best mixture vs full | HateClipSeg: alone vs full | HateClipSeg: best mixture vs full |
|---|---|---|---|---|
| no verdict turn (`nostance`) | +.001 | +.002 (λ 1) | −.020 | −.005 (λ .25) |
| no transcript context (`noctx`) | −.072 | +.003 (λ .25) | −.036 | −.002 (λ .25) |
| no frames (`noframes`) | −.063 | +.003 (λ .25) | −.033 | −.001 (λ .25) |
| no context at all (`winonly`) | −.079 | −.001 (λ .25) | −.037 | +.002 (λ .25) |

**Reading:**
- Reads made with less context carry no within-video information that the full-context read lacks.
  - No mixture gains .01; larger weights lose.
  - Removing the verdict turn does not help the window ranking: HateClipSeg −.020.
- So the premise of the layer-split pilot is not supported.
- Caveat: restricting only the last layers is a different computation from removing context at the input. Only a GPU
  pilot tests it directly.

## Test-read log


2026-09-28: the gold of all three corpora (hate share per video and per 4 s cell), used only for the bounds above. No
method changed.
2026-09-28 (worth trying): `runs/20260928_headroom/analysis_ensemble`, `analysis_verdict8` (gold for the metrics only). No
method changed.
2026-09-28 (Codex check): gold of HateMM and HateClipSeg test (hate share per 8 s window and 4 s cell), used only for
`runs/20260928_headroom/continuity`. No method changed.
2026-09-28 (Codex round 2): gold of HateMM and HateClipSeg test (hate share per 8 s window), used only for
`runs/20260928_headroom/context_mix`. No method changed.
