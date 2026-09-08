# Entropy-adaptation kill test: result note

**Date:** 2026-08-09. **Verdict: DEAD.**
**Preregistration:** `docs/duplex/PREREG_entropy_tta_killtest.md`, frozen before
any adaptation step ran.
**Machine-readable result:** `docs/duplex/reports/entropy_tta_killtest.json`
(also written to the gitignored `results/entropy_tta/results.json`).
**Compute:** one RTX 5090, 12 minutes 47 seconds of GPU time for the whole
program, against a preregistered budget of 8 hours.

## Headline

Entropy minimization on the judge's own {Yes, No} answer distribution does not
move the label-free operating point on HateMM. After one epoch of TENT-style
adaptation over all 215 unlabeled test videos, the valley macro-F1 is
**0.6460**, slightly below the frozen baseline's 0.6562 and far below the
preregistered bar of 0.75. Ranking is untouched at AUC 0.9236. The placebo arm,
trained on deliberately incoherent frame-transcript pairs, lands at 0.6511 —
closer to the baseline than the real arm is.

The reason is visible in the objective itself. The judge's answer distribution
is already saturated on 82% of the corpus: the median per-video answer entropy
before any training is 4.0e-05 nats, and only 39 of 215 videos exceed 0.01
nats. An objective that pushes a posterior away from 0.5 has almost nothing to
push. This is the first training-based family opened in this project, and it
dies on its most favorable case.

## Frozen clauses

| # | Clause | Bar | Observed | Result |
|---|---|---:|---:|---|
| 1 | Operating point recovers | valley macro-F1 >= 0.75 | 0.6460 | **FAIL** |
| 2 | Ranking preserved | AUC >= 0.90 | 0.9236 | PASS |
| 3 | Evidence-driven, not update-driven | placebo gain < 0.5 x real gain | not satisfiable | **FAIL** |

Clause 3 deserves a plain statement rather than a ratio. The real arm's gain
over baseline is **-0.0102** and the placebo arm's is **-0.0051**. The
preregistered comparison divides one gain by the other, which is meaningless
when the numerator is negative. Recorded as not satisfied: a clause designed to
show that a gain came from evidence cannot be met by an arm that produced no
gain. Read descriptively, the placebo arm did marginally *better* than the real
arm, which is the opposite of the mechanism's prediction.

Two of three clauses fail, so the family is dead under the frozen rule.

## What ran

The frozen judge is `src/duplex/extract_duplex_readout.py`: 16 frames from
`frames_16`, max_pixels 100352, the frozen `prag` reader block, the gated fresh
Whisper transcript fed uncapped, and the raw readout
`z = logsumexp(Yes ids) - logsumexp(No ids)` at the final prompt position. The
adaptation script imports the prompt construction, the frame resolution, the
transcript resolution and the readout from that module rather than restating
them.

Trainable parameters were the normalization gains of both towers: 203 tensors
and 385,408 scalars, of which 58 tensors sit in the vision encoder and 145 in
the language model. Everything else stayed frozen. The objective was the
Shannon entropy of the renormalized two-way {Yes, No} distribution and nothing
else. AdamW, learning rate 1e-5, weight decay 0, batch size 1, exactly one
epoch over the 215 videos in a fixed shuffled order (seed 20260808), gradient
checkpointing on, no early stopping. All 203 tensors received gradients on
every step. Peak memory was 19.20 GiB of the card's 32 GB, so no memory lever
from the protocol was needed.

The placebo arm is the same run with each video's frames paired against a
different video's transcript under a fixed derangement (seed 20260808), then
rescored with coherent inputs. Its update count, data statistics and optimizer
trajectory length match the real arm exactly.

Four rescoring passes ran, each a single forward call per video over all 215
videos, with every frame decoded before the pass began.

### Baseline self-check

The baseline was produced by the same HF-transformers code path, so there is no
vLLM-versus-transformers gap to reconcile. Rescoring the 215 videos with the
frozen model through the new adaptation script reproduced the committed test_c2
numbers exactly: AUC 0.923247 against 0.923247, valley macro-F1 0.6561809
against 0.6561809, and a maximum per-video absolute z difference of **0.0**
across all 215 videos. The instrument is identical, so every difference below
is attributable to adaptation alone.

## Arm comparison

| Arm | AUC | Valley threshold | Valley macro-F1 | FP | FN | Trough depth |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (frozen) | 0.9232 | -2.3455 | 0.6562 | 70 | 3 | 0.2831 |
| Self-check (frozen, this script) | 0.9232 | -2.3455 | 0.6562 | 70 | 3 | 0.2831 |
| Real, lr 1e-5 | 0.9236 | -2.4923 | 0.6460 | 72 | 3 | 0.2828 |
| Placebo, lr 1e-5 | 0.9229 | -2.4567 | 0.6511 | 71 | 3 | 0.2911 |
| Real, lr 1e-6 (descriptive) | 0.9250 | -2.4679 | 0.6511 | 71 | 3 | 0.2690 |

The labeled oracle threshold sits at 12.75 for the baseline and 13.0 for the
real arm, with oracle macro-F1 of 0.8879 in both cases. The valley moved by
0.15 logits; the gap it needs to close is 15 logits. The lr 1e-6 sensitivity
arm carries no confirmatory weight under the preregistration and is reported
only to show that the failure is not a step-size artifact.

## Descriptives

**The z distribution did not change shape.** The two KDE modes sit at -11.13
and 14.24 before adaptation and at -11.25 and 14.25 after it. Relative trough
depth goes from 0.2831 to 0.2828, a change of -0.0003. The congested middle the
mechanism promised to drain is still there. The median z is 9.75 in every arm.

**Individual scores barely moved.** After adaptation, 96 of 215 videos have a
bit-identical z. The median absolute change is 0.25, which is one step of the
readout's own quantization grid, and the largest change anywhere in the corpus
is 1.25 logits.

**No error changed class.** Not one of the 70 baseline false positives crossed
the adapted arm's own valley threshold, and neither did any of the 3 baseline
false negatives. Two true normals crossed in the wrong direction, which is
where the false-positive count of 72 comes from. Per-cohort median z change is
0.0 for the false positives, the false negatives, all 86 hateful videos and all
129 normal videos alike.

**The objective had almost no gradient to offer.** Mean entropy over the real
arm's 215 steps was 0.0403 nats with a median of 4.0e-05; 22 steps exceeded 0.1
nats and 193 sat below it. The placebo arm ran hotter, at mean 0.0716 nats and
31 steps above 0.1, which is the expected consequence of feeding the judge
mismatched evidence. Quarter-by-quarter means are not a training curve here:
each step is a different video, so the quarters differ in which videos they
contain as much as in how far training has progressed.

## Interpretation

The mechanism assumed that the congestion visible in the middle of the z
distribution is congestion in the model's *confidence*. It is not. A z of 9.75,
the corpus median, is a posterior of 0.99994; the entropy objective sees a
model that has already made up its mind. Only 39 of 215 videos carry more than
0.01 nats of answer entropy, and those are not concentrated where the
label-free threshold fails. Entropy minimization therefore optimizes a quantity
that is nearly constant over the corpus, and 215 steps at lr 1e-5 on 385,408
norm gains move the scores by less than the readout's own quantization step.

This separates two things the project had been treating as one. The z axis has
a wide dynamic range and a genuine bimodal structure, but the *probability* it
maps to is saturated almost everywhere. Any label-free objective defined on the
answer probability inherits that saturation and will be similarly inert. The
failure is not that entropy minimization is the wrong sharpening rule; it is
that there is nothing left to sharpen.

The falsifiable asymmetry stated in the preregistration was never reached.
HateMM was chosen as the decisive corpus because its annotation boundary
coincides with the model's own construct, so a mechanism that sharpens that
construct should show up here if anywhere. It did not show up here, so the
MHClip contrast has nothing to contrast against.

One negative result is worth keeping. Adaptation did not damage anything
either: AUC spans 0.0006 across the baseline and the two confirmatory arms, and
0.0020 once the descriptive lr 1e-6 arm is included, with bootstrap intervals
that overlap almost completely. Norm-gain adaptation on this judge is
inert rather than destructive, which rules out mode collapse as the explanation
and rules it in as a genuine absence of gradient signal.

## Decision

- Retire the entropy objective on this judge. Clause 1 failed on the family's
  cheapest and most favorable case, which is exactly the death the
  preregistration specified.
- Do not promote any successor built on the answer-probability posterior. The
  saturation census above applies to every such objective, not only to Shannon
  entropy, so a marginal-distribution or confidence-margin variant would fail
  for the same reason without needing a separate run.
- The wider label-free adaptation family is not closed by this test, but any
  successor must first exhibit a training signal that is not already saturated
  on this corpus, and must say so in advance with a measured quantity rather
  than an argument.
- Adapted norm gains for both arms are kept under `results/entropy_tta/`
  (gitignored), so nothing needs retraining if a follow-up wants to inspect
  what the 385,408 scalars actually learned.
