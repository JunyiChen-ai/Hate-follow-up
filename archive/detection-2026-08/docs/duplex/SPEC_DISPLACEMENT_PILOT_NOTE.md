# Spec-displacement pilot: result note

**Date:** 2026-08-10. **Verdict:** **DEAD** under the frozen rule.
**Compute:** one GPU pass, 1,110 forward passes in 5 min 46 s of scoring on an
RTX 5090 at 0.30 s per video, then under a minute of CPU analysis.
**Preregistration:** `docs/duplex/PREREG_spec_displacement_pilot.md`
**Machine-readable result:** `results/spec_displacement/results.json`

Reading the same video under two committed policy specs and subtracting the
layer-27 hidden states does produce a per-video signal, and the signal does
separate the videos whose class membership depends on the policy. It is not,
however, a spec interaction. On MHClip-EN the separation is genuinely paired but
too small; on HateClipSeg it is large enough but survives destroying the
pairing, because it is the strict arm's own state read through a new direction.
Neither preregistered readout passes all four clauses, so the family does not
proceed.

## What ran

Two new judge calls per video on 555 videos: the union offensiveness spec frozen
in the dual-axis appendix, and an off-construct spam-and-copyright spec written
into this pilot's preregistration and committed before the first call. The
strict protected-group arm already existed on disk and was not recomputed. Both
new arms reuse the frozen judge without alteration — prompt skeleton, `prag`
judgment block, system message, uniform 16-frame grid, pixel budget, uncapped
transcripts with each corpus's own override map, Yes/No token-id sets, and the
fp16 37 × 4096 per-layer dump at the final prompt position. Only the policy
sentence, the rule list and the scope sentence differ between arms.

Every frame of every video decoded before the model loaded, and the run asserted
161, 161, 394 and 394 finished cells before writing its marker. No video was
dropped or retried. The GPU stage queued behind another pilot for two hours and
never signalled it.

The union arm reproduces the dual-axis run exactly. Over all 161 MHClip-EN
videos the largest absolute difference between the new union z and the committed
`results/dual_axis/z_off_scores.jsonl` is 0.0, at Spearman and Pearson 1.000.
The two runs are the same computation, so the hidden states dumped here belong
to the scores already on record.

The strata rebuilt to their preregistered sizes exactly: 34 blind-coded
no-protected-target positives, 15 protected-target positives and 112 shipped
Normals on MHClip-EN; 344 union positives, 180 strict positives and therefore
164 union-positive strict-negative videos on HateClipSeg. HateClipSeg's
`lexicons.json` was never opened.

## Frozen clauses

Per readout, as preregistered. A readout survives only by passing all five rows
on its own; readouts are never mixed.

| Clause | c1, carrier-aligned displacement | c2, residual PC1 |
|---|---|---|
| C1a MHClip-EN flip separation | **FAIL** 0.710 (floor 0.72) | **FAIL** 0.663 (floor 0.74) |
| C1b HateClipSeg flip separation | **PASS** 0.724 (floor 0.70) | **FAIL** 0.295 (floor 0.72) |
| C2 shuffled-pairing placebo | **FAIL** margin 0.218 EN, 0.059 HCS (floor 0.10 on both) | **FAIL** margin 0.151 EN, −0.373 HCS |
| C3 off-construct policy arm | **PASS** 0.341 EN, 0.469 HCS (ceiling 0.60) | **FAIL** 0.720 EN, 0.507 HCS |
| C4 composition rule | **FAIL** 0.652 union, 0.527 strict (floors 0.6522 / 0.6767) | **FAIL** 0.652 union, 0.355 strict |

The pilot is DEAD. The carrier-aligned readout passes two of five clauses and
the residual axis passes none.

## The two corpora fail for opposite reasons

This is the result worth keeping. On MHClip-EN the displacement is a real paired
quantity: the shuffled placebo, which pairs video *i*'s union state with video
*j*'s strict state, falls to 0.492, chance, while the true pairing reaches 0.710.
Destroying the pairing destroys the signal, exactly as the mechanism predicts.
The signal is simply too small — 0.710 against a floor of 0.72, and against the
supervised probe's 0.837 on the same 49 videos.

On HateClipSeg the displacement clears the floor at 0.724, and the placebo
reaches 0.665. Three quarters of the above-chance separation survives pairing
each video's strict state with a stranger's union state. A post-hoc split of the
score explains why. Because c1 is `<h^union, d> − <h^strict, d>`, the placebo
leaves the second term's per-video identity intact, and that term alone is the
whole effect:

| Layer-27 quantity, HateClipSeg flip stratum | AUC | MHClip-EN | AUC |
|---|---:|---|---:|
| strict-arm projection alone, −`<h^strict, d>` | 0.721 | same | 0.682 |
| union-arm projection alone, `<h^union, d>` | 0.448 | same | 0.478 |
| the full displacement c1 | 0.724 | same | 0.710 |

On HateClipSeg the second arm contributes 0.003. The score is a single-call
readout of the strict arm's own hidden state, projected on a direction that
happened to be estimated from a difference. That is the readout-bottleneck
failure repeating itself: a new label-free direction in one call's states, doing
about as well as the scalar and no better. On MHClip-EN the same split shows the
union arm adding 0.028, and the near-zero rank correlation between c1 and the
strict term there, 0.041 against 0.519 on HateClipSeg, confirms that the two
corpora are measuring different things under the same name.

The scalar readout, read favourably, is the stronger score on the very stratum
this pilot was built for. On the HateClipSeg flip stratum `z_strict` scores 0.251
in the hypothesis direction, which is 0.749 reversed, above c1's 0.724. On
MHClip-EN it is 0.321, or 0.679 reversed, below c1's 0.710. One corpus each way,
with the displacement never clearly ahead.

## The off-construct control separates the two readouts

The spam-and-copyright policy carries no hostility, insult or protected-attribute
vocabulary and is matched to the union list on rule count, word count and, within
6 %, characters. Its displacement should not predict flips.

For the carrier-aligned readout it does not: 0.341 on MHClip-EN and 0.469 on
HateClipSeg, both below the 0.60 ceiling, and the MHClip-EN value is
anti-predictive rather than merely uninformative. Whatever c1 measures, it is not
a generic response to having the policy text rewritten.

For the residual axis the control fails outright. The spam arm's residual PC1
reaches 0.720 on MHClip-EN, above the real spec contrast's own 0.663 on the same
stratum. The leading direction of per-video residual variation is a property of
the prompt having changed, not of which two constructs it changed between. That
alone would have killed the residual readout even if its separation had cleared
the floor.

The residual axis also fails to transfer. Its direction, fixed once on MHClip-EN
as preregistered, gives 0.295 on HateClipSeg, so the axis points the opposite way
in the second corpus. The rank correlation between the two readouts is +0.846 on
MHClip-EN and −0.972 on HateClipSeg: they are near-duplicates whose sign relation
flips between corpora, which is the same instability seen from the other side.

## Energy decomposition

Share of the displacement's energy carried by the corpus-level carrier `‖m‖²`
against the mean per-video residual `mean‖r‖²`, with the residual's leading
component's variance share alongside.

| Corpus | Layer | Carrier share | Residual PC1 share | `‖m‖²` | `mean‖r‖²` | mean `‖Δh‖` |
|---|---:|---:|---:|---:|---:|---:|
| MHClip-EN | 18 | 0.608 | 0.177 | 4.7 | 3.0 | 2.8 |
| MHClip-EN | 27 | 0.341 | 0.425 | 592 | 1,143 | 37.9 |
| MHClip-EN | 36 | 0.208 | 0.590 | 4,030 | 15,376 | 125.0 |
| HateClipSeg | 18 | 0.601 | 0.103 | 4.7 | 3.1 | 2.8 |
| HateClipSeg | 27 | 0.177 | 0.582 | 203 | 940 | 29.2 |
| HateClipSeg | 36 | 0.122 | 0.631 | 1,630 | 11,736 | 97.6 |

The premise of the decomposition holds. At layer 18 the policy swap is mostly a
common carrier, four fifths of a very small displacement. By layer 27 the
per-video residual carries three to five times the carrier's energy, and by layer
36 five to nine times. There is abundant per-video interaction between the two
specs; it is simply not about the construct. The off-construct arm makes the same
point in reverse: on HateClipSeg its layer-27 carrier is fifteen times the union
arm's, 3,152 against 203, because a spam policy is a further move from a hate
policy than an offensiveness policy is — and its per-video residual still fails
to predict anything.

## The composition rule

The frozen rule takes the label-free KDE valley of the strict-arm z as the
offensive base set and removes from it the videos the c score flags. The valley
recipe reproduced its committed value exactly, −5.445, with the committed
per-collapse macro-F1 of 0.6522 under the union collapse and 0.4479 under the
strict one, so the composition was built on the same operating point as the
published one.

Applied to c1, the same valley recipe on the c distribution found three modes and
a cut at 38.19, flagging 32 of 394 videos. Removing them lifts the strict-collapse
macro-F1 from 0.4479 to 0.5267, well short of the 0.6767 the anchored rule
reaches, while the union collapse is inherited at 0.6522. The crossover control
passes — each decision is better on its own collapse than the other decision is,
0.527 against 0.448 under strict and 0.652 against 0.608 under union — so the
composition is spec-specific in the weak sense that it is oriented correctly. It
is simply far too small a correction. Applied to c2 the same recipe flags 362 of
394 videos and the strict collapse falls to 0.3546.

## Other numbers reported outside the verdict

Neither readout is prompt length in disguise: the rank correlation between each
score and the total prompt token count is at most 0.028 in absolute value on
either corpus.

The displacement directions are close to orthogonal to the direction that
actually carries the distinction. Refitting the readout-bottleneck probe on the
same 49 MHClip-EN videos and mapping it back to raw hidden-state space, the
cosine with the residual PC1 loading is 0.034 and with the carrier direction
0.037. The probe remains a measurement instrument and was not promoted to a
method component.

The layer sweep does not rescue the design. The carrier-aligned readout peaks
near layer 22 rather than the preregistered 27, at 0.769 on MHClip-EN and 0.725
on HateClipSeg, with 0.739 at layer 24 on HateClipSeg. At those layers the
MHClip-EN placebo is still at chance, 0.518, so the paired effect there is real
and larger than at 27; the HateClipSeg placebo is 0.615 to 0.665, so its margin
never clears 0.10 at any layer. Layer 0 sits at exactly 0.500 in every cell,
because the final prompt token is identical across arms and its embedding
displacement is exactly zero — a useful confirmation that the pipeline is
subtracting what it claims to subtract.

The stratum profiles show the predicted ordering only in part. The hypothesis was
that the score is large on the flip stratum and small both on protected-target
hate and on benign content. On MHClip-EN the c1 means are 31.0 for
no-protected-target positives, 23.2 for shipped Normals and 17.6 for
protected-target positives; on HateClipSeg they are 19.8 for the flip stratum,
11.7 for clean normals and 9.9 for strict positives. The flip stratum is highest
in both corpora, as predicted, but benign content sits in the middle rather than
at the bottom, and protected-target hate is lowest. The score behaves less like
"this video's membership depends on the policy" and more like "this video moves
little when the protected-group requirement is dropped, because the strict spec
already committed to it".

## Interpretation

The pilot was built on the observation that the decision boundary is
annotation-owned and that the missing input is the task specification. It asked
whether the judge represents the specification it was given, so that the
difference between two committed specifications would locate the videos the two
disagree about. The answer is that the judge does represent the specification —
the displacement is large, structured and mostly per-video by layer 27 — but the
per-video part of that representation is not organised around the construct
boundary.

Two independent controls say so. The residual axis, which is where a
spec-by-video interaction would have to live, is better predicted by an
off-construct spam policy than by the real construct swap. And the carrier-aligned
score, on the corpus where it clears the bar, is reproduced to within 0.003 by a
single arm's projection, meaning no interaction is being used at all.

This closes the route the readout-bottleneck note left open. That note ended by
observing that any future attempt on the targeted-hate versus generic-offence
distinction must obtain its direction from somewhere other than one judge call's
own unlabeled variance. The obvious remaining source was the contrast between two
committed specifications, since it supplies a direction from outside the corpus.
It has now been tried at the layer where the distinction is known to be linearly
present, with both a pairing placebo and an off-construct policy control, and the
direction it supplies is nearly orthogonal to the one that works: cosine 0.034
against the supervised probe.

What is left of the constraint-relaxation menu's fourth item is narrower than it
was. Task specification as a first-class input is not refuted as an idea; what is
refuted is reading it off the geometry of a second judge call. The specification
moves the states, and it moves them per video, but the movement is not a readout
of which specification the video is a borderline case for.

## Decision

The spec-displacement family does not proceed. No follow-on preregistration is
written, and the two-call budget's second slot is not spent this way.

Three facts are recorded for reuse. The union arm's hidden states now exist for
both corpora and are byte-consistent with the committed dual-axis scores, so any
later analysis of that spec is free. The off-construct spam policy is a working
negative control for policy-swap experiments, and it separated the two readouts
here where the pairing placebo alone would not have. And the layer-0 identity
check plus the exact z reproduction give this pipeline a pair of cheap integrity
tests worth keeping in later runs.

No video id, title or transcript text reaches any committed artifact.
