# Commitment-state precision across annotation boundaries: result note

**Date:** 2026-08-08. **Verdict:** **FAIL** under the frozen rule. Clause 1
requires the saturation band to be at least 80 percent strict hate in every
powered corpus, and HateClipSeg reaches 0.689 with a 95 percent interval of
0.616 to 0.755, so the upper end of the interval does not reach the threshold
either. Clauses 2 and 3 both pass.
**Compute:** CPU only, a few seconds, on existing scores and existing
annotations. No model call was made.
**Preregistration:** `docs/duplex/PREREG_commitment_precision.md`
**Code:** `scripts/duplex/commitment_precision.py`
**Machine-readable result:** `results/commitment_precision/results.json`

## What was measured

The claim under test was that the positive saturation band, raw judge score at
or above +13, is the model's commitment state for unambiguous hate
specifically. Three quantities were computed on each of the five scored corpora.
The first is the strict-hate precision of the saturation band. The second is the
strict-hate precision of the interior comparison band, +5 up to but not
including +13. The third is the rate at which each annotation stratum enters
the saturation band.

Strict hate means the shipped binary hate label on HateMM and ImpliHateVid, the
hateful-strict collapse of the video-level category set on HateClipSeg, and the
Hateful class alone on MHClip-EN and MHClip-ZH. The MHClip three-class field is
the `Label` key of `annotation(new).json`, which carries Hateful, Offensive and
Normal verbatim, so the strict collapse is read directly from the shipped
annotation rather than reconstructed. The offensive-only stratum is the set of
videos that are positive under the offensive-inclusive collapse and negative
under the strict collapse. All intervals are exact Clopper-Pearson binomial
intervals at 95 percent.

A cell counts as powered when it holds at least 30 saturation-band videos. That
leaves ImpliHateVid, HateMM and HateClipSeg inside the verdict, with MHClip-EN
and MHClip-ZH descriptive, exactly as the preregistration anticipated.

No component was fitted in this analysis. Every label use is evaluative.

## Strict-hate precision by band

| Corpus | Powered | Band n | Saturation precision | 95 percent CI | Interior n | Interior precision | 95 percent CI | Gap |
|---|---|---:|---:|---|---:|---:|---|---:|
| ImpliHateVid | yes | 70 | 0.9714 | 0.9006 to 0.9965 | 98 | 0.8469 | 0.7601 to 0.9117 | 0.124 |
| HateMM | yes | 81 | 0.8765 | 0.7847 to 0.9392 | 50 | 0.2200 | 0.1153 to 0.3596 | 0.657 |
| HateClipSeg | yes | 183 | 0.6885 | 0.6160 to 0.7548 | 136 | 0.3088 | 0.2325 to 0.3937 | 0.380 |
| MHClip-EN | no | 5 | 0.2000 | 0.0051 to 0.7164 | 33 | 0.0909 | 0.0192 to 0.2433 | 0.109 |
| MHClip-ZH | no | 2 | 1.0000 | 0.1581 to 1.0000 | 57 | 0.2281 | 0.1274 to 0.3584 | 0.772 |

The gap column is the saturation precision minus the interior precision. The
two MHClip rows are reported for completeness and carry no weight: five and two
videos give intervals wide enough to accommodate almost any hypothesis.

## Saturation rate by annotation stratum

Each cell is the fraction of that stratum whose score reaches +13.

| Corpus | Strict hateful | 95 percent CI | Offensive only | 95 percent CI | Normal | 95 percent CI |
|---|---:|---|---:|---|---:|---|
| ImpliHateVid | 0.3417 (68/199) | 0.2761 to 0.4121 | not applicable | | 0.0100 (2/201) | 0.0012 to 0.0355 |
| HateMM | 0.8256 (71/86) | 0.7287 to 0.8990 | not applicable | | 0.0775 (10/129) | 0.0378 to 0.1379 |
| HateClipSeg | 0.7000 (126/180) | 0.6274 to 0.7659 | 0.2927 (48/164) | 0.2243 to 0.3687 | 0.1800 (9/50) | 0.0858 to 0.3144 |
| MHClip-EN | 0.0769 (1/13) | 0.0019 to 0.3603 | 0.0833 (3/36) | 0.0175 to 0.2247 | 0.0089 (1/112) | 0.0002 to 0.0487 |
| MHClip-ZH | 0.1176 (2/17) | 0.0146 to 0.3644 | 0.0000 (0/28) | 0.0000 to 0.1234 | 0.0000 (0/104) | 0.0000 to 0.0348 |

HateMM and ImpliHateVid ship a binary hate label, so they have no offensive-only
stratum and their normal column is the whole negative class.

## Clause-by-clause verdict

| Clause | Requirement | Result |
|---|---|---|
| 1. The band is strict hate | Strict-hate precision at least 0.80 in every powered cell | **FAIL**: HateClipSeg 0.6885 against the 0.80 floor, a shortfall of 0.115. ImpliHateVid 0.9714 and HateMM 0.8765 both hold |
| 2. Strict over offensive asymmetry | On HateClipSeg, the saturation rate among strict-hateful videos is at least twice the rate among offensive-only videos | PASS: 0.7000 against 0.2927, a ratio of 2.39 |
| 3. The band exceeds the interior | Saturation precision above interior precision by at least 0.10 in every powered cell | PASS: gaps of 0.124, 0.657 and 0.380 |
| **Overall** | All three clauses hold | **FAIL** |

## MHClip strict recollapse, descriptive

Reported outside the verdict, to complete the five-corpus regime map. Macro-F1
under the strict collapse, where only Hateful maps to 1. The union column
repeats the committed offensive-inclusive numbers for reference.

Predictions were reproduced point by point from the fit parameters stored in
`results/anchored_operating_point/results.json`, not refitted. The valley uses
its stored threshold, and each mixture uses the posterior-0.5 rule applied to
its stored component means, standard deviations and mixing weight. The
reproduction is exact: recomputing the union-collapse macro-F1 from the same
parameters reproduces every committed value to zero absolute difference, which
is the self-check the script aborts on. The two oracles are refitted against the
strict labels, because an oracle is label-fitted by construction.

| Method | EN strict | EN union | ZH strict | ZH union |
|---|---:|---:|---:|---:|
| KDE valley | 0.4912 | 0.6962 | 0.5274 | 0.7256 |
| Free 2-GMM | 0.4965 | 0.6932 | 0.5938 | 0.7932 |
| Anchored +15.0 | 0.4756 | 0.4292 | 0.4698 | 0.4111 |
| Labeled oracle, macro-F1 max | 0.5442 | 0.6962 | 0.7472 | 0.7932 |
| Labeled oracle, hateful-F1 max | 0.4840 | 0.6888 | 0.7472 | 0.7932 |

Under the strict collapse the oracle threshold moves up sharply, to +8.5 on EN
and +11.25 on ZH, against -0.25 and +5.0 under the union collapse. Every
label-free rule stays below its oracle by a wide margin on EN. The anchored rule
is the only one whose strict number exceeds its union number, and it is still
the worst label-free rule on both corpora. The anchored fit is degenerate on
both MHClip corpora, with a positive component of weight 0.011 on EN, so this
column carries no information about the anchoring idea whatever the labels say.

The three-class composition of the saturation band itself, counts only: MHClip-EN
places 1 Hateful, 3 Offensive and 1 Normal video in the band, and MHClip-ZH
places 2 Hateful and no others.

## Interpretation

The band is not a strict-hate commitment state. The preregistration set the
clause-1 floor at 0.80 and stated in advance what a clause-1 failure means:
retire the commitment interpretation entirely, including the mechanism gloss
that was attached to the HateMM result. HateClipSeg misses that floor by 0.115,
and the whole 95 percent interval sits below it, so the failure is not a power
problem. That corpus is also the largest powered cell, at 183 band videos
against HateMM's 81 and ImpliHateVid's 70, so it is the cell with the most
evidence, not the least.

What survives is weaker and more graded than the claim. The band is strongly
enriched for strict hate relative to the interior on every powered corpus, by
between 0.12 and 0.66 in precision, and the enrichment is monotone in score on
all three. On HateClipSeg the strict-hateful stratum enters the band at 2.39
times the rate of the offensive-only stratum, so the strict and offensive
categories do separate along the saturation axis. Those are real ordering
effects. They describe a score that ranks strict hate above offensive content,
which is what any usable score would do, and they do not establish a discrete
state that the model enters only for unambiguous hate.

The clause-2 pass is worth keeping separate from the clause-1 failure. The
asymmetry the operating-point results needed does exist: at 0.700 against 0.293
the strict class reaches the band far more readily than the offensive-only
class. But an asymmetry in rates is not the same as purity in the band. The
HateClipSeg band contains 126 strict-hateful, 48 offensive-only and 9 normal
videos, so nearly a third of it is not strict hate. A rule that treats band
membership as a commitment to hate would be wrong about one video in three
there.

The corpus ordering also has an alternative explanation the preregistration
warned about. Precision in the band tracks how clean each corpus's negative
class is, at 0.971 on ImpliHateVid with an almost empty negative tail, 0.877 on
HateMM, and 0.689 on HateClipSeg, whose negatives are 164 offensive-only videos
against only 50 genuinely normal ones. The band looks purest exactly where there
is least ambiguous content available to contaminate it. That is a property of
the annotation pools, not evidence about the model's internal geometry.

The MHClip recollapse closes the regime map without rescuing anything. Moving
those two corpora to the strict boundary lowers every label-free number, from
0.696 to 0.491 on EN and from 0.726 to 0.527 on ZH for the incumbent valley, and
the oracle falls too on EN. The strict boundary is harder there, not easier, so
the pattern that motivated this preregistration, that anchoring wins on strict
boundaries and loses on offensive-inclusive ones, does not extend to the two
corpora where the band is nearly empty. Occupancy, not the label definition, is
what those two corpora lack.

## Decision

- Retire the commitment-state interpretation. The band is a high-score region
  enriched for strict hate, and it should be described that way from now on,
  including anywhere the HateMM anchored result was glossed as the model
  committing to hate.
- Do not rescue the claim by dropping HateClipSeg from the powered set or by
  lowering the 0.80 floor. Both moves were available before the run and neither
  was taken, and HateClipSeg is the largest powered cell.
- Keep the two surviving measurements as descriptive facts. Saturation-band
  precision exceeds interior precision on every powered corpus, and the strict
  stratum enters the band at more than twice the offensive-only rate on the one
  corpus that can test it. Both are reportable; neither is a mechanism.
- Record that three preregistrations in this line have now failed:
  `ANCHORED_OPERATING_POINT_NOTE.md`, `GATED_ANCHOR_HATECLIPSEG_NOTE.md`, and
  this one. The first two failed on performance and the third on mechanism, so
  the anchoring idea has no remaining defence from either direction.
- The open problem is unchanged from the previous note and now better bounded. A
  label-free rule needs a prevalence signal, and the saturation band does not
  supply one, because band membership does not mean what the commitment story
  said it meant.

## Implementation choices not fixed by the preregistration

- Saturation rates are reported with Clopper-Pearson intervals as well, although
  the preregistration required intervals only on the precisions. They cost
  nothing and the clause-2 comparison is easier to read with them.
- The HateClipSeg annotation is a category set rather than a three-class field,
  so its stratum assignment is derived from the union and strict pair rather
  than from a single label. This matches the collapse code frozen in the B1
  pilot. `lexicons.json` was not read.
- Both oracle conventions from the predecessor script are carried into the
  strict recollapse table. The hateful-F1-maximising oracle is reported as
  macro-F1 so the column is comparable; its selection criterion is unchanged.
