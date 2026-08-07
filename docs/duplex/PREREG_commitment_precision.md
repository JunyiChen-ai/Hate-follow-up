# Pre-registration — Commitment-state precision across annotation boundaries

**Frozen:** 2026-08-08, before computing any label statistic conditioned on the
saturation band on any corpus.
**Compute:** CPU only; existing scores and existing annotations, no new model
calls.
**Status:** mechanism analysis, not a threshold method. Successor to the two
FAILED operating-point preregistrations
(`ANCHORED_OPERATING_POINT_NOTE.md`, `GATED_ANCHOR_HATECLIPSEG_NOTE.md`).

## Motivating pattern (post-hoc, disclosed)

Across the five scored corpora, the anchored threshold won exactly where the
positive class is strict hate (HateMM 0.8491; HateClipSeg hateful-strict
0.6767, all clauses passing) and lost exactly where the positive class
includes broader offensive content (HateClipSeg offensive-union;
MHClip-EN/ZH, whose binary collapse is Hateful+Offensive → 1 and where the
saturation band is nearly empty). HateClipSeg showed this within a single
corpus on byte-identical predictions: only the label collapse changed, and
the winning label-free rule flipped. This motivates a sharper mechanism
claim than "anchoring helps sometimes".

## Claim under test

The positive saturation band (z ≥ +13, band frozen from the saturation-anchor
pilot) is the model's commitment state for **unambiguous hate specifically** —
not a generic high-score region and not an "offensive content" region. If
true, the operating-point results stop being a patchwork: a label-free
threshold derived from the model's commitment geometry recovers the
strict-hate annotation boundary, and no z-only rule can recover an
offensive-inclusive boundary, because the model does not commit on
offensive-but-not-hateful content.

## Frozen data and definitions

- Scores: the five existing 8B test score files (ImpliHateVid, HateMM,
  MHClip-EN, MHClip-ZH testruns; HateClipSeg results/hateclipseg).
- Strict-hate label per corpus: HateMM hate label; ImpliHateVid hate label;
  HateClipSeg hateful-strict collapse; MHClip-EN/ZH Hateful class only
  (Offensive and Normal both → 0) from the 3-class annotation.
- Offensive-only stratum (corpora with both boundaries): union-positive AND
  strict-negative videos (HateClipSeg; MHClip-EN/ZH).
- Saturation band: z ≥ +13. Interior comparison band: +5 ≤ z < +13.
- A cell is **powered** iff it has ≥ 30 saturation-band videos: HateMM (81),
  ImpliHateVid (70), HateClipSeg (183). MHClip-EN (5) and ZH (2) are
  descriptive only.
- All label use is evaluative (measuring what the band contains); nothing is
  fit.

## Frozen readout

Per corpus: n and strict-hate precision of the saturation band with exact
binomial 95% CI; same for the interior band; saturation RATE within the
strict-hateful stratum, the offensive-only stratum, and the normal stratum.

## Frozen decision rule

The commitment-precision claim **PASSES** only if all clauses hold:

1. In every powered cell, P(strict-hate | z ≥ +13) ≥ 0.80.
2. On HateClipSeg (the powered corpus with both boundaries), the saturation
   rate among strict-hateful videos is ≥ 2× the saturation rate among
   offensive-only videos.
3. In every powered cell, saturation-band strict precision exceeds
   interior-band strict precision by ≥ 0.10 (the band is more than a smooth
   continuation of the interior; this clause bounds magnitude, it does not by
   itself prove a discrete state).

Descriptive, outside the verdict: the MHClip strict recollapse of all
existing comparators (valley, free 2-GMM, anchored, oracle) on EN and ZH —
reported to complete the five-corpus regime map, with the caveat that
anchored is degenerate there whatever the labels say.

## Interpretation boundaries

- Pass: licenses the follow-up's mechanism narrative — commitment geometry
  encodes the model's own strict-hate boundary; label-free thresholding is
  well-posed for strict-hate annotation schemes and provably ill-posed for
  offensive-inclusive ones given only z. It does NOT by itself produce a new
  operating-point rule.
- Clause 1 failing: the band is not a hate-commitment state; retire the
  commitment interpretation entirely, including the HateMM result's
  mechanism gloss.
- Clause 2 failing: the strict/offensive asymmetry is not what separates the
  occupied and unoccupied regimes; the occupancy difference needs another
  explanation (e.g., language or style), and the "model's strict boundary"
  story dies.
- This analysis uses labels to characterize a mechanism, matching the
  precedent of the HateMM false-positive premise audit.
