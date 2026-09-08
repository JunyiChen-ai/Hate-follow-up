# Closeout — the label-free operating-point program, five corpora

**Date:** 2026-08-08. This note consolidates the four preregistered
experiments run today, all of which FAILED under their frozen rules, and
records what the program established before closing.

## What was run (all frozen before results; commits in order)

1. **Unconditional anchored operating point** (prereg faa8e85, result
   6e14304): pin the positive mixture component at the model's replicated
   +15.0 saturation location. FAIL — the component collapses wherever the
   saturation band is unoccupied (MHClip-EN 3.1%, MHClip-ZH 1.3%), and a
   placebo anchor at +10.0 beats it on the four-corpus mean.
2. **Text-region pixel-budget probe** (prereg faa8e85 + frozen scheme
   a9d374a): reroute visual tokens to native-resolution crops of detected
   text on MHClip-EN. FAIL on all three clauses — TEXT gains +0.004 AUC
   against a required +0.03, the text-avoiding control ranks highest, and
   93.8% of donor-frame text was already legible at baseline resolution.
   MHClip-EN's low AUC is not a visual-text delivery problem.
3. **Occupancy-gated anchor, confirmed on HateClipSeg** (prereg 74bdc08,
   result 3acdc12): gate fired correctly (46.4% occupancy, the highest
   measured) and selected the better mixture arm, but the KDE valley beats
   both mixtures under the offensive-union collapse because the corpus is
   87.3% positive and the valley tracks that prevalence. FAIL.
4. **Commitment-state precision** (prereg 210f361, result 8215671): is the
   saturation band the model's unambiguous-hate state? FAIL — on
   HateClipSeg the band is only 68.9% strict-hateful (CI 0.62–0.75, floor
   0.80), containing 48 offensive-only and 9 normal videos. The band is a
   graded enrichment, not a semantic state. Per the prereg, the commitment
   interpretation is retired, including the HateMM gloss.

## The five-corpus regime map (8B judge, restored channel, macro-F1)

| Corpus | Positive class | AUC | Valley | Free 2-GMM | Anchored | Oracle |
|---|---|---:|---:|---:|---:|---:|
| ImpliHateVid | strict (implicit) hate | 0.947 | 0.882 | 0.861 | 0.872 | 0.893 |
| HateMM | strict hate | 0.923 | 0.656 | 0.817 | 0.849 | 0.888 |
| MHClip-EN | hate + offensive | 0.785 | 0.696 | 0.693 | 0.429 | 0.696 |
| MHClip-ZH | hate + offensive | 0.855 | 0.726 | 0.793 | 0.411 | 0.793 |
| HateClipSeg (union) | hate + offensive | 0.754 | 0.652 | 0.563 | 0.547 | 0.670 |
| HateClipSeg (strict) | strict hate | 0.771 | 0.448 | 0.641 | 0.677 | 0.720 |

## What the program established

1. **No z-only label-free rule is universal, and none can be.** HateClipSeg
   demonstrates this within one corpus: on byte-identical predictions, the
   winning rule flips when the label collapse changes (valley 0.652 vs
   anchored 0.547 under the union; valley 0.448 vs anchored 0.677 under
   strict). The decision boundary is a property of the annotation scheme;
   the score distribution does not determine it. Any future proposal for a
   universal unsupervised threshold on judge scores must answer this
   demonstration.
2. **The oracle ceiling itself is thin.** The labeled-oracle mean over the
   four original test corpora is 0.8175, about one point above TRIAGE's
   0.808. Threshold selection, even solved perfectly, cannot deliver a
   meaningful margin over the base method. The real gap is the judge's
   ranking on MHClip-EN (0.785) and HateClipSeg (0.754).
3. **Replicated engineering facts, kept without a mechanism story:** where
   the saturation band is occupied, a pinned positive component is the most
   prevalence-robust threshold measured (drift 0.22–0.23× the free
   mixture's on HateClipSeg, resample macro-F1 0.808/0.749 vs 0.670/0.524
   on HateMM) and its location is load-bearing (both placebos collapse on
   both occupied corpora). Saturation-band strict-hate precision exceeds
   interior precision by 0.12–0.66 everywhere (graded enrichment).
4. **The evidence-side search on the weak-ranking corpora is exhausted.**
   Temporal coverage (B1), spatial text legibility (today), speaker
   provenance tags (Sortformer arms), prosody, and visual generic-harm 2-D
   all ran and died under frozen rules. The channel-restoration mechanism
   itself remains confirmed and is neutral only where its precondition
   (starved speech channel) is absent.

## What retires

Symmetric two-anchor geometry (owner's pilot), one-sided unconditional
anchor, occupancy-gated anchor, and the commitment-state semantic
interpretation. No further variant in this family should run without a new
phenomenon that explains the HateClipSeg band composition (126 strict / 48
offensive-only / 9 normal).

## The open front

The judge's ranking on MHClip-EN and HateClipSeg is the only place where
performance is still being lost and no autopsy has ever been done (the only
completed autopsy, the HateMM false-positive audit, is what produced the
last confirmed mechanism). The next step is a ranking-error autopsy on
those two corpora: what the misranked videos are, in aggregate terms, and
which attributable hypotheses survive contact with them.

## Addendum, same day: the open front resolved

The autopsy ran (`RANKING_AUTOPSY_NOTE.md`) and its two central claims were
then CONFIRMED by a preregistered blind audit
(`PREREG_annotation_validity_audit.md`, `ANNOTATION_VALIDITY_NOTE.md`) with
score-blind cross-family coders and a model-independent duplicate-transcript
check. Blind rates: 36.4% of HateClipSeg's clean-normal stratum contains
protected-group hostility or extremist glorification (CI 0.22–0.52, floor
0.20); 69.4% of MHClip-EN's union positives contain no protected-group
target (CI 0.55–0.82, floor 0.40), while coders still endorse the shipped
positive label on 84% of them — the class is offensive on an axis the
judge does not measure, not benign. Corrected AUC: HateClipSeg 0.754 →
0.895; MHClip-EN restricted to blind protected-target positives 0.785 →
0.866. Together with the regime map above, the weak-corpus story closes:
the judge is a construct-consistent ranker of protected-group hate; the
residual gaps on these two corpora are substantially properties of their
labels (contamination and construct mixing), not of the model. The
evidence-delivery search stays closed, and no method component is licensed
by this finding — what it licenses is label-corrected evaluation reporting
alongside shipped-label numbers.
