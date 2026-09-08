# Pre-registration — Occupancy-gated anchored operating point, confirmed on HateClipSeg

**Frozen:** 2026-08-08, before any judge scoring of HateClipSeg and before any
threshold fitting on it.
**Compute:** one GPU pass to score HateClipSeg with the frozen c2 pipeline
(fresh Whisper large-v3 transcripts, degeneracy gate, uniform-16 frames,
single-call Qwen3-VL-8B judge, raw z); CPU for all fitting.
**Status:** successor to `PREREG_anchored_operating_point.md`, whose
unconditional form FAILED (note: `ANCHORED_OPERATING_POINT_NOTE.md`).

## What the failed experiment established

The anchored mixture (positive component pinned at the model-owned +15.0
saturation state) collapses exactly where the saturation band is unoccupied:
MHClip-EN has 5/161 videos at z ≥ +13 (3.1%), MHClip-ZH has 2/149 (1.3%, max
z = 13.5), and there the pinned component degenerates and classifies the
whole corpus as normal. Where the band is occupied — HateMM 81/215 (37.7%),
ImpliHateVid 70/400 (17.5%) — the anchored fit is the best label-free
threshold measured on HateMM (macro-F1 0.8491 vs valley 0.6562, free 2-GMM
0.8173), the anchor location is load-bearing there (placebo +10 falls to
0.6460, +20 to 0.3750), and it is the only prevalence-robust rule on HateMM
(median resample macro-F1 0.808/0.749 at retention 0.5/0.25, vs free 2-GMM
0.670/0.524 and valley 0.583/0.498).

## Claim under test

The saturation state's usefulness is conditional on its occupancy, and
occupancy is measurable label-free. A gated rule — anchor when the model
exposes its commitment state on this corpus, fall back to a fully corpus-fit
mixture when it does not — selects the correct regime on a corpus it has
never seen.

## Post-hoc disclosure

The gate statistic, its cutoff, and the fallback arm were chosen AFTER seeing
the four test corpora. They carry no confirmatory weight there. All
confirmatory weight rests on HateClipSeg, which no judge in this project has
ever scored.

## Frozen method

1. Score HateClipSeg (all 436 annotated videos, or all with usable media)
   with the exact c2 test pipeline. Label collapse, frozen from the B1 pilot
   code: primary = offensive-union → 1; secondary = hateful-strict → 1.
   `lexicons.json` remains quarantined and unused.
2. Gate statistic: fraction of corpus with z ≥ +13 (band frozen from the
   saturation-anchor pilot). **Gate fires iff ≥ 10%.**
3. If the gate fires: anchored 2-GMM, μ+ = +15.0 pinned, protocol identical
   to `PREREG_anchored_operating_point.md` (EM, 50 restarts, seed 20260808,
   posterior-0.5 decision). If not: free 2-GMM, same protocol, decision =
   posterior of the higher-mean component ≥ 0.5.
4. No labels enter the gate or either fit.

## Frozen comparators and stress test

KDE valley, free 2-GMM, anchored (both computed regardless of gate), placebo
anchors +10.0 / +20.0, labeled oracle. Prevalence resampling exactly as in
the predecessor prereg (retention {0.5, 0.25} × 200, seed 20260808).

## Frozen decision rule (primary label collapse)

The gated mechanism **PASSES** only if all applicable clauses hold:

1. **Correct selection:** the arm the gate selects has macro-F1 within 0.02
   of the better of {anchored, free 2-GMM} (comparison uses labels for
   evaluation only).
2. **If the gate fires:** anchored ≥ valley + 0.03; anchored ≥ free 2-GMM −
   0.005; anchored ≥ each placebo − 0.01; anchored median threshold drift ≤
   0.5× free-2-GMM drift at both retention rates.
3. **If the gate does not fire:** free 2-GMM ≥ valley − 0.005, and the
   (forced) anchored fit underperforms free 2-GMM by more than 0.02 —
   i.e., the gate demonstrably avoided a real failure.

Clause 1 failing kills the gated method outright: a gate that picks the wrong
regime on the first unseen corpus is fitted noise. The secondary label
collapse is reported for robustness but does not enter the verdict.

## Interpretation boundaries

- Expected regime, stated before scoring: HateClipSeg's normal class contains
  violence-only videos, the same contamination structure as HateMM, so we
  expect the occupied regime and a fired gate. If the gate does not fire and
  clause 3 still passes, the method survives but the HateMM-contamination
  story for WHY corpora differ in occupancy is weakened; say so.
- This experiment simultaneously provides the first cross-benchmark
  measurement of the c2 channel-restoration pipeline itself (judge AUC,
  valley baseline) on a fifth corpus; those numbers are reported
  descriptively, outside the decision rule.
