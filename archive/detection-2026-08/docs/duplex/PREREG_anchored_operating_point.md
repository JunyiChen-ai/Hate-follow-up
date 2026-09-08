# Pre-registration — Anchored operating point (one-sided positive-saturation successor)

**Frozen:** 2026-08-08, before fitting any anchored or baseline mixture on the four
test-split score files.
**Compute:** CPU only; no new model calls. Inputs are the existing
`results/testruns/{implihatevid,hatemm,mhclip_en,mhclip_zh}/judge_8b/scores.jsonl`.
**Status:** threshold-method experiment. This is the one-sided successor that
`SATURATION_ANCHOR_PILOT_NOTE.md` said would require a new preregistration.

## Claim under test

The judge's positive saturation state is model-owned: its location (+15.0,
tolerance ±2.5) replicated in 6/6 reader-condition cells while occupancy moved
by up to 1.64×. The KDE-valley threshold fails on HateMM because the middle of
the score distribution is corpus-owned and bends. The claim: fixing the
hateful-class component's location at the model-owned saturation state — while
leaving the background component and the mixing weight corpus-fit — yields a
label-free operating point that (a) recovers most of the HateMM oracle gap,
(b) is robust to prevalence shift where a fully corpus-fit mixture is not, and
(c) depends on the anchor being at the true saturation location.

How a one-sided anchor supports a full decision line: prevalence and the
normal-class geometry are legitimately corpus properties and must be fit; what
the anchor supplies is identifiability of the positive component under
occupancy/prevalence change. E7 showed free mixture component locations drift
up to 1.66× of inter-mode distance under prevalence resampling; a pinned
positive location removes that degree of freedom.

## Frozen method (primary)

For each test corpus separately, on the 8B raw-z scores:

- Two-component 1-D Gaussian mixture fit by EM.
- Positive component mean **fixed at μ+ = +15.0** (frozen constant; never
  re-estimated, identical across all four corpora). σ+, μ−, σ−, and mixing
  weight π are free.
- EM: 50 random restarts (seed 20260808), keep best log-likelihood;
  initialization must not use labels.
- Decision: predict hateful iff posterior of the anchored component ≥ 0.5.
- No labels, no per-dataset switching, no other tuning.

## Frozen comparators

1. **KDE valley** (current method) — the incumbent.
2. **Free 2-GMM** — same EM protocol, μ+ also free. This is the "ordinary
   mixture operating point" that the generic-harm probe refused to promote.
3. **Placebo anchors** — same as primary but μ+ ∈ {+10.0, +20.0} (both outside
   the ±2.5 replication tolerance).
4. **Labeled oracle threshold** — upper reference only.

## Frozen stress test (prevalence resampling)

For each corpus: retain the positive class at rates {0.5, 0.25} (labels used
only to construct the resample), 200 resamples per rate (seed 20260808).
Refit valley, free 2-GMM, and anchored model per resample. Record threshold
drift = median |threshold − full-corpus threshold| in logits, and macro-F1 on
the resample.

## Frozen decision rule

The mechanism **PASSES** only if all clauses hold:

1. **Anchor is load-bearing:** mean macro-F1 across the four corpora of the
   primary model exceeds each placebo-anchor variant by ≥ 0.01.
2. **Not just an ordinary mixture:** (a) mean macro-F1 of the primary model ≥
   free-2-GMM mean − 0.005, AND (b) at every resampling rate, the anchored
   model's median threshold drift is ≤ 0.5× the free-2-GMM's median drift on
   at least 3 of 4 corpora.
3. **Performance:** HateMM test macro-F1 ≥ valley + 0.08; mean macro-F1 across
   the four corpora ≥ valley mean + 0.03; no corpus below valley − 0.03.

Failure of clause 1 retires the anchor as a mechanism (whatever the F1).
Failure of clause 2 means the result is the ordinary-mixture trick and must
not be promoted, matching the generic-harm precedent. Failure of clause 3
alone means the mechanism is real but does not pay; report honestly.

## Interpretation boundaries

- 2B scores are out of scope here: the anchor pilot only validated the 8B
  positive state, and 2B showed no saturation anchors in E7.
- MHClip-EN is threshold-blameless (judge AUC is the binding constraint), so
  clause 3 expects approximately no change there; that is consistent with the
  claim, not evidence against it.
- No transcript text or video content enters this analysis.
