# Anchored operating point: result note

**Date:** 2026-08-08. **Verdict:** **FAIL** under the frozen rule; all three
clauses fail.
**Compute:** CPU only; existing 8B test-split scores, no new model calls.
**Preregistration:** `docs/duplex/PREREG_anchored_operating_point.md`
**Code:** `scripts/duplex/anchored_operating_point.py`
**Machine-readable result:** `results/anchored_operating_point/results.json`

Pinning the positive mixture component at the model-owned saturation location
+15.0 costs 0.10 mean macro-F1 relative to the incumbent KDE valley and 0.15
relative to a fully free two-component mixture. The placebo anchor at +10.0
beats the preregistered anchor by 0.09 mean macro-F1. The one corpus the method
was designed for, HateMM, does improve by 0.19, but a free mixture recovers most
of that improvement without any anchor.

## What was run

Each of the four held-out test corpora was fitted separately on its own 8B
raw-z score distribution. The primary model is a two-component one-dimensional
Gaussian mixture whose positive component mean is held at +15.0 while its
standard deviation, the whole negative component and the mixing weight are
estimated by EM with 50 random restarts under seed 20260808. A video is called
hateful when the posterior of the positive component reaches 0.5. The
comparators are the same fit with the anchor moved to +10.0 and to +20.0, the
same fit with the positive mean free, the incumbent KDE-valley threshold, and a
labeled oracle threshold.

The KDE-valley recipe and the dataset label loading are imported unmodified
from `scripts/duplex/crossbench_analyze.py`, the module that produced the
committed `docs/duplex/reports/test_c2_*.json` files. The valley macro-F1
reproduced to the last digit on all four corpora, so the score and label
loading in this experiment is the same loading as in those reports. The script
aborts if the reproduction error exceeds 0.002.

## Per-corpus macro-F1

| Corpus | Anchored +15.0 | Placebo +10.0 | Placebo +20.0 | Free 2-GMM | KDE valley | Labeled oracle |
|---|---:|---:|---:|---:|---:|---:|
| ImpliHateVid | 0.8721 | 0.8747 | 0.3344 | 0.8609 | 0.8823 | 0.8925 |
| HateMM | 0.8491 | 0.6460 | 0.3750 | 0.8173 | 0.6562 | 0.8879 |
| MHClip-EN | 0.4292 | 0.6299 | 0.4103 | 0.6932 | 0.6962 | 0.6962 |
| MHClip-ZH | 0.4111 | 0.7719 | 0.4111 | 0.7932 | 0.7256 | 0.7932 |
| **Mean** | **0.6404** | **0.7306** | **0.3827** | **0.7911** | **0.7401** | **0.8175** |

The oracle column is the macro-F1-maximizing labeled threshold. The committed
test-split reports maximize hateful-class F1 instead; that convention is also
recorded in the JSON and differs only on MHClip-EN, where it gives 0.6888.

The decision boundaries these models place, in raw-z units:

| Corpus | Anchored +15.0 | Placebo +10.0 | Free 2-GMM | KDE valley | Oracle |
|---|---:|---:|---:|---:|---:|
| ImpliHateVid | -5.85 | -5.58 | -10.57 | -4.45 | -2.25 |
| HateMM | 10.38 | -2.92 | 9.35 | -2.35 | 12.75 |
| MHClip-EN | 14.62 | 5.14 | -2.18 | -0.50 | -0.25 |
| MHClip-ZH | none | 6.17 | 4.57 | 0.61 | 5.00 |

Every mixture decision region is bounded on both sides, because the two fitted
variances are unequal and the posterior therefore crosses 0.5 twice. On three
corpora the second crossing falls outside the score range and the region behaves
like an ordinary threshold. On MHClip-EN it does not: the anchored fit collapses
the positive component to a spike of standard deviation 0.26 carrying 1.1
percent of the mixing weight, so the decision region is the narrow band
[14.62, 15.40] and the single highest-scoring video in the corpus is called
normal. On MHClip-ZH the anchored component receives essentially zero weight,
the posterior never reaches 0.5, and the model calls every video normal.

## Prevalence stress test

Positives were subsampled to 50 percent and to 25 percent of their original
count, negatives kept whole, 200 resamples per rate under seed 20260808, with
all three label-free thresholds refitted on every resample. Drift is the median
absolute distance in raw-z units between the resample threshold and the
full-corpus threshold, taking the posterior crossing nearest the median score of
the set being fitted.

| Corpus | Rate | Valley drift | Free 2-GMM drift | Anchored drift | Valley F1 | Free 2-GMM F1 | Anchored F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| ImpliHateVid | 0.50 | 2.05 | 0.74 | 4.13 | 0.8716 | 0.8153 | 0.8213 |
| ImpliHateVid | 0.25 | 4.22 | 1.33 | 4.31 | 0.8273 | 0.7377 | 0.7605 |
| HateMM | 0.50 | 0.10 | 5.46 | 0.32 | 0.5831 | 0.6695 | 0.8082 |
| HateMM | 0.25 | 0.22 | 10.39 | 0.65 | 0.4980 | 0.5243 | 0.7493 |
| MHClip-EN | 0.50 | 0.99 | 0.56 | 0.09 | 0.6471 | 0.6348 | 0.4516 |
| MHClip-EN | 0.25 | 1.60 | 0.62 | 0.16 | 0.5959 | 0.5741 | 0.4746 |
| MHClip-ZH | 0.50 | 0.88 | 0.92 | none | 0.6694 | 0.7312 | 0.4522 |
| MHClip-ZH | 0.25 | 1.88 | 1.93 | none | 0.6200 | 0.6337 | 0.4749 |

Three entries need reading with care. The MHClip-ZH anchored drift is
undefined because no crossing exists in 200 of 200 resamples at rate 0.50 and
in 193 of 200 at rate 0.25. The MHClip-EN anchored drift of 0.09 and 0.16 is
the stability of the degenerate spike, not the stability of a usable operating
point, and it is computed on the 125 and 113 resamples where a crossing exists
at all. The KDE valley failed to find two modes in 19 and 34 MHClip-EN
resamples, and those resamples are excluded from its two MHClip-EN rows.

The ImpliHateVid anchored drift of about 4.2 is a systematic shift rather than
instability: the threshold moves from -5.85 at the full corpus to a tight
cluster near -10.0 once positives are subsampled, with an interquartile spread
below 0.5. Pinning the positive mean did not stop the boundary from tracking
prevalence on that corpus.

## Clause-by-clause verdict

| Clause | Requirement | Result |
|---|---|---|
| 1. Anchor is load-bearing | Anchored mean macro-F1 exceeds each placebo by at least 0.01 | **FAIL**: +0.258 against placebo +20.0 but -0.090 against placebo +10.0 |
| 2a. Not an ordinary mixture | Anchored mean at least free-2-GMM mean minus 0.005 | **FAIL**: 0.6404 against 0.7911, a shortfall of 0.151 |
| 2b. Not an ordinary mixture | Anchored drift at most half the free-2-GMM drift on at least 3 of 4 corpora, at every rate | **FAIL**: 2 of 4 at rate 0.50 and 2 of 4 at rate 0.25 |
| 3a. Performance on HateMM | HateMM macro-F1 at least valley plus 0.08 | PASS: 0.8491 against 0.6562, a gain of 0.193 |
| 3b. Performance on the mean | Mean macro-F1 at least valley mean plus 0.03 | **FAIL**: 0.6404 against 0.7401, a shortfall of 0.100 |
| 3c. No corpus regresses | No corpus below valley minus 0.03 | **FAIL**: MHClip-EN -0.267 and MHClip-ZH -0.315 |

Clause 1 fails, which under the frozen rule retires the anchor as a mechanism
regardless of any F1 number. Clause 2 fails as well, so the part of the result
that does work is the ordinary-mixture operating point and not the anchor.

## Interpretation

The anchor is not the active ingredient in the HateMM improvement. The free
mixture, given no anchor at all, lands its positive component at +14.55 on
HateMM and reaches 0.8173 macro-F1 against the anchored model's 0.8491. Most
of the 0.19 gain over the valley is therefore attributable to replacing a
density-minimum rule with a generative two-component decomposition, which is
precisely the ordinary-mixture result that the earlier generic-harm probe
declined to promote.

The anchor's failure has a simple and checkable cause: the +15.0 state is
barely occupied outside HateMM and ImpliHateVid. MHClip-EN has 2 videos out of
161 scoring at or above +15.0, and MHClip-ZH has none, its maximum score being
+13.5. A component pinned where there is no data either collapses to a
near-zero-weight spike or vanishes, and in both cases the corpus is classified
as entirely normal. The saturation pilot established that the positive state,
when occupied, sits reliably at +15.0; it did not establish that every corpus
occupies it. The one-sided successor inherited that gap and the gap is what
broke it.

The placebo comparison makes the same point from the other side. The +10.0
placebo, which the preregistration expected to be inferior because it sits
outside the replication tolerance, beats the preregistered anchor on three of
four corpora and by 0.09 on the mean. It wins for a reason unrelated to
saturation geometry: +10.0 is inside the occupied range of all four corpora, so
its component always has data to explain. What the experiment measures is
therefore occupancy, not location, and location was the claim.

The stress test does contain one real and non-trivial finding, and it is worth
keeping separate from the verdict. On HateMM the free mixture's threshold is
violently prevalence-sensitive, drifting 5.46 and 10.39 raw-z units, while the
anchored threshold drifts 0.32 and 0.65 and holds a median macro-F1 of 0.81 and
0.75 where the free mixture falls to 0.67 and 0.52 and the valley to 0.58 and
0.50. Where the anchored state is genuinely occupied, pinning it does remove a
degree of freedom that prevalence otherwise exploits. That effect is real on
one corpus, absent on ImpliHateVid, and untestable on the two MHClip corpora.
One corpus is not a mechanism.

## Decision

- Retire the one-sided anchored operating point as a method. Clause 1 failed,
  which the preregistration made decisive on its own.
- Do not promote the free two-component mixture either. It beats the valley
  mean by 0.05 and beats it by 0.16 on HateMM, but it is the ordinary-mixture
  trick with no hateful-video-specific story, and the generic-harm precedent
  already refused it. Promoting it now would be a post-hoc reversal.
- Record, but do not build on, the prevalence-robustness observation on HateMM.
  Any successor would first have to explain why a corpus should be expected to
  occupy the +15.0 state at all, and would need a label-free way to detect
  non-occupancy before fitting. Without that test the same failure repeats on
  any corpus whose scores stay below the anchor.
- The HateMM valley-to-oracle gap, 0.656 against 0.888, is still open. This
  experiment did not close it in a way that survives its own falsification
  rule.

## Implementation choices not fixed by the preregistration

- EM stops when the total log-likelihood changes by less than 1e-10 or after
  5000 iterations. Every reported fit converged before the cap.
- Component standard deviations are floored at 0.1 and mixing weights are
  clipped to [1e-6, 1 - 1e-6] to prevent collapse onto a single point.
- Predictions come from the per-video posterior, never from a scalar threshold,
  so the two-sided decision regions are handled exactly. The reported threshold
  is the posterior crossing nearest the median score of the set being fitted,
  and the full boundary set is recorded in the JSON.
- The labeled oracle is reported under both the macro-F1-maximizing convention
  used above and the hateful-F1-maximizing convention used in the committed
  test-split reports.
