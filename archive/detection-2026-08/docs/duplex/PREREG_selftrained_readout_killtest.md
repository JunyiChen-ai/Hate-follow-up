# Pre-registration — Self-trained readout kill test (commitment-extreme pseudo-labels, hidden-state head)

**Frozen:** 2026-08-09, before fitting any head on any corpus.
**Compute:** CPU only; hidden states and scores already on disk. No new
model calls, no GPU.
**Status:** kill test. Combines the two surviving facts of this program:
(a) layer-27 hidden states contain judgment-relevant structure the scalar
readout discards (supervised probe 0.837 where z scores 0.321,
`READOUT_BOTTLENECK_NOTE.md`); (b) the z extremes are high-precision on
strict-hate corpora while the middle is where all threshold errors live
(`COMMITMENT_PRECISION_NOTE.md`; HateMM band precision 0.8765, all 70
valley false positives mid-band).

## Claim under test

The model's own committed extremes can supervise a better reading of its
own representation. Pseudo-labels taken from the frozen saturation bands
(z ≥ +13 positive, z ≤ −13 negative; bands frozen since the saturation
pilot) train a linear head on layer-27 hidden states; that head re-reads
the whole corpus, including the congested middle that the scalar readout
cannot resolve. No human labels, no benchmark labels, no new calls;
deployment cost is one stored hidden vector and one dot product per video.

Disclosure of adjacency: the falsification map kills self-confidence
GATING — using confidence to route or veto decisions at inference. This
design uses the extremes once, offline, as a training set for a readout;
no inference-time gating exists. The distinction is declared here so the
result is judged against it.

## Frozen protocol

Per corpus, fully label-free:

1. Pseudo-label set: videos with z ≥ +13 → 1, z ≤ −13 → 0. If either side
   has fewer than 10 videos, the method abstains on that corpus (reported
   as not-applicable; no fallback tuning).
2. Head: L2 logistic regression (lambda = 1.0, fixed) on standardized
   layer-27 final-token hidden states (mean/std from the corpus's own
   unlabeled data). Fit once on the pseudo-labeled extremes only.
3. Decision: head posterior ≥ 0.5 on every video, including the extremes
   themselves. No threshold search of any kind.
4. Controls, same pipeline: (i) **z-only control** — identical logistic
   fit and decision using the scalar z as the sole feature (tests whether
   hidden space, not mere recalibration, is load-bearing); (ii) **shuffled
   pseudo-label placebo** — labels permuted within the extreme set (seed
   20260808, 20 permutations, mean performance reported).

## Corpora

Primary: HateMM test (baseline valley macro-F1 0.6562, oracle 0.888;
extremes 57 positive-band / well-populated negative band). Secondary,
same frozen pipeline, reported in the same note: ImpliHateVid test
(regression check), HateClipSeg (strict collapse), MHClip-EN and ZH
(expected not-applicable or weak by the occupancy facts; their behavior is
reported, not load-bearing).

## Frozen decision rule

The design **SURVIVES** only if all four clauses hold on HateMM:

1. **Performance:** head macro-F1 ≥ 0.75 (same bar the entropy-adaptation
   test failed).
2. **Ranking preserved:** AUC of the head's posterior ≥ 0.9132 (baseline
   0.9232 − 0.01).
3. **Hidden space is load-bearing:** head macro-F1 ≥ z-only control
   macro-F1 + 0.03.
4. **Signal, not band arithmetic:** shuffled-placebo mean macro-F1 ≤
   head macro-F1 − 0.10.

Regression guard, also required: ImpliHateVid head macro-F1 ≥ its valley
baseline (0.8823) − 0.03.

## Interpretation boundaries

- Pass: licenses the full method preregistration — cross-corpus protocol,
  the composition with channel restoration, and the deployment story
  (single call, stored hidden state, linear head). Not itself the method
  result.
- Clause 1 or 2 fails: extreme-supervision does not resolve the middle;
  retire the family honestly.
- Clause 3 fails: the gain is recalibration of z, not representation
  access — the hidden-state story collapses; do not promote a z-rescaling
  as a method.
- Clause 4 fails: performance comes from band membership counts, not
  learned structure; dead.
- Labels are used only in evaluation, as everywhere in this project.
