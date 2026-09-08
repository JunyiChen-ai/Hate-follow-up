# Pre-registration — Generic-harm conditioned 2-D boundary

**Frozen:** 2026-08-08 after the linear nuisance probe and before fitting any
2-D mixture or reading any 2-D predictions.  
**Inputs:** the already frozen corrected-8B raw `z` and preregistered SigLIP
generic-harm feature `g_max`; HateMM and ImpliHateVid `train_clean` only.

## Motivation from the preceding frozen probe

On HateMM, `g` separates original-valley FP from TP with AUC `0.6685`, passing
the frozen 0.65 signal bar. But the full-corpus OLS slope is negative, so the
nonnegative linear-removal rule correctly makes no change. On ImpliHateVid the
slope is positive and removal destroys ranking. Generic harm is therefore a
conditional nuisance in the high-z region, not a globally removable factor.

## Frozen candidate

Robustly standardize `(z, logit(g))` per unlabeled corpus by median and MAD.
Fit one deterministic two-component full-covariance Gaussian mixture by EM:

- initialization: bottom and top z quartiles define the two initial means and
  covariances;
- variance floor `1e-3 I`;
- 500 iterations, tolerance `1e-8`;
- the component with higher mean standardized z is the positive component;
- predict positive iff its posterior is at least 0.5.

The full covariance is the mechanism: it permits the required z boundary to
rise with generic-harm nuisance where the fitted unlabeled geometry supports
that interaction. No labels choose K, initialization, covariance, posterior
cutoff, or standardization.

## Frozen ablations

1. **1-D GMM:** identical deterministic K=2 EM on standardized z alone.
2. **Shuffled g:** seed-20260808 permutation, identical 2-D fit.
3. **Diagonal covariance:** removes z×g interaction while preserving both
   marginal features.

## Pass rule

All must hold:

1. HateMM macro-F1 improves at least 0.10 over the frozen KDE valley.
2. HateMM hateful-vs-normal AUC of the positive posterior improves at least
   0.03 over raw z.
3. ImpliHateVid macro-F1 and AUC decline no more than 0.02.
4. Full covariance beats 1-D GMM and diagonal covariance by at least 0.03
   HateMM macro-F1.
5. Shuffled g retains less than half the real HateMM macro-F1 gain.

**Pass:** implement the full method and freeze held-out tests.  
**Fail:** generic-harm signal is diagnostic but not methodizable by unlabeled
density geometry; close this family and advance to the predeclared non-lexical
audio/prosody restoration probe.

