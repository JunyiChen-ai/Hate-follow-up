# Scale-Orthogonal Transport PoS — Frozen Candidate V1

## Status

- Task-specific idea novelty: **6/10** (same-family reviewer).
- Implemented/evidenced novelty: **5/10**.
- Evaluation status: retrospective development on the 611-video common test cohort.
- SOTA / confirmatory claim: **not yet authorized**.

## Method Modules

Data decoding, ASR loading, temporal resampling, and cache construction are
preprocessing and are not counted as modules.

### Module 1 — Dual-Support Temporal Evidence Transport

Each timestamped transcript chunk defines two temporal measurements:

1. an exact hard support over its observed timestamp span;
2. a duration-adaptive triangular support whose shoulders model temporal
   uncertainty.

Sparse transcript confidence relations are retained only after empirical
qualification against all non-trivial circular transcript rotations. Each
support view independently induces a minimum-change convex projection of the
LESS dense logit field.

### Module 2 — Scale-Orthogonal Consensus Projection

The hard projection owns the centered within-video temporal residual. The
mean of the sign-agreeing hard/soft geometric correction owns only a constant
video-level propensity shift. Formally,

\[
\Delta z_v(t)=\Delta z_h(t)-\overline{\Delta z_h}
+\overline{\Delta z_{hs}^{\mathrm{agree}}}.
\]

The constant term cannot alter within-video ranking. Conflicting hard/soft
frame corrections do not enter the propensity consensus.

### Module 3 — Authority-Gated Endpoint Lattice

LESS supplies a fixed tight–midpoint–broad endpoint lattice. The corrected
field may traverse left and right endpoint shells only when Module 2 actually
changes that video's field. Otherwise the decoder exactly returns the frozen
midpoint baseline. A shell is admitted when its corrected-field mean is at
least the video's corrected-field median.

## Development-Cohort Results

| Metric | LESS midpoint | Full V1 | Delta |
|---|---:|---:|---:|
| Frame PR-AUC | 0.464324 | 0.465964 | +0.001640 |
| Frame ROC-AUC | 0.688602 | 0.689916 | +0.001315 |
| Within-video ROC-AUC | 0.632558 | 0.632969 | +0.000411 |
| Interval F1@0.3 | 0.329104 | 0.330281 | +0.001176 |
| Interval F1@0.5 | 0.285437 | 0.286613 | +0.001176 |
| Interval F1@0.7 | 0.247547 | 0.247850 | +0.000303 |

All six point estimates are positive. Existing paired bootstrap confidence
intervals include zero. These results are exploratory, not confirmatory.

## Load-Bearing Evidence

- Hard-only projection owns the within-video gain.
- Adding the soft-support propensity term raises pooled PR/ROC while preserving
  hard-only within-video ROC exactly.
- Disabling Module 3 restores the midpoint interval metrics; enabling it gives
  positive point-estimate deltas at all three IoU thresholds.
- A circular-shift full control has worse within-video ROC and worse F1@0.3 and
  F1@0.7, although its pooled metrics are higher, consistent with pooled
  metrics being sensitive to video-level propensity.

## Frozen Confirmation Gate

On an untouched cohort, run exactly:

1. LESS midpoint;
2. hard-only projection;
3. full Modules 1–2 with Module 3 disabled;
4. full Modules 1–3;
5. circular-shift/mismatched-transcript full control.

Primary metric: within-video ROC-AUC. Key secondary metric: interval F1@0.7.
Also test the pre-registered module-specific contrasts: full vs hard-only
pooled PR/ROC, hard-only vs LESS within-video ROC, and Module 3 enabled vs
disabled interval F1. Claims require multiplicity-aware paired confidence
intervals and must not be promoted from the development cohort.
