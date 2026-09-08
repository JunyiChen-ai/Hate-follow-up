# TRACE Candidate V1

## Status

**Current best performance candidate; novelty claim remains scoped.**

TRACE is a label-free vision-language dense localization method.  It improves
the frozen Scale-Orthogonal field from `0.632969` to `0.649943` macro
within-video ROC on the four-dataset 611-video development protocol, with a
dataset-balanced paired 95% interval of `[0.002467, 0.032185]`.  The direction
is positive on all four datasets, pooled ROC also improves, pooled PR is
effectively unchanged, and the inherited intervals remain bit-identical.

The same frozen operation order transfers to a different VASTA field on the
276-video HCS cohort: within-video ROC improves from `0.549303` to `0.564264`,
with paired CI `[0.000577, 0.029463]`.  This cohort had already been used to
evaluate the T-core mechanism, so this is a cross-base replication of the GCV
extension, not an untouched end-to-end confirmation.

## Method

The data preparation used to obtain frame, ASR, and timestamp-language
features is deliberately excluded from the module count.

### Module 1 — Base-Agnostic Semantic Event Field

A frozen label-free localizer produces:

- a dense visual/multimodal field `q(t)`;
- a video-level propensity represented by the field mean;
- a tight semantic event core `C`.

Scale-Orthogonal LESS is the main instantiation and VASTA is the transfer
instantiation.  This module exposes a common field/core interface; it does not
use frame labels or train a task-specific model.

### Module 2 — Propensity-Preserving Semantic-Core Transport

Timestamp-language embeddings inside `C` form a core prototype.  Their dense
cohesion field is converted into a zero-mean correction and transported into
the base logits with fixed gain `0.01`.  A scalar offset then restores the
original mean probability exactly.  Thus language may change within-video
ordering but cannot change the video's average hateful propensity.

This is the central novel module.  Its supported role is semantic temporal
reranking, not hate-existence voting and not endpoint control.

### Module 3 — Self-Calibrated Temporal Proximal Stabilization

The transported logit residual is smoothed on a content-independent temporal
chain.  The regularization strength is selected independently for every video
by generalized cross-validation over the frozen path
`{0.01, 0.04, 0.16, 0.64, 2.56}`.  No labels, dataset identity, or cohort-level
statistics enter the selection.  The output is shifted to preserve video mean
probability exactly; interval geometry stays frozen.

This module is an effective adaptive stabilizer, not a claimed language graph:
the aligned transcript graph lost to a uniform chain and was rejected.

## Unified view

The sequential implementation can be written as a semantic-force plus
temporal-regularity objective,

\[
\min_{\mathbf 1^\top u=0}
\frac12\lVert u-r_V\rVert_2^2
-\eta\langle c_T,u\rangle
+\frac{\lambda}{2}u^\top L_{\mathrm{time}}u,
\]

with fixed semantic gain `eta` and label-free per-video `lambda`.  Factorial
controls show that the two corrections are approximately additive, so the
paper must not claim nonlinear synergy or order dependence.

## Four-dataset development results

| Method | PR-AUC | ROC-AUC | Within ROC | F1@.3 | F1@.5 | F1@.7 |
|---|---:|---:|---:|---:|---:|---:|
| Scale-Orthogonal base | 0.465964 | 0.689916 | 0.632969 | 0.330281 | 0.286613 | 0.247850 |
| T-core only | 0.463615 | 0.689777 | 0.645130 | 0.330281 | 0.286613 | 0.247850 |
| Uniform GCV only | **0.466035** | **0.694752** | 0.648553 | 0.330281 | 0.286613 | 0.247850 |
| **TRACE: T-core then GCV** | 0.465888 | 0.694651 | **0.649943** | 0.330281 | 0.286613 | 0.247850 |

Per-dataset within-video ROC for TRACE:

| Dataset | Base | TRACE | Delta |
|---|---:|---:|---:|
| HateMM | 0.652828 | 0.672637 | +0.019809 |
| HateClipSeg | 0.529706 | 0.533553 | +0.003847 |
| MHC | 0.696953 | 0.727229 | +0.030276 |
| MHC-zh | 0.652387 | 0.666355 | +0.013968 |

## Factorial and order controls

| Arm | Within ROC |
|---|---:|
| base | 0.632969 |
| T only | 0.645130 |
| GCV only | 0.648553 |
| reverse: GCV then T | 0.649039 |
| additive corrections | 0.649934 |
| sequential: T then GCV | 0.649943 |

Sequential and additive are numerically indistinguishable.  Consequently,
TRACE is supported as a composition of two independently useful operations,
not as a learned or emergent interaction.

## Cross-base HCS replication

| VASTA arm | PR-AUC | ROC-AUC | Within ROC |
|---|---:|---:|---:|
| base | 0.477214 | 0.554019 | 0.549303 |
| T only | 0.484878 | 0.563681 | 0.561152 |
| GCV only | 0.486617 | 0.564523 | 0.554388 |
| **T then GCV** | 0.485842 | 0.564540 | **0.564264** |

- full minus base: `+0.014961`, CI `[0.000577, 0.029463]`;
- full minus T-only: `+0.003113`, CI `[0.000168, 0.006279]`;
- full minus GCV-only: `+0.009877`, CI `[-0.003988, 0.023697]`.

## Same-protocol baseline context

The current 611-video confirmation files give the following four-dataset macro
values.  These are internally comparable because they use the same 4-fps GT
archives and evaluator, although some external methods skip videos that lack a
valid generated output.

| Method | PR-AUC | ROC-AUC | Within ROC | F1@.3 | F1@.5 | F1@.7 |
|---|---:|---:|---:|---:|---:|---:|
| VideoTGB adaptation | 0.300071 | 0.521496 | 0.539106 | 0.188615 | 0.114808 | 0.041195 |
| VidGroup zero-shot | 0.318413 | 0.520392 | 0.615386 | 0.289886 | 0.163131 | 0.026462 |
| TimeLens-8B adaptation | 0.328502 | 0.572757 | 0.579513 | 0.135635 | 0.063324 | 0.026837 |
| **TRACE** | **0.465888** | **0.694651** | **0.649943** | **0.330281** | **0.286613** | **0.247850** |

This supports “best observed under the internal unified protocol” against these
three recent open baselines.  A broad SOTA claim still requires adding every
previously reproduced label-free baseline, including T3AL, and verifying exact
video/rate/evaluator parity.

## Mechanisms that were tested and rejected

- MLLM canvas/signature authorization;
- audio-language additive, orthogonal, and concordance transport;
- local core-shell density-ratio transport;
- aligned transcript-graph smoothing;
- cohesion-discontinuity endpoint selection;
- dense- and chunk-based multi-span decoding.

These are not hidden ablations of TRACE and must not reappear as claimed
modules.

## Current claim and remaining gate

Safe claim:

> TRACE performs label-free semantic-then-structural temporal transport: a
> frozen event core induces a propensity-preserving language correction, and a
> per-video GCV proximal operator stabilizes the resulting dense ordering.

Current novelty assessment is approximately `6/10` for the task-specific
semantic-core transport contribution and `5.5/10` for the full composition.
The full-system novelty may only be raised after a truly untouched confirmation
shows that the complete method beats base and both single-operation controls.
No interval improvement is claimed.

