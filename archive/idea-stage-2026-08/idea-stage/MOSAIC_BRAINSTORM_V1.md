# MOSAIC: Multimodal Orbit-Ranked Same-Event Membership for Hateful Video Localization

## Status

- Selected after three rounds of method design, hostile review, and novelty review.
- Idea novelty before the new pilot: **6.5/10** (same-family provisional review).
- Evidenced novelty: **3/10** because Modules 2--3 have not yet passed their
  kill controls.
- The existing 611-video cohort remains development evidence only.
- PACT-style proposal persistence, topology, cohort routing, tribunals, and
  post-hoc certificates are explicitly excluded.

## Paper Thesis

Current label-free localizers identify high-scoring temporal regions, but do
not determine whether the context immediately across each endpoint still
belongs to the same multimodal hateful event. MOSAIC reformulates boundary
localization as **missing-aware, modality-balanced same-event membership**:
an endpoint may move only when timestamped language and at least one perceptual
modality independently support the same event identity under temporal-orbit
controls.

> We introduce a label- and training-free framework that edits each temporal
> boundary only when its adjacent evidence is independently attributable to
> the same hateful event by the available visual, acoustic, and linguistic
> views.

Data decoding, ASR loading, feature extraction, temporal resampling, and cache
construction are preprocessing and are not counted as modules.

## Module 1 -- Scale-Orthogonal Evidence Transport

This module preserves the empirically working core:

- exact timestamp boxes and duration-adaptive triangular transcript supports;
- qualification against all non-trivial circular transcript rotations;
- independent minimum-change projections;
- a centered hard-support correction for within-video temporal ranking;
- a video-constant hard/soft consensus correction for video propensity.

The output is a corrected dense field, timestamped chunk identity, and aligned
visual, acoustic, and linguistic evidence fields. The module does not emit the
final interval.

## Module 2 -- Missing-Aware Orbit-Rank Endpoint Membership

For candidate endpoint `e`, timestamped chunk `k`, and available modality
`m in {V, A, L}`, define the contrast between the endpoint interior and its
adjacent real shell:

\[
g_{m,e,k}=\operatorname{mean}_{t\in I_e}F_{m,k}(t)
-\operatorname{mean}_{t\in O_e}F_{m,k}(t).
\]

Compare the aligned contrast with the same chunk under every non-zero circular
rotation and convert it to an empirical rank:

\[
r_{m,e,k}=\frac{1+\#\{g^{\mathrm{rot}}_{m,e,k}<
g^{\mathrm{align}}_{m,e,k}\}}{1+N_{\mathrm{rot}}}.
\]

An interval obtains same-event membership only when:

1. its left and right endpoints are supported by the same chunk identity;
2. language and at least one perceptual modality are structurally available;
3. every available modality supports both endpoints above the frozen median
   orbit rank;
4. hard and triangular supports recommend the same endpoint direction.

The interval membership score is the weakest available view:

\[
M(c,k)=\min_{m,e} r_{m,e,k}.
\]

The minimum makes the rule non-compensatory: a large language/MLLM value cannot
erase a visual or acoustic contradiction. Audio is first-class when valid and
structurally absent when silent, corrupt, uncovered, or orbit-degenerate. A
negative audio vote must never be relabeled as missing.

## Module 3 -- Membership-Gated Bilateral Endpoint Lattice

The frozen tight--midpoint--broad geometry supplies legal endpoint candidates.
Left and right endpoints are edited independently, but the selected interval
must retain one shared chunk identity:

- choose the widest legal candidate with valid two-sided membership;
- fall back per endpoint on disagreement, ties, or invalid geometry;
- return the midpoint exactly when Module 1 made no field correction;
- never use dataset identity, cohort statistics, learned weights, or labels.

This module makes dense evidence and boundary action part of the same mechanism,
instead of using a separately inherited proposal geometry.

## Optional Upgrade -- Coalition Interaction Warrant

This is not part of V1. It is promoted only after the cached V1 pilot passes.
For the single most disputed shell on each side, query comparable modality
coalitions with one prompt, layout, output vocabulary, and token-logit scale.
After removing video propensity, compute the endpoint-level interaction:

\[
I_{VAT}=R_{VAT}-R_{VA}-R_{VT}-R_{AT}+R_V+R_A+R_T-R_\emptyset.
\]

The interaction may authorize a move only among candidates already admitted by
non-compensatory modality membership. It cannot rescue a candidate rejected by
an available modality. This upgrade is killed if the no-interaction control
retains at least 80% of its gain.

## Why This Is One Story

Module 1 establishes a scale-separated temporal evidence field. Module 2 asks
the missing question: whether the evidence immediately across each endpoint is
still attributable to the same multimodal event. Module 3 converts that event
membership into independent start/end actions. Removing the modules respectively
causes unqualified evidence, modality-compensating boundary contrast, or a score
field disconnected from interval geometry.

## Required Development Pilot

### Preflight: editable headroom

On the frozen endpoint lattice, the independent-endpoint oracle must offer at
least `+0.04` macro F1@0.5 or `+0.06` F1@0.7 over the current candidate, with a
positive ceiling on at least three datasets. Otherwise MOSAIC is stopped before
new model calls.

### Cached V1 pilot

Run Modules 2--3 on existing visual, audio, and language fields for all 611
development videos. Primary mechanism statistics are endpoint-action coverage,
same-chunk coverage, modality-availability strata, endpoint error, F1@0.7, and
within-video ROC.

Continue only if:

- at least 20% of evaluable videos receive a real endpoint action;
- F1@0.7 or endpoint error improves over the current method;
- within-video ROC falls by no more than 0.001;
- improvements are not confined to one dataset or one endpoint direction.

### Decisive kill control

Use the identical fields, orbit ranks, and endpoint lattice, but compute one
inside/outside contrast on the already fused corrected field. If this
**Fused-Field Endpoint Gate** matches or beats MOSAIC, modality-balanced
same-event membership is not load-bearing and the proposed novelty is killed.

Additional controls are best single modality, every modality pair, ordinary
2-of-3 voting, same-center/same-length geometry, transcript rotation, visual
frame rotation, audio temporal rotation, and removal of the transport core.

## Confirmation Gate

After freezing and hashing every window, availability rule, orbit comparison,
tie rule, and fallback, evaluate once on untouched same-domain and cross-domain
cohorts. The preregistered primary contrast is MOSAIC versus the fused-field
kill control on F1@0.7 or endpoint error. Evidenced novelty reaches 6/10 only
if the full method beats that control with a paired confidence interval whose
lower bound is positive, shows positive effects on at least three datasets,
and targeted rotation of each modality selectively removes its contribution.

## Claims Not Yet Authorized

- SOTA performance;
- statistically reliable improvement;
- causal multimodal interaction;
- confirmed novelty of at least 6/10;
- a persistence/topology contribution.
