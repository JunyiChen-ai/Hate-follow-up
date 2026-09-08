# COHERE: Core-Anchored Heterogeneous Evidence Transport

## Status

**V1 strong acoustic-linguistic claim is killed by the factorial placebo.**
Aligned AT reaches within-video ROC `0.646518`, but shifted-T AT reaches
`0.645977` and jointly shifted AT still reaches `0.643369`.  Alignment is not
responsible for most of the gain.  V1 is retained as a useful generic temporal
cohesion observation, not promoted as the final method.

The subsequent structured MLLM event-signature V2 is also killed.  On a
balanced 30-video pilot, factual VAT signature transport scores `0.619316`
within-video ROC at gain `0.01`, versus `0.619442` for the same-length shifted
core.  At mechanism-amplification gain `0.1`, factual scores `0.628181` versus
shifted `0.628818`.  The generated semantic signature is not load-bearing.

COHERE is the current development candidate.  It was selected after AMBI,
fused-orbit endpoint testing, and modality-knockout endpoint voting failed
their controls.  It is evaluated on all 611 videos across HateMM,
HateClipSeg, MHC, and MHC-zh.

The current cohort and its ground truth were used to compare modality subsets
and residual gains, so the result is development-selected and not untouched
confirmation.  SOTA and evidenced novelty >=6 remain unconfirmed.

Decoding, ASR, feature extraction, temporal resampling, cache loading, and
timestamp alignment are preprocessing and are not modules.

## Thesis

Existing label-free methods conflate two different questions:

1. **semantic relevance** -- where is evidence of hateful content strongest?
2. **event cohesion** -- which surrounding moments belong to the same episode
   as that evidence?

COHERE assigns these questions to different, complementary multimodal roles
instead of symmetrically averaging every modality at every frame:

> We formulate label-free hateful-video localization as core-anchored
> heterogeneous evidence transport: semantic evidence discovers an event
> seed, while complementary acoustic-linguistic cohesion transports only
> within-video support around that seed without changing video propensity.

## Module 1 -- Scale-Orthogonal Semantic Seeding

The working visual/query field and timestamped MLLM language support are
decomposed into:

- video-level hateful propensity;
- a centered within-video localization residual.

Hard timestamp support owns temporal correction; hard/soft consensus may alter
only video propensity.  Qualified corrections are imposed by minimum-change
projection.  The module emits a corrected dense semantic field and a
conservative tight event core.

The core answers **what candidate hateful episode anchors localization?**  It
does not define the final frame ranking or interval alone.

## Module 2 -- Role-Specialized Core Cohesion Transport

Audio and timestamp-aligned language embeddings within the event core form two
normalized event prototypes.  Every temporal position receives acoustic and
linguistic cosine cohesion to those prototypes.  Per-view within-video ranks
are combined by an equal-authority geometric mean.

The resulting cohesion is centered and robustly scale-matched to the semantic
field, then transported in logit space with a frozen small gain.  A scalar
minimum-change correction preserves the original video's mean probability
exactly.  Therefore this module can reorder frames inside a video but cannot
create a stronger video-level hate verdict.

Visual evidence is not absent: it owns semantic seed discovery in Module 1.
Audio and language own continuation/cohesion in Module 2.  This deliberate
division avoids counting the same visual evidence twice and empirically works
better than symmetric V/A/T cohesion on the development cohort.

## Module 3 -- Authority-Gated Endpoint Lattice

The transported dense field owns frame ranking.  Interval geometry remains
under the existing conservative tight/mid/broad endpoint lattice:

- left and right endpoints are selected independently;
- corrected fields may authorize lattice movement;
- the tight core cannot be deleted;
- failure falls back per endpoint.

In V1, cohesion does not directly move endpoints.  This intentionally keeps the
positive frame-ranking mechanism separate from the still-unverified cohesion
boundary decoder.

## Development result on all four datasets

Selected development arm: acoustic + linguistic core cohesion, gain `0.01`.

| Metric | Previous candidate | COHERE V1 | Delta |
|---|---:|---:|---:|
| pooled frame PR-AUC | 0.465964 | 0.463820 | -0.002144 |
| pooled frame ROC-AUC | 0.689916 | 0.689814 | -0.000102 |
| within-video macro ROC-AUC | 0.632969 | **0.646518** | **+0.013549** |
| interval F1@0.3 | 0.330281 | 0.330281 | 0 |
| interval F1@0.5 | 0.286613 | 0.286613 | 0 |
| interval F1@0.7 | 0.247850 | 0.247850 | 0 |

Dataset-balanced paired bootstrap for within-video ROC gives a 95% interval of
`[0.005552, 0.022084]`, with `p(delta <= 0) = 0.00015` over 20,000 samples.

Within-video gains are positive on all four datasets:

| Dataset | Previous | COHERE | Delta |
|---|---:|---:|---:|
| HateMM | 0.652828 | 0.661839 | +0.009011 |
| HateClipSeg | 0.529706 | 0.538074 | +0.008368 |
| MHC | 0.696953 | 0.722146 | +0.025193 |
| MHC-zh | 0.652387 | 0.664012 | +0.011625 |

## Completed controls

- Visual-only, audio-only, language-only, VA, VT, AT, and VAT cohesion arms;
- arithmetic/median versus geometric cohesion;
- gains `0.01, 0.025, 0.05, 0.125, 0.25, 0.5` for VAT transport;
- exact fallback on videos without an event core;
- exact preservation of video mean probability;
- unchanged interval readout.

The AT arm is strongest and positive on all four datasets.  This was discovered
on the evaluation cohort, so it is a hypothesis, not confirmation.

## Mandatory kill controls

1. Replace the true event core with same-length circularly shifted cores for
   both A and T; rerun the complete prototype and transport path.
2. Pair the true audio core with a shifted language core and vice versa.  The
   aligned AT arm must beat both misaligned controls.
3. Use whole-video A/T prototypes.  If it matches the core prototype, anchoring
   is not load-bearing.
4. Use scene/position kernels with no semantic core.  If they match COHERE,
   the gain is generic temporal smoothness.
5. Compare frozen AT against the best single modality and equal-gain additive
   noise under identical mean preservation.
6. Confirm that Module 1 without MLLM language support and Module 2 without
   audio each lose measurable performance.

The first factorial implementation failed this gate: shifting audio reduced
the result more than shifting text, while shifting text preserved nearly all
of the aligned score.  Therefore the proposed acoustic-linguistic interaction
is not load-bearing in V1.

## V2 direction -- Structured Event-Signature Transport

**Outcome: killed by the shifted-core control.**

The next pilot replaces self-similarity prototypes with an MLLM-derived,
structured event signature from the semantic core:

- depicted source/actor;
- protected target;
- hostile action or proposition;
- author/speaker stance;
- observable acoustic cue.

Frozen cross-modal encoders then ground the same signature densely in visual,
audio, and timestamped language streams.  This retains the empirically useful
core-to-context transport but makes the transported object semantic rather than
an arbitrary segment prototype.  It is viable only if the true-core signature
beats same-length shifted-core signatures and each modality-specific signature
field passes targeted corruption controls.

## Surviving narrow mechanism

The only new mechanism still positive on all four development datasets is
text-only core cohesion transport at gain `0.01`:

- macro within-video ROC `0.644014` versus base `0.632969`;
- positive direction on HateMM, HateClipSeg, MHC, and MHC-zh;
- near-preserved pooled metrics and bit-identical interval predictions.

This is frozen as a prospective confirmation hypothesis.  It does not support
the original acoustic-linguistic interaction claim.  Its novelty must be
reviewed as role-separated visual-semantic seeding plus timestamp-language
event cohesion.

## Additional killed directions

- core-oriented V/A/T kernel change-points improve only F1@0.3 by `0.00397`
  and leave F1@0.5/F1@0.7 unchanged;
- dense excursion-set multi-span decoding produces 1,093--2,999 spans and
  collapses precision;
- timestamped positive chunk spans, alone or appended to the base interval,
  reduce every interval F1 metric.

## Confirmation gate

Freeze the AT subset, gain, core rule, reliability rule, feature backbones,
mean-preserving solver, and fallback before evaluation on an untouched cohort.
Require:

- positive paired within-video delta with CI lower bound above zero;
- positive direction on at least three of four datasets;
- pooled ROC non-inferiority within `0.002` and pooled PR within `0.005`;
- aligned core cohesion beats shifted-core, whole-video, and best-single-view
  controls;
- interval metrics do not regress;
- comparison against T3AL and all strongest label-free baselines under the same
  video IDs, frame grid, and evaluator.

## Novelty framing

Closest paradigms include anchor-aware similarity cohesion in supervised video
moment retrieval and training-free prompt decomposition/self-attention in
spatial action grounding.  COHERE's task-specific distinction is the
role-separated, mean-preserving transport from a label-free hateful event seed
using complementary audio-language event cohesion.

The intended claim is a new **semantic-seed / event-cohesion decomposition** for
label-free multimodal hateful-video localization, not a new backbone and not an
MLLM boundary oracle.

## 2026-08-28 role-orthogonal and endpoint follow-up

Two richer versions were implemented as deterministic mechanism tests on the
611-video development cohort.  Neither survived, so they must not be included
as positive modules in the final method.

### ROUTE: role-orthogonal audio admission -- killed

`project_role_orthogonal_transport.py` keeps vision in the frozen base field,
uses timestamp-language core cohesion as the first correction, residualizes
audio against the base, language, and a quadratic time trend, and tests both
additive and noncompensatory concordance transports.  Every correction uses the
same fixed gain `0.01` and preserves the video mean to numerical precision
(`max error = 3.33e-16`).

| Arm | Macro within-video ROC | Delta versus T-only |
|---|---:|---:|
| frozen base | 0.632969 | -0.012161 |
| T-only | **0.645130** | 0 |
| raw AT | 0.646071 | +0.000941 |
| T + orthogonal audio | 0.641320 | -0.003809 |
| concordant AT | 0.636213 | -0.008917 |
| concordant shifted A | 0.635391 | -0.009739 |
| concordant shifted T | 0.637865 | -0.007265 |
| concordant shifted AT | 0.637580 | -0.007550 |

The paired bootstrap interval for raw AT minus T-only is
`[-0.00673, 0.00841]`; concordant AT minus T-only is
`[-0.02189, 0.00352]`.  Therefore conditional audio does not earn authority,
and the proposed audio-language joint mechanism is rejected.

### COHERE cohesion-discontinuity endpoints -- killed

`project_cohere_endpoint_decoder.py` selects left and right endpoints from the
fixed tight/current/broad lattice using the geometric rank consensus of a
visual inside/outside jump and a timestamp-language graph cut.  It changed 217
of 611 predictions, but interval F1 became `0.329810 / 0.286310 / 0.220549` at
IoU `0.3 / 0.5 / 0.7`, versus `0.330281 / 0.286613 / 0.247850` for the frozen
base.  The high-IoU regression is decisive.  Language cohesion currently
supports dense temporal ordering, not endpoint placement.

### Surviving claim after the follow-up

The surviving scientific object is a base-agnostic, propensity-preserving
semantic-core transport for dense within-video ranking.  On this independently
implemented ROUTE path, T-only improves macro within-video ROC by `+0.012161`;
the paired 95% interval is `[-0.000033, 0.024533]`.  This is consistent with the
earlier four-dataset and sealed-VASTA results, but the interval narrowly crosses
zero and therefore remains directional rather than conventionally confirmed.
