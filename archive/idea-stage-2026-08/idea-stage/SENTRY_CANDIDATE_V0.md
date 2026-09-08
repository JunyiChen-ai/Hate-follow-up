# SENTRY: Selective Null-Tested Endpoint Refinement

## Current decision

**Superseded after cached falsification.**  Fused orbit tails changed no metric,
and simple modality-knockout gates either changed no metric or reduced F1.  A
new core-conditioned cohesion candidate produces a positive four-dataset
within-video signal and is now preferred.

SENTRY replaces AMBI as the research direction.  The decisive observation is
that the dense multimodal field contains useful endpoint geometry, while the
MLLM same-event gate does not.  On the 278-video development subset, a
dense-only control with the same 115 endpoint actions reaches interval
F1@0.3/0.5/0.7 of `0.383343/0.308892/0.292700`, compared with
`0.381846/0.308892/0.254239` for AMBI.

The fixed top-115 rule is an oracle-free diagnostic but not a deployable method:
its action budget is inherited from AMBI and therefore must be replaced by a
video-conditional label-free warrant.

Preprocessing such as decoding, ASR, feature extraction, temporal alignment,
and cache construction is not a module.

## Module 1 -- Propensity-Orthogonal Multimodal Evidence Field

Separate video-level propensity from within-video residual evidence for visual,
acoustic, and linguistic views.  The existing dual-support transport supplies
the working dense field and conservative event core.  A reliable modality has
capped authority; a structurally missing modality abstains and an observed
contradiction is never relabeled as missing.

## Module 2 -- Video-Conditional Multimodal Endpoint Tests

For each fixed left/right contract-or-expand action, compare event core,
boundary shell, and adjacent outside evidence.  Construct length-matched,
block-preserving temporal reassignments independently for the available V/T/A
fields.  Every null member reruns the identical action maximization and
direction selection used by the factual field.

The action statistic is non-compensatory across available views: at least two
valid modalities must agree, no available modality may contradict, and the
full aligned statistic must exceed single-view and cross-modally misaligned
controls.  The resulting empirical tail is described as a randomization score,
not a conformal p-value or distribution-free certificate.

## Module 3 -- Selective Endpoint Action Controller

Left and right endpoints are controlled independently.  Execute at most one
fixed-lattice action per side only when:

1. its pipeline-level randomization score passes a frozen per-endpoint level;
2. the direction is stable across the preregistered temporal scales;
3. leave-one-reliable-modality-out evaluation preserves the action;
4. the edit retains the event core.

Otherwise fall back exactly on that side.  The controller changes interval
geometry but preserves the dense frame ranking.

## Paper thesis

> We formulate label-free multimodal hateful-video localization as selective
> endpoint testing: a boundary moves only when aligned multimodal residual
> evidence rejects a video-conditional no-action null.

The story is `field -> test -> action`, rather than `frame score -> threshold`.
MLLM endpoint authority is removed.  An MLLM may remain only if a future
matched-evidence ablation proves that its frozen dense feature is useful.

## Completed falsifications

1. Fixed-lattice MOSAIC lacked sufficient endpoint oracle headroom.
2. Full-video Visual Temporal Canvas did not scale.
3. AMBI same-event authorization lost to its equal-action dense control.
4. A fused-field circular-orbit tail at alpha `0.05/0.10/0.20` accepted only
   `9/19/35` endpoints and changed none of F1@0.3/0.5/0.7.  Fused-only orbit
   calibration is therefore killed; SENTRY must use load-bearing V/T/A action
   agreement rather than a more complicated fused-confidence threshold.

## Next decisive cached pilot

Use one frozen five-position endpoint lattice and the cached V/T/A fields.
Define the observed action statistic before reading ground truth as the weakest
within-modality aligned improvement among reliable views, requiring at least
two valid views and no contradiction.  Generate hash-seeded block-preserving
V/T/A shifts and rerun the full max-and-select pipeline in every null member.

Mandatory controls are:

- dense confidence with exactly the same action count and coordinates;
- best single modality and every modality pair;
- 2-of-3 agreement without randomization calibration;
- cross-modal temporal misalignment;
- scene-cut/change magnitude;
- count-matched random actions.

Development GO gate:

- act on 20--30% of eligible endpoints;
- correct moves at least twice incorrect moves;
- beat same-count dense confidence by at least `0.01` F1@0.7 and five
  net-correct endpoints;
- trail it by no more than `0.005` at F1@0.3 and F1@0.5;
- full V/T/A beats the best modality pair;
- breaking alignment reduces correct-action enrichment.

Only then freeze for untouched four-dataset confirmation.  The confirmatory
gate is delta F1@0.7 at least `+0.02` with paired CI lower bound above zero,
low-IoU non-inferiority, and positive direction on at least three datasets.

## Novelty status

- Selective fused editing only: at most `5--5.5/10`.
- Pipeline-level null testing with load-bearing V/T/A agreement and untouched
  confirmation: approximately `6--6.5/10` task-specific novelty.
- Current evidenced novelty remains below 6 and SOTA is not established.
