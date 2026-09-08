# Frozen confirmation: Relation-Envelope Localizer

This document was written before selecting or evaluating the confirmation
cohort. The cohort is selected without GT by SHA256 order from decode-safe test
videos outside `full_curve_eligible_manifest.jsonl`, with up to 16 videos per
dataset and salt `relation-envelope-confirm-v1`. Availability inspection before
selection showed that full288 already contains 212/215 HateMM test videos, so
all three remaining HateMM videos are retained and the other datasets retain 16
each (51 total). HateMM is explicitly underpowered in this confirmation.

## Frozen method

1. Infer one global directed relation `(source, hostile_act, protected_target,
   stance)` from a 16-bin visual canvas and aligned transcript.
2. Given that fixed relation, classify every bin with one compute-matched field:
   `FULL / NOT_FULL / UNKNOWN`. Only endorsement or ambiguous stance can produce
   an event.
3. Convert positive connected components to intervals and expand by exactly one
   bin on each side. This radius equals the frozen local transcript receptive
   field (target bin ±1), not a tuned boundary hyperparameter. Merge overlapping
   expanded intervals.

No T3AL curve, annotation, dataset-specific routing, typed negative states,
generic fallback, modality responsibility fusion, citation, lifecycle, or
counterfactual certificate is used. Decode/parser failures are all-zero,
empty-interval predictions and remain in evaluation.

## Frozen comparison and metrics

- Same-cohort A10 VidGroup and A12 TimeLens predictions.
- Per-dataset event-micro interval F1 at tIoU 0.3/0.5/0.7.
- Four-dataset unweighted mean and event-pooled F1 are both reported.
- The confirmation passes only if F1@0.5 improves over both baselines in either
  event-pooled performance or at least three datasets; exploratory old cohorts
  cannot be used to rescue a failed confirmation.

This is a confirmation of the fixed mechanism, not a new tuning cohort.
