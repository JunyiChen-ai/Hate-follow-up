# MELT E0 Deployment Review

## Initial verdict

Deployment was blocked. The first implementation described counterfactual
conditions only in text, accepted uncertified boundaries, used an unsafe
OUT-only null path, supported one interval, mapped seconds with an implicit
4-FPS assumption, trusted arbitrary cohort fields, and hard-coded call counts.

## Fixes landed before E0

- Phase evidence now comes from batched forced-choice token logits, not generated
  numeric JSON. A red visual focus box binds each classification to one bin and
  only local timestamped speech is exposed to that query.
- Counterfactual conditions now render distinct visual inputs and construct
  distinct transcript inputs for full, trim, expansion, cited removal,
  cited-only, matched-control removal, and timestamp shift.
- Failed certificates explicitly fall back to the lifecycle interval and are
  recorded as `certified=false`; refined boundaries are never mislabeled.
- The no-event baseline uses the stronger of OUT and UNKNOWN. Lifecycle paths
  are cited connected components and can represent multiple disjoint events.
- Seconds are mapped to the actual dense curve length through
  `time / duration * len(curve)`.
- Cohort rows use a strict inference-only allowlist; relation fields and stance
  are validated; endorsed/ambiguous relations require cited evidence.
- Calls use measured per-video model-forward deltas.
- Tests cover UNKNOWN-dominant all-OUT behavior and rejection of GT-bearing
  cohort fields, in addition to schema, lifecycle, and certificate arithmetic.

## Residual limitation

The counterfactual certifier uses the same frozen MLLM as the parser, so its
evidence is paired and executable but not model-independent. The pilot must
therefore demonstrate incremental boundary value over lifecycle-only; otherwise
M3 is rejected rather than retained for narrative value.
