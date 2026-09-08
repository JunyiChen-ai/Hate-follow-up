# Experiment Plan: MELT

## E0 — Interface Sanity (MUST-RUN)

- Cohort: deterministic clean Stage-A, two videos per dataset.
- Systems: scalar canvas, phase-only, lifecycle decoder, full MELT.
- Verify structured output coverage, non-constant phase evidence, all-OUT
  availability, legal paths, intervention direction, call counts, and complete
  provenance.
- Prediction code must not load ground truth.

Kill if fewer than half of videos yield a parsable relation and non-constant
phase evidence, or if the certifier cannot produce any finite paired effect.

## E1 — Four-Dataset Pilot (MUST-RUN)

- Cohort: deterministic clean Stage-B, eight videos per dataset.
- Compare T3AL, current P2 scalar canvas, phase-only, phase+lifecycle, full
  MELT, shifted-transcript MELT, and compute-matched scalar self-refinement.
- Metrics: interval F1@0.3/0.5/0.7, temporal mAP where supported, start/end
  normalized MAE, within-video ROC/PR, pooled ROC/PR, certificate coverage,
  all-constant rate, fallback rate, and actual MLLM calls.

Promotion requires macro F1@0.3 above P2, positive direction on at least three
datasets, within-video ROC no more than 0.02 below P2, full MELT beating at
least one of phase-only or uncertified lifecycle on a boundary metric, and
aligned transcript beating timestamp shift.

## E2 — Mechanism Audit (MUST-RUN after E1 promotion)

- T3AL proposal shift/expand/shrink/delete.
- visual-only, transcript-only, aligned joint, timestamp-shifted joint.
- cited-evidence removal versus equal-area/equal-modality matched control.
- quotation/counterspeech, asserted hate, OCR-led, and visually grounded slices.

## E3 — Resolution and Robustness (after E2)

- 8/16/32 coarse bins and native 4 FPS boundary candidates.
- short/long videos, single/multiple intervals, transcript dropout and jitter.
- frozen thresholds selected without target-test labels; paired bootstrap CIs.

## Run Order

E0 sanity -> code review -> E1 pilot -> mechanism gate -> E2 -> E3.

## Budget

E0 is capped at 32 MLLM calls. E1 is launched only after E0 passes. Any retry
must be recorded; invalid output fails closed rather than silently reverting to
ground-truth-informed behavior.
