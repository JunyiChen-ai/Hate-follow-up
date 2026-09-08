# MELT: Minimal-Entailment Lifecycle Tracking

## Problem Anchor

Existing label-free hateful-video localizers can rank frames but do not define
what makes an onset or offset semantically valid. The current numbered-canvas
pilot improves within-video ordering, yet its thresholded intervals remain
weak. MELT changes the prediction object from an independent frame score to a
structured hateful-event lifecycle with executable boundary evidence.

## Method Thesis

Given a coarse label-free visual anchor, a frozen MLLM parses a structured
event relation and its temporal phases. A boundary is accepted only when the
event changes state across the cut and the cited evidence passes a paired
counterfactual test against a matched control. Ranking and intervals are two
readouts of the same certified lifecycle.

## Method Modules

Data decoding, frame sampling, ASR loading, timestamp alignment, and canvas
rendering are preprocessing and are explicitly not counted as modules.

### M1. Multimodal Event-Hypothesis Induction

Parse `R = (source, hostile_act, protected_target, stance)` and cite the numbered
time bins that materially support the hypothesis. The relation is allowed to
contain unknown fields; the model must not invent an entity merely to satisfy
the schema.

### M2. Evidence-Grounded Multimodal Phase Field

For each visually focused numbered bin, obtain forced-choice token logits over
`OUT / ENTER / SUPPORT / EXIT / UNKNOWN`, conditioned on the fixed relation,
the local timestamped speech, and global visual context. This avoids generated
score calibration and makes the temporal state an actual model probability.

### M3. Resolution- and Stance-Aware Lifecycle Decoder

Decode connected event-phase components against the stronger of OUT and UNKNOWN,
retain only components intersecting M1's cited evidence envelope, and preserve
an all-OUT path. The resulting state field yields dense ranking and its connected
components yield intervals; the original T3AL curve is never thresholded.

The lifecycle intervals always come from this decoder. Dense ranking is routed
without target labels: phase-field ranking is used only for asserted/ambiguous
relations when one coarse bin spans at most four seconds; otherwise the original
label-free dense prior is retained. This prevents a coarse 16-bin canvas from
dominating long videos while preserving its semantic correction on sufficiently
resolved events.

Counterfactual cited-removal, matched-control, cited-only and timestamp-shift
tests are retained as a mechanism audit, not counted as a method module. E0
produced zero certified refinements, so the certifier was removed from the main
method rather than retained for narrative value.

## Core Claims

1. Label-free hateful-video localization is better represented as certified
   event-lifecycle parsing than independent frame relevance scoring.
2. Evidence-linked lifecycle decoding converts the numbered-canvas ranking gain into temporal
   boundary gain.
3. Correct audiovisual-transcript alignment contributes beyond either modality
   marginal; this is tested through aligned-versus-shifted paired interventions.

## Scope and Integrity

- T3AL/Poset supplies anchors only and is replaceable.
- No target labels are available to prediction code or prompt selection.
- Free-form rationales are not certificates; a certificate requires a measured
  response to an executable intervention and a matched control.
- Counterfactual certification is not part of the method claim unless a later
  held-out experiment demonstrates nonzero certified coverage and incremental
  boundary value.
