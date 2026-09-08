# Pre-registration — Temporal attribution pilot (single-pass evidence localization, HateClipSeg)

**Frozen:** 2026-08-12, before any attribution value is computed.
**Compute:** one RTX 5090; one forward (+ one backward) per video over ≤394
HateClipSeg videos; well under 1 GPU-hour. CPU analysis.
**Status:** signal-existence pilot for the localization method family
("the judge's video-level hate decision, attributed back onto its input
tokens, localizes the hateful segments in time — zero labels, zero
training, zero extra calls"). Novelty check running in parallel; if it
returns NOT NOVEL the family is dropped regardless of this pilot.

## Phenomenon

Hateful signal is temporally sparse: in HateClipSeg, 363 of 435 annotated
videos contain both offensive-labeled and normal-labeled segments (mean
segment 8.84 s). A video-level judge must be driven by the offending
spans; where its decision-relevant attention/gradient mass falls is
therefore a candidate label-free localizer. Prior evidence on this exact
backbone (Qwen3-VL family): prefill attention of a sparse set of heads
localizes queried moments before any token is generated (2605.21954).

## Frozen protocol

- Corpus: all HateClipSeg videos that are (a) locally complete (394),
  (b) sparse — at least one offensive-union segment AND one normal-only
  segment in `segment_level_annotation.csv`, and (c) ASR-timestamp
  routed (`route == "timestamped"` in the interleaved-timeline census).
  The resulting set (expected ≈300–330; exact count reported) is fixed
  by these three conditions — no further selection.
- One forward pass per video with the EXACT frozen judge prompt (16
  frames, title, full c2 transcript, union rules block, Yes/No
  question), eager attention, capturing: (i) last-prompt-position
  attention row to all input tokens, every layer and head; (ii) one
  backward pass of z = logsumexp(Yes) − logsumexp(No) onto the input
  embeddings.
- Attribution variants, both computed, named in advance:
  - **A1 (primary): gradient×input** — per input token, |∇_e z · e|,
    parameter-free.
  - **A2 (secondary): late-layer attention** — mean over the last third
    of layers and all heads of the last-position attention row,
    parameter-light (no head search of any kind).
- Time mapping: transcript tokens inherit the [start,end] of the ASR
  chunk they belong to (chunk-to-prompt alignment by exact text
  matching; videos where alignment fails are reported and excluded, not
  patched); frame tokens inherit the frame's nominal timestamp
  ((i+0.5)/16 × duration); title and rules tokens are excluded from the
  curve. Per-token weights are summed into per-second bins, then
  averaged within each gold segment's time span.
- **Statistic:** per video, AUC of segment-mean attribution for
  offensive-union segments vs normal-only segments; macro-averaged over
  videos (equal weight per video). Reported for A1 and A2, plus for the
  hateful-strict collapse (descriptive).
- **Control (mandatory):** token-density baseline — identical pipeline
  with all attribution weights set to 1 (measures "more speech happens
  in offensive spans" and tokenization artifacts). Same AUC statistic.
- Descriptive extras: transcript-channel-only vs frame-channel-only
  curves; per-video AUC distribution; correlation of per-video AUC with
  video-level z; positional-artifact check (mean attribution by prompt
  position quartile on 20 randomly chosen videos).

## Frozen decision rule

The family **SURVIVES** only if, for at least one of A1/A2:

1. Macro AUC (offensive vs normal segments, union collapse) ≥ 0.65, and
2. Macro AUC ≥ token-density control + 0.05.

**DIES** if both variants fail either clause. No post-hoc variant, layer
subset, or head selection may be added to rescue the pilot; those are
method-stage refinements permitted only after survival.

## Interpretation boundaries

- Survive: licenses the full method preregistration — interval
  extraction, HateClipSeg Task-2 tIoU F1 vs the ActionFormer baseline
  (F1@0.5 = 31) and LELA-style frame-level AUC comparison, HateMM span
  evaluation (requires downloading the benchmark's own span
  annotations — evaluation-side gold, charter-legal), efficiency
  comparison (1 call vs LELA's ~5 per frame). This pilot itself claims
  only signal existence.
- Dies with control ≈ attribution: the curve is token-density in
  disguise; record and drop.
- Dies with both AUCs < 0.65: the judge's decision mass does not land
  on the annotated segments — the evidence-localization premise is
  false for this judge; the family dies and the goal loop re-enters
  idea discovery.
- Labels (segment gold) are used only in evaluation; no label touches
  the attribution computation. Single call per video at deployment; the
  backward pass is part of the same call's computation, not a second
  MLLM query.
