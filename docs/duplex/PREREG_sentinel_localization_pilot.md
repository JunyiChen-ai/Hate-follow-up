# Pre-registration — Sentinel logit-lens localization pilot (direction 2, HateClipSeg)

**Frozen:** 2026-08-12, before any sentinel margin is computed. Novelty
check running in parallel; a NOT-NOVEL verdict voids the direction
regardless of pilot outcome.
**Compute:** one RTX 5090; ONE new forward pass per video (the
localization call — second of the ≤2 budget, role "locator"; the frozen
detection call is untouched); ≤394 videos, well under 1 GPU-hour.
**Status:** signal-existence pilot for direction 2 of the localization
goal loop. Direction 1 (attribution of the detection verdict) died
(TEMPORAL_ATTRIBUTION_PILOT_NOTE.md): the global verdict's mass is not
temporally concentrated. Direction 2 differs at the root: it does not
attribute the global verdict — it ELICITS A LOCAL JUDGMENT at every
point of the timeline inside one pass and reads each judgment from the
model's own answer direction.

## Mechanism

The locator call lays out the video's evidence in temporal order: 16
frames, title, then the timestamped ASR chunks one by one, each
followed by a fixed sentinel marker. The instruction (before the
content) states that at each marker the model is to judge whether THE
SEGMENT JUST SHOWN violates the policy (same union rules block as the
frozen judge). No text is generated: at each sentinel's final token
position, late-layer hidden states are projected through the final norm
and unembedding onto the same Yes/No token sets as the frozen judge —
a per-chunk margin m_k = logit(Yes) − logit(No), read at the position
where the model's running computation has just consumed chunk k.
Diagnostic arm B (same videos, separate single pass): all content
first, then one short numbered question per chunk ("Does segment k
violate the rules? Answer:"), reading the margin at each answer
position — explicit local questions with full context. Arm B is a
capability ceiling only; its method-level novelty is weak (packed
LELA) and it is not the claim.

## Frozen protocol

- Cohort: HateClipSeg videos that are locally complete, sparse (≥1
  offensive-union segment AND ≥1 normal-only segment in the segment
  gold), and have timestamped chunks with usable spans (chunk end
  times; a missing final end time is replaced by wav_duration — this
  touches no model input). No alignment to the frozen collapsed
  transcript is needed (the locator presents the chunks themselves);
  the direction-1 exclusion therefore does not recur. Exact count
  reported.
- Locator prompt (frozen): union rules block (verbatim from the frozen
  judge), locator instruction (fixed wording, written in the
  implementation before any scoring and hashed in the report), 16
  frames, title, then chunks in order, each as
  "[k | start–end s] <chunk text> ⟦ASSESS⟧". Sentinel string fixed:
  "⟦ASSESS⟧". One forward, no generation, bf16, batch 1.
- Readout: margin at each sentinel position, computed per layer;
  frozen aggregate = median of the last 4 layers' margins (final-layer
  margin reported descriptively).
- Curve and statistic (identical machinery to direction 1's pilot, for
  comparability): chunk margin spread uniformly over the chunk's
  [start,end]; per-second bins; gold-segment mean; per-video AUC of
  offensive-union vs normal-only segments; macro-average over videos.
  Hateful-strict collapse descriptive.
- Controls: (i) token-density control (chunk weight = chunk token
  count); (ii) position drift check — Spearman(margin, chunk index)
  distribution across videos (an artifact fingerprint, reported).
- Arm B: same curve machinery from its answer-position margins.

## Frozen decision rule

Direction 2 **SURVIVES** only if Arm A (sentinel):

1. Macro AUC ≥ 0.65 (union collapse), and
2. Macro AUC ≥ token-density control + 0.05.

Interpretation grid, fixed in advance:

- A passes → licenses the full method preregistration (interval
  extraction, HateClipSeg Task-2 tIoU F1 vs ActionFormer 31, LELA-style
  frame AUC comparison, efficiency claim).
- A fails but Arm B reaches ≥ 0.65 → the model CAN judge locally but
  the sentinel readout misses it; the family survives only as Arm B,
  which carries a novelty penalty (packed per-chunk querying ≈
  compressed LELA) — recorded and taken to the owner; not promoted
  unilaterally.
- Both A and B < 0.60 → Qwen3-VL-8B lacks usable local judgment on
  this corpus under zero labels; the elicited-local-judgment family
  dies, and the goal loop re-enters idea discovery.
- Segment gold is evaluation-only; no label touches any computation.
  Deployment shape: detection call + locator call = 2 calls, distinct
  roles, within budget.
