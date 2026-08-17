# Landscape — label-free temporal localization of hateful video content (2026-08-12)

Two parallel scans (domain + methodology) plus a formal novelty check.
This file records the verified grid; details and sources in the scan
outputs (session 2026-08-12) and the notes cited below.

## Occupancy grid (hate-video temporal localization)

| | Zero labels | Video-level labels | Segment-level labels |
|---|---|---|---|
| ≤2 model calls | **EMPTY — our target cell** | empty for hate (HiProbe-VAD occupies it for anomaly, with a label-trained probe) | empty |
| Many calls (per-frame/per-window) | LELA (2602.09637: ~5 GPT calls PER FRAME, HateMM frame AUC 72.6 on GPT-4o-mini; open 7B "substantially worse"; call count unreported in paper) | VERA (anomaly only) | — |
| No MLLM calls | — | MultiHateLoc (WWW'26, 2512.10408: MIL over ViT/VGGish/BERT features, HateMM mAP 0.645) | ActionFormer on HateClipSeg (official baseline, F1@0.5 = 31) |
| Trained / RL | — | TANDEM (ICWSM'27: cross-modal RL, target-ID F1 0.73) | SafeLens AAAI-26 demo (LoRA on HateClipSeg) |

Also: RAMF 2512.02743 (trained fusion, caption front-end); WWW'26
agentic "evidence attribution" paper (10.1145/3774905.3796488 —
tool-output bookkeeping, multi-call; full text paywalled, must be read
before any submission); MARS/ARCADE (video-level only). Motivation
paper for the whole line: 2508.04900 (temporal label noise in hateful
video classification).

## Mechanism toolbox (from the methodology scan)

- TG-Heads 2605.21954: sparse attention heads localize grounding
  queries at PREFILL on Qwen3-VL-8B; decode drifts; selection criterion
  unstated.
- DecAF 2510.19592 (ICLR'26): training-free head selection + rollout,
  contrastive background subtraction.
- Chefer GAE 2103.15679: gradient×attention, answer-specific, one
  backward.
- NumPro 2411.10332: frame numbers burned into pixels → model
  verbalizes timestamps, training-free.
- OmniTrace 2604.13073: attention/gradient attribution in omni-modal
  LLMs, time-grounded spans — but explains GENERATED TEXT provenance,
  not decisions; no moderation use.
- ViToSA 2506.00636: text-cascade toxic-span + ASR timestamps (separate
  models, not internal attribution).
- Known attribution failure modes: attention sinks; last-frame
  positional bias (up to 86.8% of top mass); text-token dominance
  (frames near-uniform under Shapley); prefill→decode drift.

## Status of our directions

1. **Direction 1 — attribution of the frozen detection logit**
   (grad×input; uniform late-layer attention): novelty check
   NOVEL-with-caveats (OmniTrace owns the machinery for generation;
   we'd own decision-evidence use). Pilot preregistered (26ccde5) and
   **DEAD** (8a17ae4): attribution ≈ token-density control (0.525 vs
   0.518), attention sink-dominated — the judge's video-level decision
   integrates globally; its mass does not concentrate on annotated
   offensive segments. See TEMPORAL_ATTRIBUTION_PILOT_NOTE.md.
2. Direction 2 — under generation (idea-discovery round in progress).

## Evaluation assets (local)

HateClipSeg segment gold (11,714 segments, 435 videos; 363 sparse; 328
local; 316 timestamped-routed; 159 with chunk-text exactly matching the
frozen collapsed transcript). Chunk-level timestamped ASR for 1105
videos (no HateMM). HateMM upstream span annotations NOT yet
downloaded (evaluation-side gold, charter-legal to fetch).

## Capability map on Qwen3-VL-8B / HateClipSeg (descriptive diagnostics, 2026-08-12)

All label-free access modes to WITHIN-VIDEO local judgment, measured
(80-video sample unless noted; segment gold used for evaluation only):

| Access mode | Statistic | Result |
|---|---|---|
| Attribution of the global verdict (direction 1, preregistered) | segment AUC | 0.525 ≈ token-density control — DEAD |
| In-context local elicitation (direction 2, preregistered; sentinel + packed questions) | segment AUC | 0.499 / 0.519, both ≈ chance; model stamps every segment with the global verdict — DEAD |
| Isolated text chunk (median 9–12 tokens), one call each | chunk AUC | 0.533 pooled / 0.571 macro; score tracks length (ρ 0.50) |
| Isolated text window (45–270 s) | window AUC | 0.690 pooled — but max-pooling the chunk scores gives 0.695: coarsening the unit, no new capability |
| Fixed-window granularity curve (max-pooled chunks) | window AUC | 15 s 0.601 · 30 s 0.617 · 60 s 0.637 · 90 s 0.675 |
| Increment of sentinel margins (exploratory reanalysis) | segment AUC | 0.499 — chance |
| Isolated single frame, one call each | frame AUC | 0.584 pooled but **0.542 macro** (within-video); frame-index-only baseline 0.537 |

Reading: the judge holds ONE global, evidence-driven verdict per video;
no access mode yields a usable per-unit local verdict on either channel
of this corpus. Open question in flight: does isolated-chunk judgment
work on HateMM (higher-contrast hate spans; LELA's own benchmark, its
multi-call ceiling frame AUC 0.726) — upstream span gold + timestamped
ASR + diagnostic chain running. If yes → direction 3 = single-call
parallel isolation via block-diagonal attention masking (mechanism
motivated by the measured contamination), evaluated head-to-head with
LELA. If no → the goal (label-free localization, ≤8B, novel + works)
has a complete measured impossibility chain for owner review.
