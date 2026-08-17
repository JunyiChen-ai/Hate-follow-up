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
