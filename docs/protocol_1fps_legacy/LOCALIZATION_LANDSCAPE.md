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
of this corpus.

## HateMM isolated-chunk diagnostic (2026-08-18, scripts 455f669)

Upstream span gold obtained (Zenodo 7799469; 85/86 test hate videos
usable; spans are COARSE — median 70% of audio covered, opposite regime
from HateClipSeg). Timestamped Whisper ASR reproduced the frozen
transcripts exactly (215/215). 2281 chunks over 212 videos, one
text-only isolated call each (prompts byte-identical to
`isolated_chunk_diag.py`):

| Contrast | AUC |
|---|---|
| Within-hate-video, pooled (span vs non-span chunks, same videos) | **0.624** |
| Within-hate-video, macro (54 videos with both classes) | 0.624 (sd 0.261) |
| Within-hate-video, strict (overlap ≥0.9 vs 0) | 0.674 |
| Cross-video: span chunks vs non-hate-video chunks | **0.720** |
| Video-level max-z over chunks (84 vs 128) | 0.901 |

Reading: isolated judgment is NOT dead on HateMM (HateClipSeg's 0.533
was partly a corpus-contrast artifact), but most of the strength is
video-level discrimination leaking through the chunk: the same scores
separate videos at 0.72–0.90 while separating moments within a video
at only 0.62. Saturation persists at chunk level (72.8% |z|>13);
length correlation ρ 0.364. Notable: the cross-video 0.720 lands on
LELA's reported HateMM frame AUC 0.726 — a single text-only pass over
isolated transcript chunks reaches the published multi-call number IF
that number is computed on the pooled (cross-video) contrast.

## LELA protocol verification (2026-08-18, full text read)

arXiv 2602.09637v1 = "Towards Training-free Multimodal Hate
Localisation with Large Language Models" (Sun et al.). Facts
established from the paper itself:

- **Metric population not stated** — no setup section, no split, no
  video count, no frame rate, no span-to-frame gold rule, and the
  promised appendix does not exist. The ONLY protocol statement:
  "follows the established practice introduced in [48]" = LAVAD (CVPR
  2024), whose convention is frame AUC POOLED over all test videos
  with normal videos contributing all-negative frames. Qualitative
  figures include non-hate videos. Verdict: pooled by inheritance;
  never stated explicitly.
- **Calls per frame: 12–16**, not ~5 (4 modality summarisation calls +
  4 × 3-stage prompting; the "5" is modalities). Backbone GPT-4o Mini.
- **Open-weight numbers (HateMM ROC-AUC):** DeepSeek-R1-7B 64.73,
  Qwen2.5-7B 62.14, LLaMA-2-7B 60.97, Qwen2.5-3B 57.49. GPT-4o Mini
  72.64, Gemini-2.0 Flash 70.28.
- Internal inconsistency: Table 1 vs Table 3 swap ROC-AUC/PR-AUC
  labels for the same two numbers (72.64/67.56); prose says 72.64 is
  ROC-AUC. No mAP/tIoU/segment metric anywhere. No within-video
  analysis anywhere. No code/data release.

Consequence for direction 3: the published comparison target is the
POOLED frame AUC. Our sequential isolated-chunk scores already reach
0.720 (span chunks vs non-hate-video chunks) on that contrast with ONE
text-only 8B call per chunk — level with the 12–16-calls-per-frame
GPT-4o-mini number and above every open 7B they report. The
within-video weakness (0.624) is reported honestly as the capability
boundary — the benchmark's operative metric does not measure it, and
no published system demonstrates it either.

## Direction 3 status (2026-08-18): SURVIVED its pilot

Novelty check NOVEL-with-caveats (IDEA_REPORT.md f0deb1f: mask
primitive conceded to SingGuard/InvariRank/T3S; contamination finding
+ isolation remedy + localization output unclaimed). Prereg 7928b9b;
pilot script aa403cb; note MASKED_PARALLEL_ISOLATION_NOTE.md. Both
frozen clauses passed: fidelity Spearman 0.9989 (packed masked forward
≡ N isolated calls, 3.9× faster, prefix once); counterfactual gap
0.0575 ≥ 0.05 (unmasking collapses within-video macro 0.620 → 0.562
while INFLATING cross-video 0.720 → 0.850 — the global-verdict smear
mechanism showing both faces). First surviving localization direction.

Endpoint 1 (frame-level, prereg febf147, note FRAME_LEVEL_EVAL_NOTE.md):
**PASS** — pooled frame ROC-AUC 0.7451 (covered-only 0.7957) vs bar
0.65; above every number LELA reports (GPT-4o Mini 72.64 at 12–16
calls/frame, best open 7B 64.73). Honesty findings recorded in the
note: the pooled metric is mask-insensitive (causal 0.7461) because
cross-video inflation cancels within-video collapse — the mask's value
is score semantics + within-video discrimination + call-equivalence;
within-video stays the measured boundary (frame macro 0.571).

## MHC temporal gold + competitor metric verification (2026-08-18)

1. **MultiHateClip temporal annotations EXIST publicly** — GitHub
   Social-AI-Studio/MultiHateClip, `{English,Chinese}_data/annotation/
   {train,valid,test}.tsv`, column `Duration` = list of (start_sec,
   end_sec) tuples. Verified against the paper's Table 2/3 mean span
   lengths (match to rounding). Our local copy is a third-party
   repackaging that DROPPED this column; join is clean (all 891 EN +
   897 ZH local Video_IDs present upstream). Span coverage on our
   subset: EN Hateful 69/72, Offensive 193/218; ZH Hateful 102/112,
   Offensive 157/180; plus 29 EN / 40 ZH Normal-majority videos with
   leftover spans (consolidation artifact — handling must be frozen
   before use) and some H/O videos with no span at all.
2. **MHC localization is near-degenerate:** videos capped at 60 s;
   marked spans average 89–92% of video length; 81% of ZH spans start
   at t=0. Frame-level evaluation there mostly re-measures video-level
   discrimination. Weaker temporal gold than HateMM. LELA evaluated
   MHC-English only.
3. **MultiHateLoc's "mAP 0.645" is FRAME-level mAP** (per-frame
   precision-recall area), NOT segment mAP@tIoU — the paper has no
   tIoU grid, no proposal matching. Its metric family = LELA's
   (frame ROC-AUC / frame AP). Numbers: HateMM mAP 0.645, AUC 0.799;
   MHC mAP 0.445, AUC 0.750 — but it is TRAINED on video-level labels
   (weakly supervised MIL), a different supervision regime from our
   zero-label setting; split/frame-rate unstated. Consequence: no
   interval extraction is needed to compare with ANY published hate
   localization number; interval extraction remains only for
   HateClipSeg Task-2 tIoU-F1 (ActionFormer 31) if we go there.
4. LELA reliability flag: duplicated/mangled bibliography entries for
   both datasets; MultiHateLoc misdescribes MHC (claims even class
   split and an "audience" field it lacks). Both papers' dataset
   sections are sloppy; every protocol detail we use must be
   self-contained.
