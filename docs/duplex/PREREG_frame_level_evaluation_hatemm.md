# Pre-registration — Frame-level evaluation on HateMM (direction 3, method stage, endpoint 1)

**Frozen:** 2026-08-18, after the pilot verdict (MASKED_PARALLEL_
ISOLATION_NOTE.md) and BEFORE any frame-level AUC is computed. CPU
only; deterministic transformation of scores already on disk
(`results/masked_parallel_isolation/per_chunk.jsonl`). No new model
calls.

## Endpoint

The published comparison target (LELA, arXiv 2602.09637) is
frame-level ROC-AUC on HateMM under the LAVAD convention: frames of
ALL test videos pooled, non-hate videos contributing all-negative
frames. LELA's numbers: GPT-4o Mini 72.64 (12–16 calls per frame),
Gemini-2.0 Flash 70.28, best open 7B (DeepSeek-R1-7B) 64.73,
Qwen2.5-7B 62.14. (Caveat recorded: LELA's Tables 1/3 swap
ROC/PR labels; its prose says 72.64 is ROC-AUC. Its split and frame
rate are unstated — our protocol below is therefore self-contained and
fully specified, and the comparison is to their reported numbers, not
a reproduction of their unpublished protocol.)

## Frozen protocol

- Cohort: the 212 pilot videos (84 hate / 128 non-hate; 3 exclusions
  documented in the diagnostic). Scores: the locator call's per-chunk
  z — primary = z_masked; z_reference (sequential) reported as a
  sensitivity row (expected ≈ identical, Spearman 0.9989).
- Frame grid: 1 frame per second over [0, wav_duration) per video.
- Frame score: the z of the chunk whose [start, end) contains the
  frame's timestamp. Chunks are non-overlapping Whisper segments.
- Uncovered frames (no chunk spans them — silence/music): score =
  (corpus-wide min chunk z) − 1. A transcript-only locator has no
  evidence there; the floor is the honest assignment. Sensitivity row:
  covered-frames-only AUC.
- Gold: frame positive iff its timestamp lies inside a hate span
  (span_gold.json). Frames of hate videos outside spans are NEGATIVE;
  all frames of non-hate videos are negative.
- Statistics: pooled frame ROC-AUC (primary), pooled PR-AUC (AP,
  secondary), per the LAVAD convention. Honesty section (mandatory):
  within-hate-video frame-level macro AUC — the number the pooled
  metric hides; reported alongside, not averaged away.

## Frozen decision rule

Endpoint 1 **PASSES** if pooled frame ROC-AUC (primary row) ≥ **0.65**
— above every open-weight model LELA reports. The comparison against
GPT-4o Mini's 72.64 is descriptive (their protocol is unstated; no
equivalence claim). **FAILS** below 0.65: the direction's headline
claim ("open 8B, one locator forward, competitive frame-level
localization") is not supported; the direction returns to the owner
with the pilot's mechanism result intact but no benchmark claim.

## Boundaries

- No tuning of the frame rate, floor value, or coverage rule is
  permitted after seeing numbers; sensitivity rows are fixed above.
- Span gold is evaluation-only. Deployment stays 2 calls per video
  (judge + locator), text-only locator; the multimodal-prefix
  variant and other corpora are separate future preregs.
