# Frozen preregistration: VASTA on unseen HateClipSeg p11 train+val

Frozen before any VASTA prediction or frame-level evaluation is produced on
this cohort.

## Cohort

- Source split: `data/gt/HateClipSeg/p11_split.json` from Retrieval-hate.
- Evaluation IDs: the union of the frozen `train` and `val` ID lists.
- Exclusion: any ID present in p11 `test`, any undecodable/missing media, and
  any duplicate ID.
- This cohort has never appeared in VASTA's 611-video confirmation set or its
  32-video mechanism pilot. Other projects may have processed these videos for
  video-level detection; no such outputs are inputs to VASTA.
- Dataset name in prediction schema: `HateClipSeg_sealed`.

## Frozen method

No parameter, prompt, rule, or parser may change after the first prediction.

1. Frozen A10/Vid-Group proposal bank, generic query already used in v3.
2. Module 1: connected top-8 proposal-support component containing the rank-1
   center, at 4 FPS.
3. Module 2: A08/VideoTGB and A12/TimeLens visual authority states. Anchored
   first Yes/No parsing is used for A08.
4. Module 3: Qwen3-VL-8B-Instruct transcript logit, deterministic batch size 1,
   left padding, one explicit empty-transcript query. In visual disagreement,
   veto iff `q_T(x) < q_T(empty)`. Text cannot create or move intervals.

Frozen implementation/report version: VASTA v3 as recorded in
`docs/idea_discovery/FINAL_METHOD_VASTA_20260828.md`.

## Gold and evaluation

- Gold conversion is isolated from inference and uses the published
  HateClipSeg exhaustive segment annotations.
- Primary positive rule: offensive union, identical to the existing p11-test
  HateClipSeg 4-FPS array: any non-normal dimension is positive.
- Report pooled frame ROC-AUC, PR-AUC, within-video macro ROC-AUC, and interval
  F1 at tIoU .3/.5/.7.
- Report A10 rank-1, Module-1 extent, visual OR, and full VASTA on exactly the
  same evaluable IDs.

## Frozen acceptance test

The sealed result supports the selective-authority mechanism if full VASTA:

1. improves visual OR at interval F1@.5;
2. does not reduce visual OR by more than .01 at F1@.3;
3. records positive net veto correction at tIoU .5; and
4. retains a positive Module-1 gain over A10 rank-1 at F1@.5.

No prompt or threshold adjustment is allowed if a condition fails. A failure is
reported as a failure, not used to revise and rerun this cohort.

