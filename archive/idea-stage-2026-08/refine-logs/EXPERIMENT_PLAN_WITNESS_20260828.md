# WITNESS mechanism pilot (preregistered before GT evaluation)

## Claim under test

Label-free hateful-video localization can be formulated as a fixed-budget
search for the smallest evaluated multimodal window whose evidence remains
positive after adding real temporal visual context.  Context exposure is used
both to select where to zoom and to define a resolution-limited witness; it is
not used as a post-hoc score fusion rule.

## Modules (preprocessing is explicitly excluded)

1. **Temporal Firewall Scorer.** An interval is scored from frames inside the
   interval plus timestamp-aligned speech from that interval.  The output is
   the frozen MLLM raw Yes-minus-No margin.  There is no gray/null subtraction,
   video normalization, dataset calibration, or ground-truth access.
2. **Paired Real-Context Intervention.** Each interval is queried twice with
   the same prompt, focus frames, focus transcript, carrier size, and number of
   visual cells.  The narrow carrier fills four context cells with additional
   frames from inside the interval; the exposed carrier replaces only those
   cells with real frames immediately outside the interval.  This avoids the
   out-of-distribution gray-null intervention found in TIDE U32.
3. **Fixed-Budget Quadtree Zoom.** Eight coarse intervals are paired-scored.
   Four intervals are selected by a frozen lexicographic rule: sign-discordant
   first, paired-positive second, then absolute exposure delta and raw margin.
   Each selected interval is split into four children and every child is
   paired-scored.  Total budget is 48 semantic branches per video.
4. **Minimal Invariant Witness Extraction.** At the evaluated resolution, a
   leaf is a witness iff both its isolated and context-exposed margins are
   positive.  A coarse leaf is retained only when it was not selected for
   refinement.  Adjacent witnesses merge only when they share an exact boundary;
   there is no dilation, closing, duration prior, Potts/TV objective, or
   threshold sweep.  “Minimal” is claimed only relative to the evaluated tree
   resolution.

Frame decoding, frame sampling, ASR extraction/alignment, and canvas rendering
are preprocessing and are not counted as modules.

## Frozen inference details

- Model: `Qwen/Qwen3-VL-8B-Instruct`, local frozen checkpoint.
- Policy definition: identical to TIDE U32.
- Coarse partition: 8 equal-duration half-open intervals.
- Refined parents: exactly 4; children per selected parent: exactly 4.
- Per query: 8 real visual cells, with 4 focus cells identical between the
  paired arms; focus transcript is identical and capped before arm derivation.
- Decision threshold: raw Yes-minus-No margin `> 0`.
- Output grid: canonical 4 FPS.
- No video label or temporal annotation is opened by the inference runner.

## Equal-budget controls

1. `uniform24_pair`: 24 uniform intervals, each paired-scored (48 branches).
2. `hash_zoom_pair`: four coarse parents selected by a dataset/video hash.
3. `score_zoom_pair`: four parents selected only by isolated raw margin.
4. `witness_no_extraction`: same WITNESS queries, but merge isolated-positive
   leaves without the paired-positive witness requirement.
5. `witness_no_zoom`: coarse paired-positive leaves only.

## Falsification and multimodal controls

- text-only with the exact same tree and decision rule;
- visual-only with the exact same tree and decision rule;
- temporal frame shuffle while leaving aligned speech unchanged;
- within-video circular ASR shift while leaving frames unchanged;
- isolation-only matched input control where only cross-branch visibility is
  changed, if the attention implementation supports it without changing tokens.

## Primary gates

- Primary: macro interval F1@0.5 across the four datasets.
- Boundary: macro interval F1@0.7.
- Localization sanity: within-video macro ROC-AUC.
- WITNESS must beat both equal-budget `uniform24_pair` and `hash_zoom_pair` at
  F1@0.5, and removal of Module 4 must reduce F1@0.5 or F1@0.7.
- Gains must be positive on at least three of four datasets.
- The aligned joint arm must beat text-only on at least two datasets, and frame
  shuffle must reduce witness persistence and localization performance.
- Empty/full-video prediction rates and calls/video are always reported.

Failure of these gates kills the WITNESS paradigm rather than triggering
test-label tuning.  Passing them supports a provisional task-specific novelty
assessment of 6--6.5/10; it does not by itself prove SOTA.
