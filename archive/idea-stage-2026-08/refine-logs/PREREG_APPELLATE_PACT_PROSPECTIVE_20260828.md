# Prospective preregistration: Appellate-PACT

Frozen before opening any temporal annotations for the selected cohorts.

## Cohorts

- Every ID in `mhclip_en_train.txt` (630 videos), named
  `MHC_en_train_prospective`.
- Every ID in `mhclip_zh_train.txt` (657 videos), named
  `MHC_zh_train_prospective`.
- The manifest builder may read only split IDs, timestamped ASR, durations,
  and media paths. It may not read class labels or temporal annotations.
- All predictions and controls must be content-hashed before temporal GT is
  constructed or inspected.

## Frozen method

The base interval is the connected union-support component of the NMS-diverse
Vid-Group top-8 proposal bank that contains the rank-1 proposal centre.

An alignment-specific transcript scorer may propose one filtration-consistent
boundary motion. Two deterministic visual shell witnesses inspect the same
motion with sampling offsets 0.25 and 0.75:

- both scores strictly positive: `SUPPORT`;
- both scores strictly negative: `REJECT`;
- opposite signs or either score exactly zero: `ABSTAIN`.

Only `ABSTAIN` delegates boundary jurisdiction to the transcript proposal.
`SUPPORT` and `REJECT` both preserve the visual base interval. No polarity
inversion is allowed.

## Confirmatory gates

The mechanism is promoted only if all conditions hold:

1. At least 10 effective boundary edits in total and edits occur in both
   prospective datasets. Otherwise the route is killed for insufficient
   coverage.
2. The number of edits that increase per-video best tIoU is strictly greater
   than the number that decrease it, both overall and in neither dataset may
   wrong edits exceed correct edits.
3. Mean per-video best-tIoU gain is positive in each dataset.
4. Interval F1 at tIoU 0.5 and 0.7 does not decrease in either dataset, and
   the two-dataset macro has a strict gain at one or both thresholds.
5. Native timestamp alignment beats full-orbit, matched-random ambiguity,
   text-only, visual-only, positive-only, and edit-count-matched random
   controls on mean best-tIoU gain; at least the native-vs-full-orbit and
   native-vs-random differences must be positive.
6. No method parameter, prompt, threshold, or routing rule is altered after
   temporal GT is opened.

Passing these gates supports the task-specific jurisdiction paradigm. It does
not by itself establish four-dataset SOTA; final SOTA language additionally
requires the frozen four-dataset benchmark and an independent integrity audit.
