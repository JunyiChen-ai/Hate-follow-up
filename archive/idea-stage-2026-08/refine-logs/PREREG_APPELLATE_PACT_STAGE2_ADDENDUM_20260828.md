# Pre-GT stage-2 addendum: complete tri-state appellate jurisdiction

This addendum is frozen after observing **only model-side witness states**, but
before reading any class or temporal annotation for either prospective cohort.
It does not retroactively change the stage-1 preregistration: the
ABSTAIN-only rule failed its coverage gate (1 edit total, none in Chinese) and
is killed.

The complete tri-state appellate decoder was proposed in the independent
pre-GT method review (`IDEA_REPORT_20260828_APPELLATE_PACT_AUDIT.md`, lines
144--161):

- `SUPPORT` (both visual witness scores positive): approve the full-orbit
  certified boundary motion;
- `REJECT` (both scores negative): deny the motion and preserve visual closure;
- `ABSTAIN` (opposite signs or either score zero): let the already
  alignment-exclusive, full-orbit-certified language motion carry the appeal.

Thus the decoder edits unless vision supplies stable negative evidence. No
score magnitude threshold is introduced and no polarity is inverted.

Before GT, the frozen state counts are SUPPORT=17, REJECT=1, ABSTAIN=1. The
resulting method has 18 effective edits (English=10, Chinese=8).

## Stage-2 confirmatory gates

1. Effective edits >=10 overall and present in both datasets (satisfied
   pre-GT by construction of the frozen rule, not a performance claim).
2. Correct best-tIoU edits must strictly outnumber wrong edits overall; wrong
   may not exceed correct in either dataset.
3. Mean per-video best-tIoU gain must be positive in each dataset.
4. Interval F1@0.5 and F1@0.7 must not decrease in either dataset; the
   two-dataset macro must strictly improve at one or both thresholds.
5. The full tri-state decoder must beat visual-support-only, ABSTAIN-only,
   text-only, single-rotation, and edit-count-matched random controls on mean
   best-tIoU gain.
6. No rule, prompt, threshold, or method parameter changes after prediction
   hashes below are frozen and temporal GT is opened.

Passing supports the complete appellate-jurisdiction mechanism, but final SOTA
language still requires the frozen four-dataset comparison and independent
integrity/novelty review.
