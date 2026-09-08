# Pre-registration — Readout-bottleneck kill test (hidden-state construct axes)

**Frozen:** 2026-08-09, before loading any hidden-state file into any
analysis. **Compute:** CPU only; hidden states already on disk (MHClip-EN
test 161 × 37 layers × 4096; HateClipSeg 394 × same). No new model calls.
**Status:** kill test gating the readout-bottleneck method family. Successor
to the dead dual-axis kill test (`DUAL_AXIS_KILLTEST_NOTE.md`).

## Phenomenon

The scalar Yes/No readout conflates constructs the evidence separates:
Hateful-vs-Offensive ranking is random (0.417) under the joint judge, and a
construct-swapped question (dual-axis test) produced an axis that decorrelates
from the original (Spearman 0.935) yet measures nothing new — it keeps the
protected-target ranking bit-for-bit (0.866 = 0.866) and gains only +0.04 on
the no-target stratum. Meanwhile score-blind coders of a different model
family separate the same constructs from the same evidence reliably. The
suspect is therefore neither the evidence nor the question but the readout:
one scalar logit contrast is the bottleneck through which a
multi-dimensional judgment is forced.

## Claim under test

The judge's internal representation carries the targeted-hate vs
generic-offensiveness distinction, and part of it is reachable label-free
(unsupervised) from the corpus's own hidden states. If true, the method
family is "widen the readout": per-axis label-free operating points on
model-internal axes, composed by task specification — the resolution of the
proven one-scalar ill-posedness that prompts (dual-axis) could not deliver.

## Frozen data and constants

- Hidden states: final-token, all layers, as stored by the frozen judge runs
  (`results/testruns/mhclip_en/judge_8b/hidden/`,
  `results/hateclipseg/judge_8b/hidden/`). No rescoring.
- Layers examined: {18, 27, 36} (mid, late, final; embedding row = index 0).
- Unsupervised axes: per layer, PCA fit on that corpus's own hidden states
  (all videos, no labels), top 8 components; axis sign oriented so Spearman
  with the corpus's z is ≥ 0. Candidate set: 3 layers × 8 PCs = 24 per
  corpus.
- Evaluation strata (labels used for evaluation only): MHClip-EN blind-coded
  no-protected-target positives (34) vs shipped Normals (112);
  protected-target positives (15) vs Normals; HateClipSeg insulting-only
  videos (59) vs clean normals (50), from the shipped multi-labels.
- Reference AUCs on those strata: z_hate 0.749 (no-target), 0.866
  (protected-target); HateClipSeg z on insulting-only vs normal 0.637.

## Frozen decision rule

The family **SURVIVES** only if all three clauses hold:

1. **Information exists (supervised ceiling, diagnostic).** A logistic
   probe on MHClip-EN layer-27 hidden states, no-target positives (34) vs
   protected-target positives (15), leave-one-out CV, reaches AUC ≥ 0.75.
   Below 0.75: the representation does not separate the constructs; the
   bottleneck hypothesis is false; the family dies regardless of clause 2–3.
   (The probe is a measurement instrument only; it can never be a method
   component under the label-free constraint.)
2. **Label-free access on EN.** At least one of the 24 unsupervised EN axes
   reaches AUC ≥ 0.799 (= z + 0.05, the same bar the dual-axis call failed)
   on the no-target vs Normal stratum.
3. **Cross-corpus replication.** For the layer of the best EN axis from
   clause 2, at least one of that layer's top-3 HateClipSeg PCA axes
   (HateClipSeg's own unlabeled fit) reaches AUC ≥ 0.687 (= its z + 0.05)
   on insulting-only vs clean-normal. This guards clause 2 against
   24-candidate fishing.

Reported descriptively either way: full per-layer per-PC AUC tables on all
strata; correlation of each qualifying axis with z; protected-target AUC of
qualifying axes (construct specificity); the same analysis at layer 36 only
for comparison with the z bottleneck.

## Interpretation boundaries

- All three pass: licenses a full method preregistration (per-axis
  label-free thresholds, task-spec composition, held-out corpora). Not
  itself a performance claim.
- Clause 1 fails: the model does not internally separate the constructs;
  retire the bottleneck story AND note that the blind-coder comparison
  localizes the gap to the model, not the readout.
- Clause 1 passes but 2 fails: information exists but is not reachable
  unsupervised; the family is blocked under the label-free constraint;
  record and stop — no supervised rescue is permitted.
- Clause 3 fails: the EN axis is corpus-fitted noise; dead, matching the
  falsification map's cross-corpus discipline.
- Hidden-state PCA uses no labels, no prompts, no extra calls; it reads the
  single existing call's internal state, so the ≤2-call budget and the
  single-pass deployment shape are untouched.
