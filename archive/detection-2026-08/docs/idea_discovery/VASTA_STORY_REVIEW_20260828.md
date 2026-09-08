# Independent review of revised VASTA

## Review status

- Review independence: same-family
- Acceptance status: provisional
- Scope: code and artifact audit of the revised left-padded, batch-one VASTA run

## Defensible method story

VASTA should be presented as **state-dependent modality authority**, not as
ordinary multimodal fusion and not, with the current evidence, as a demonstrated
use/mention reasoner. Its contribution is that a modality is allowed to affect
the answer only in the state where its evidence is useful and least destructive.

The three method modules are:

1. **Proposal-set Temporal Support Recovery.** A frozen proposal bank recovers
   temporal support missed by the top proposal. This is an extent-calibration
   module, not the core novelty claim: the existing falsification indicates that
   much of its gain is explainable by longer support.
2. **Visual Authority State Estimation.** Two heterogeneous frozen visual
   experts map a video to visual-consensus-positive,
   visual-consensus-negative, or visual-disagreement. Visual consensus retains
   final authority; disagreement is an explicit defer state.
3. **Null-Referenced Selective Text Arbitration.** A frozen transcript judge is
   calibrated by one explicit empty-transcript query at deployment. It receives only a
   veto, and only under visual disagreement. This is asymmetric arbitration,
   rather than score averaging or unrestricted text dominance.

Defensible one-sentence claim:

> We formulate label-free hateful-video localization as constrained allocation
> of modality authority: visual experts recover temporal support and retain
> control under consensus, while a null-referenced transcript expert receives a
> selective veto only when visual evidence disagrees.

## Evidence checked

- `run_melt.py` sets tokenizer padding to the left before next-token scoring.
- The batch-one stance run contains 611 rows. Relative to the left-padded batched
  run, 361 logits differ numerically and three final keep/drop decisions differ;
  macro interval F1@0.5 is 0.28138 versus 0.27977. The conclusion is stable, but
  inference is not bitwise batch invariant.
- The canonical null anchor is now obtained from one separately executed
  `Transcript: ""` query: -12.0. The subsequent 611 batch-one video scores are
  byte-for-byte numerically identical to the prior batch-one run. This removes
  reliance on corpus prevalence or on the availability of naturally empty
  transcripts and costs 612 forwards in total. The older natural-null audit is
  useful only as a compatibility observation: its median also happened to be
  -12.0.
- The extent factorial separates the two effects. With the revised null gate,
  A10 extent reaches 0.20354 macro F1@0.5, global calibrated extent reaches
  0.27300, and top-eight proposal extent reaches 0.28138.
- Matched informativeness controls are below the learned veto: random veto
  0.26288 and shortest-transcript veto 0.26730 versus VASTA 0.28138 macro
  F1@0.5. Thus the transcript score contains useful information beyond veto
  sparsity and transcript length.
- The stance and generic prompts are nearly indistinguishable: only five of 611
  final decisions differ, with macro F1@0.5 0.28138 versus 0.28093. Therefore
  the current experiment does **not** establish stance-specific understanding.
- The A08 parser audit finds 611/611 decisions equal to the anchored first
  yes/no answer. Twenty-five generations contain contradictory dialogue after
  EOS. Parsing the EOS-anchored first answer is coherent, but this checkpoint
  pathology must be disclosed.

## Novelty and claim limits

- Task-specific novelty: **6.1/10**, provided the paper centers constrained
  modality authority and selective arbitration.
- General methodological novelty: approximately **4.5/10**.
- If framed as demonstrated stance-aware use/mention reasoning, the score falls
  below six because the generic-prompt control is almost identical.
- Current evidence does not yet prove an untouched-test SOTA claim. The existing
  leave-one-dataset-out analysis is explicitly retrospective, and the four
  datasets participated in method development.

## Highest-value missing experiment

Freeze all modules, thresholds, prompts, and the choice of extent recovery, then
evaluate once on a genuinely sealed external cohort. Report both top-eight
proposal support and global extent calibration so the core authority claim is
not dependent on the more elaborate boundary module. This has greater review
value than adding another module because it converts the current exploratory
SOTA-looking result into evidence of generalization.

If an additional mechanism experiment is possible, construct label-free
counterfactual transcript pairs that preserve hateful words while changing
assertion into quotation or condemnation. Require the actual-minus-null margin
and veto decision to change in the predicted direction. Unless this succeeds,
Module 3 should be called an informativeness veto rather than a stance veto.
