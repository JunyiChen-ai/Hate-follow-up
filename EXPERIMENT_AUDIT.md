# Experiment Audit — authoritative OMSL-v6

**Date:** 2026-08-29

**Auditor:** fresh GPT-5.6-Sol ultra agent, same-family/provisional

**Overall verdict:** WARN

**Integrity status:** PASS (provisional)

No GT leakage, prediction-derived metric normalization, phantom result, arithmetic mismatch, duplicate key, nonfinite timeline, or v5/v6 score discrepancy was found. The warnings concern claim scope and evaluator hardening, not corruption of the current artifact.

## A. Ground-truth provenance: PASS

The v6 inference executable accepts manifest, one visual prediction source, timestamped text scores, audio embeddings, and frozen whole-video scores. It has no GT or label input. GT is loaded only by the isolated evaluator. All 643 rows record `gt_access=false`, zero runtime label-selected parameters, `development_selected_design=true`, and exploratory eligibility.

## B. Score normalization: PASS

Within-video ranking and robust scaling are disclosed predictor operations. ROC, PR, and within-video ROC consume the raw unbounded final scores; no metric is divided by a statistic of the prediction itself.

## C. Result existence and arithmetic: WARN

Independent recomputation reproduced v6, MultiHateLoc, and T3AL point metrics. v6 has 643 unique, finite, error-free rows with dataset counts 215/118/161/149 and exact `ceil(duration*4)` prediction lengths. The current code, predictions, and metrics hashes match the recorded values. The 5k pooled point differences were reproduced; its random bootstrap CI stream was not independently replayed by this audit, so the stored pooled CIs retain that qualifier.

## D. Live/dead/mismatched code: WARN

Critical score, rank, calibration, alignment, and evaluator functions are live. The general evaluator uses common-prefix truncation and removes nonfinite samples rather than failing on arbitrary mismatch. Current v6 lengths satisfy the bootstrap scripts' stricter `GT` or `GT+1` rule, so this is evaluator hardening debt rather than current-result corruption. v5 bootstrap artifacts apply to v6 because all 643 score arrays were independently verified bit-for-bit identical.

## E. Scope: WARN

Coverage is 643/644 GT videos. The one missing HateClipSeg video is the documented truncated/undecodable object. Within-video ROC is defined on 226 videos: HateMM 84, HateClipSeg 99, MHC 33, MHC-zh 10. T3AL has 611 predictions and is a lower-coverage reference. The design was iterated on this cohort and is explicitly exploratory rather than untouched-confirmatory.

## F. Evaluation type: PASS — `real_gt`

The evaluator uses dataset-derived `y4` arrays. The local 4-fps protocol is fully specified, but it is not an official published-method protocol; LELA and official MultiHateLoc do not publish enough grid/conversion detail for a matched official comparison.

## Claim impact

- **C1 label-free inference:** supported with qualifier. The final inference stage consumes no target labels, but the design itself was development-selected and upstream frozen models were pretrained.
- **C2 macro point-estimate wins:** supported with qualifier. v6 exceeds reproduced T3AL and MultiHateLoc on equal-dataset macro ROC, PR, and within-video ROC. This is not an every-dataset or official-protocol claim.
- **C3 confirmatory/published-method SOTA:** unsupported. No untouched confirmation or matched official LELA/MultiHateLoc protocol exists.
- **C4 within-video improvement versus MultiHateLoc:** supported. Direct v6 replay gives +0.118251, 95% CI [0.044303, 0.194076], empirical one-sided p=0.00065.
- **C5 positive interval-decoder performance:** unsupported. v6 deliberately emits empty intervals and all interval F1 values are zero.

## Required reporting constraints

1. Call the result exploratory and development-selected.
2. Define the SOTA evidence as local-protocol macro point estimates and the significant core within-video comparison, not universal published-method SOTA.
3. Do not attribute any interval performance to v6.
4. Harden the general evaluator to reject unexpected length mismatches before a submission artifact freeze.
