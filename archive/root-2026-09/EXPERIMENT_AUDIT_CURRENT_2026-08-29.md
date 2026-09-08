# Experiment Audit — current evidence-capital candidate

**Date:** 2026-08-29  
**Auditor:** fresh GPT-5.6-Sol ultra agent, same-family/provisional  
**Overall verdict:** **FAIL for confirmatory SOTA or final-method claims**

The raw results are real and numerically reproducible, but the current
evidence-capital boundary candidate is a development-selected, partial-coverage
experiment whose claimed boundary mechanism does not survive its controls.

## A. Ground-truth provenance: WARN

The evaluator reads dataset `y4` arrays only in the evaluation process
(`scripts/label_free_adapt/evaluate.py:121-126`). Four distinct dataset cohorts
are genuinely evaluated: HateMM 207 predictions, HateClipSeg 110, MHC 153, and
MHC_zh 141. Dataset/version provenance is not embedded in the NPZ files, and
equivalence to an official interval evaluator has not been established.

Classification: provisional `real_gt`, custom evaluator.

## B. Score normalization: PASS

ROC-AUC and PR-AUC are computed directly from finite raw predictions
(`scripts/label_free_adapt/evaluate.py:93-109`). No metric is divided by a
prediction-derived maximum or other self-normalizer.

## C. Results and arithmetic: PASS with reporting warnings

Independent recomputation exactly reproduces the candidate, A10, and
MultiHateLoc metric files. Both audited candidate JSONLs contain 611 unique,
properly aligned records and 257,569 finite frame scores. Every score length is
`floor(4*duration)`.

The evidence-capital artifact has exactly the same score curve as its upstream
dual-channel artifact in all 611 videos; only intervals differ. Therefore the
new boundary module supports no frame-score improvement claim.

The interval confidence field is an e-value, not a probability: 168/611 third
interval values exceed one. Downstream consumers must not interpret it as a
`[0,1]` confidence.

## D. Live implementation: WARN

The code obtains endpoints from an external visual proposal
(`project_evidence_capital_boundary.py:44-63`), computes language/audio endpoint
witnesses (`:65-90`), and transports open video edges by `e/(1+e)` (`:92-108`).
Array alignment and numerical execution pass.

However, the current frozen M1/M2 artifacts predate metadata now emitted by the
source code and do not pin source hashes or the exact generating CLI. The A10
proposal source was selected after inspecting the 611-video development cohort,
contradicting the output metadata `label_selected_parameters=0` at lines
123-128.

## E. Scope and controls: FAIL

- Coverage is 611/644 videos (94.876%); 33 GT videos have no candidate score.
- Frame metrics are complete-case while interval metrics count missing rows as
  empty (`evaluate.py:142-159`).
- Within-video ROC is defined on only 216 videos overall, including 32/161 MHC
  and 9/149 MHC_zh videos.
- All 611 rows record `development_selected_design=true` and
  `eligible_evidence_status=exploratory_until_untouched_confirmation`.
- The e-capital attribution fails controls. Macro interval F1 at IoU
  0.3/0.5/0.7 is 0.284885/0.250044/0.211143, versus
  0.280189/0.250267/0.217337 for fixed `e=0.5` and
  0.285097/0.238936/0.218399 for the full-video control. The proposed dynamic
  evidence loses to fixed capital at 0.5 and 0.7 and to full-video at 0.3 and
  0.7.
- Against A10, the 0.3 gain is not significant. Existing bootstrap files use
  the 611 common complete cases rather than the 644-row coverage-aware
  estimand; their point estimates therefore do not match the headline metric.
- MultiHateLoc is a one-seed local reimplementation with inferred protocol
  choices and 1-fps-to-4-fps interpolation. It is a useful local comparator,
  but by itself cannot establish published-method SOTA.

## F. Evaluation type: WARN

Real dataset GT with a custom evaluator. Frame AUC is complete-case;
within-video ROC skips videos whose GT is constant; event F1 performs greedy
one-to-one micro matching per dataset and then four-dataset macro averaging
(`evaluate.py:30-66`; `evaluate_four_datasets.py:48-57`).

## Claim impact

- Raw numerical reproducibility: **supported**.
- Four genuinely distinct datasets: **supported**.
- Evidence-capital improves frame localization: **unsupported** (scores are
  unchanged).
- Evidence-capital is the causal source of boundary gains: **unsupported by
  controls**.
- Confirmatory label-free SOTA: **unsupported**.
- Final three-module method: **unsupported; candidate withdrawn from mainline**.

## Required actions

1. Replace the external/development-selected A10 proposal and the failed
   e-capital boundary module.
2. Complete all 644 predictions or report both common-cohort and conservative
   missing-score sensitivity.
3. Align uncertainty estimates with the exact headline estimand.
4. Pin source hashes, CLI, environment, and candidate status in a current
   tracker.
5. Obtain protocol-aligned published-baseline evidence before any SOTA claim.

