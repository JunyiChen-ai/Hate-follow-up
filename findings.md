# Research findings

## 2026-08-29 — OMSL-v6 result-to-claim gate

- Method: Orthogonal Möbius Semantic Localization, exactly three modules; preprocessing is outside the module count.
- Scope: 643 decodable videos over HateMM, HateClipSeg, MHC, and MHC-zh on the local 4-fps protocol.
- Development result: macro ROC 0.754464, PR 0.568166, within-video ROC 0.645245. These exceed reproduced T3AL and MultiHateLoc-DMS macro point estimates.
- Core inference: within-video gain over MultiHateLoc-DMS is +0.118251 with 95% CI [0.044303, 0.194076] and empirical one-sided p=0.00065.
- Novelty: fresh same-family provisional review gives 6.1/10 for the exact constrained integration.
- Integrity: current v6 audit passes artifact integrity with WARN-level scope/evaluator qualifications.
- Honest boundary: the design was selected after repeated evaluation on this cohort, so the result is exploratory rather than untouched-confirmatory. V6 emits no intervals. Published LELA/MultiHateLoc protocols are not sufficiently specified for a matched official comparison.
- Result-to-claim verdict: supported for the explicit development/local-protocol claim; submission-facing confirmatory SOTA remains unproven.
