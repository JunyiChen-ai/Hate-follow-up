# Claims from OMSL-v6 results

## Verdict

- `claim_supported: yes`
- `confidence: medium`
- `integrity_status: pass`
- `review_independence: same-family`
- `acceptance_status: provisional`

## Supported development claim

OMSL-v6 is a three-module, label-free-at-inference frame-scoring localizer. On the development-selected local four-dataset 4-fps protocol, it has higher equal-dataset macro point estimates than MultiHateLoc-DMS on the same 643-video coverage and than the lower-coverage 611-video T3AL reconstruction on pooled ROC, pooled PR, and the prespecified within-video ROC metric. Its paired within-video gain over MultiHateLoc-DMS is supported by a video bootstrap. The exact constrained integration has a fresh same-family provisional novelty score of 6.1/10 among the inspected works. V6 makes no interval-decoder claim.

## Unsupported extensions

The evidence does not support untouched-confirmatory SOTA, universal published-method SOTA, a matched official LELA/MultiHateLoc comparison, every-dataset superiority, significant pooled ROC/PR gains, interval localization, causal/model-native interactions, or calibrated interaction probabilities.

## Routing

The development objective is met under explicit exploratory/local-protocol scope. A submission-facing confirmatory SOTA claim requires an untouched cohort, fail-closed evaluator, exact-grid/common-coverage T3AL, all predesignated MultiHateLoc variants, and reproducible published competitors.
