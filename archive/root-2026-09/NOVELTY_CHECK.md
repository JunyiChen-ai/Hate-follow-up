# Novelty Check: Three-scale LESS Candidate

**Date:** 2026-08-28  
**Reviewer:** GPT-5.6-Sol xhigh with a fresh same-family verifier  
**Status:** Provisional

## Verdict

**Overall novelty: 2.5/10. Required threshold (6/10): FAIL.**

The exact combination of L-curve, run-level GCV, and spectral empirical-noise
smoothing was not found in prior temporal-localization work. That exactness is
not enough: the selectors and equal model averaging are classical, there is no
new common risk objective or guarantee, the propensity tribunal changes only a
video intercept, and all intervals are inherited from T3AL.

## Claim assessment

| Claim | Novelty | Finding |
|---|---|---|
| Training-free multimodal hate localization | Low | LELA already defines this setting. |
| Global propensity / temporal residual separation | Low | Global-local separation is established in T3AL, FreeZAD, and TAL. |
| Three self-selected temporal smoothers | Medium | Exact trio appears new, but is a classical-method assembly. |
| Role-separated propensity tribunal | Low | Mean/max/peak pooling is established and the implementation is intercept-only. |
| Overall paradigm | Low | Engineering integration rather than a new localization paradigm. |

## Scores

- Task novelty: 1/10
- Mechanism novelty: 3.5/10
- Integration novelty: 2.5/10
- Overall novelty: 2.5/10

## Closest work

- LELA, *Towards Training-free Multimodal Hate Localisation with Large
  Language Models*, arXiv:2602.09637 (2026).
- MultiHateLoc, WWW 2026 / arXiv:2512.10408.
- T3AL, CVPR 2024.
- Memory Matters, CVPR 2026.
- Moment-GPT, AAAI 2025.
- Modality-Collaborative Test-Time Adaptation, CVPR 2024.
- FreeZAD, 2025.
- Self-SiMS, arXiv:2607.19027 (2026).

## Required redesign

The plausible path above 6/10 is a certified, identifiable separation of
global multimodal propensity from localized temporal evidence:

1. derive both quantities from exactly one frozen multimodal posterior;
2. reinterpret the three temporal fields as uncertainty observations under one
   joint label-free risk model rather than as an ensemble;
3. normalize modalities against within-video temporal nulls and use
   bounded-influence aggregation;
4. inject aligned semantic evidence into the temporal residual;
5. decode uncertainty-aware boundaries from the final posterior and validate
   them against full-video and nuisance controls on an untouched cohort.

