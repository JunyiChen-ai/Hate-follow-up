# Paradigm Adaptation Results — 2026-08-26

Scope: label-free target inference on deterministic clean cohorts. Stage-A has 2 videos per dataset; Stage-B has 8 per dataset. Ground truth was opened only by `scripts/label_free_adapt/evaluate.py` after prediction artifacts were closed. Empty-positive subsets have undefined ROC and cannot support a claim.

## Decision summary

| ID | Adapted paradigm | Stage | Decision | Evidence |
|---|---|---:|---|---|
| P1 | Counterfactual evidence localization | 8 | **Stop** | T3AL improved on HateMM but degraded on HateClipSeg; F1@0.5 was zero |
| P2 | Visual temporal canvas | 8 → 32 | **Survives** | T3AL within-video ROC improved on every defined Stage-B dataset |
| P3 | Controller-guided coarse-to-fine search | 8 | **Stop** | worse than P2 on both defined T3AL within-video ROCs |
| P4 | Distributional boundary posterior | 8 | **Stop** | all defined interval F1@0.3/0.5 were zero |
| P5 | Set-valued localization | 8 | **Stop** | wider possible set still produced zero interval F1 |
| P6 | Temporal role experts | 8 exploratory | **Survives provisionally** | strong HateMM gains; Poset within-video ROC also improved on HateClipSeg; auxiliary TimeLens is externally temporal-supervised |
| P7 | Prequential cross-video memory | 8 × 3 orders | **Stop** | gain existed only in the one order where the sole certified write preceded another video |

## P2 confirmation (8 videos per dataset)

| Dataset | Method | ROC | PR | within-video ROC | F1@0.3 | F1@0.5 |
|---|---|---:|---:|---:|---:|---:|
| HateMM | T3AL base | .469 | .256 | .363 | .000 | .000 |
| HateMM | T3AL + canvas | **.706** | **.520** | **.402** | .000 | .000 |
| HateClipSeg | T3AL base | .545 | .453 | .575 | .031 | **.010** |
| HateClipSeg | T3AL + canvas | .513 | .441 | **.583** | **.045** | .007 |
| MHC | T3AL base | .563 | .276 | .791 | .000 | .000 |
| MHC | T3AL + canvas | **.644** | **.421** | **.809** | .000 | .000 |
| HateMM | Poset base | .522 | .287 | .568 | .021 | .000 |
| HateMM | Poset + canvas | **.703** | **.497** | **.597** | **.039** | .000 |
| HateClipSeg | Poset base | **.504** | **.417** | **.510** | .034 | .000 |
| HateClipSeg | Poset + canvas | .473 | .402 | .496 | **.058** | **.010** |
| MHC | Poset base | .522 | .238 | .504 | .000 | .000 |
| MHC | Poset + canvas | **.597** | **.359** | **.514** | .000 | .000 |

MHC_zh had zero positive frames in this 8-video subset, so its ROC is undefined. P2 used 96 successful local Qwen calls, plus no retries; two AV1 videos used audited FFmpeg fallback.

## P6 exploratory role experts (2 videos per dataset)

| Dataset | Method | ROC | PR | within-video ROC | F1@0.3 |
|---|---|---:|---:|---:|---:|
| HateMM | T3AL base | .299 | .003 | .262 | .000 |
| HateMM | T3AL role experts | **.837** | **.012** | **.846** | **.182** |
| HateMM | Poset base | .737 | .017 | .768 | .054 |
| HateMM | Poset role experts | **.880** | **.384** | **.887** | **.087** |
| HateClipSeg | T3AL base | **.676** | **.142** | **.690** | **.014** |
| HateClipSeg | T3AL role experts | .420 | .059 | .682 | .000 |
| HateClipSeg | Poset base | **.449** | **.074** | .445 | **.019** |
| HateClipSeg | Poset role experts | .326 | .048 | **.467** | .000 |

This is post-selection exploratory evidence, not untouched confirmation. The frozen TimeLens-8B expert was rerun on the exact clean cohort with model revision, runner SHA, manifest SHA, FPS, seed, and call count recorded.

## P7 order audit

Only order 0 placed the sole certified HateMM write before another video. In that order, T3AL within-video ROC changed `.310 → .375`, Poset `.836 → .870`, and Poset F1@0.5 `.000 → .043`. Orders 1 and 2 placed the write last, so xmemory equaled the paired no-memory control exactly. No writes occurred on the other three datasets.

## Recommendation

Develop P2 as the primary follow-up and keep P6 as an auxiliary-checkpoint branch. The most defensible new direction is a boundary-aware refinement of P2 that directly attacks its frequent all-constant 16-bin outputs; P3–P5 show that a single zoom, posterior sampling, or set widening alone does not solve that failure.
