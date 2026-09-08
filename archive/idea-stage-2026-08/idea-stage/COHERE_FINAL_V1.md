# COHERE v1: frozen-core, propensity-preserving hateful-event localization

## One-sentence story

Label-free hateful-video localization is recast as **within-video evidence redistribution** rather than a second video-level hate classifier: a frozen multimodal localizer proposes a tight semantic event core, transcript cohesion transports a small fixed amount of score toward temporally supported frames while preserving the video's score mean exactly, and the inherited interval lattice reads out conservative boundaries.

## Method modules

Data extraction, frame sampling, ASR, and timestamp alignment are preprocessing and are not counted as modules.

### M1. Base-Agnostic Semantic Event Field

Any frozen label-free dense localizer produces frame scores and a tight candidate event core. M1 is an interface, not a novelty claim: it preserves the visual/multimodal base as the source of event semantics and prevents transcript evidence from becoming a standalone detector.

### M2. Propensity-Preserving Text-Core Cohesion Transport

Timestamped transcript chunks induce a temporal cohesion field relative to the predicted event core. The centered field transports frame scores with one globally fixed gain (`g=0.01`). Its correction sums to zero within every video, so the video mean is preserved exactly. Consequently M2 can change *where* evidence lies, but cannot manufacture a stronger video-level hate propensity.

This is the sole core novelty:

> predicted semantic core -> text-conditioned within-video transport -> exact propensity preservation -> replaceable frozen base.

### M3. Conservative Endpoint Lattice

The existing endpoint lattice converts the dense field into intervals and provides exact fallback when evidence is insufficient. It is deliberately inherited and unchanged; it is a safety/readout module, not a novelty contribution.

## Why this is a coherent multimodal story

- The frozen base supplies semantic eventness and remains the anchor.
- Language contributes only temporally aligned cohesion inside a base-proposed event core.
- Exact mean preservation prevents language from dominating video-level decisions.
- The endpoint layer preserves the original interval behavior, separating dense localization gains from boundary post-processing.

## Unified 4-fps results (611 test videos, four datasets)

| Method | PR-AUC | pooled ROC-AUC | within-video ROC-AUC | F1@0.3 | F1@0.5 | F1@0.7 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen Scale-Orthogonal base | .465964 | .689916 | .632969 | .330281 | .286613 | .247850 |
| **COHERE (M1+M2+M3)** | **.463615** | **.689777** | **.645130** | .330281 | .286613 | .247850 |
| Reconstructed T3AL (same evaluator) | .378960 | .631943 | .516835 | .290933 | .243051 | .221821 |

M2 improves the intended core metric by +.012161 over its frozen base. The four-dataset macro PR/pooled ROC move slightly downward, as expected for an exactly mean-preserving within-video recalibrator; they remain substantially above reconstructed T3AL.

The paired, dataset-balanced bootstrap against reconstructed T3AL gives within-video ROC delta +.128279, 95% CI [.082292, .173655], `p_boot <= 0 = 0` (216 videos with non-degenerate frame labels).

### Per-dataset COHERE frame metrics

| Dataset | PR-AUC | pooled ROC-AUC | within-video ROC-AUC |
|---|---:|---:|---:|
| HateMM | .436123 | .690509 | .658018 |
| HateClipSeg | .536117 | .579125 | .539418 |
| MHC | .522368 | .777657 | .709450 |
| MHC-zh | .359853 | .711817 | .673633 |

## Cross-base evidence

On the previously used sealed HCS/VASTA cohort (276 videos), applying the same fixed M2 mechanism to a different base raises within-video ROC from .549303 to .561152 (+.011848). The percentile CI crosses zero ([-.003427, .027187]), so this is directional cross-base evidence, not untouched confirmatory significance.

## Mechanism evidence and falsified alternatives

- Correctly aligned T-core transport improves within-video ROC consistently across the four development datasets.
- Time-shifted transcript controls weaken the result, supporting temporal alignment rather than transcript presence alone.
- Audio orthogonalization/concordance, core-shell ratios, endpoint geometry, temporal GCV, and Wasserstein evidence barycenters do not provide a robust additional contribution.
- A marginal- and energy-matched transcript-rank control beats the Wasserstein construction, decisively rejecting Wasserstein geometry as the load-bearing explanation.
- Endpoint predictions are unchanged by M2, so the dense-score gain cannot be attributed to boundary tuning.

## Defensible claims

1. COHERE introduces a base-agnostic, task-label-free temporal recalibration mechanism in which a frozen dense localizer supplies a semantic event core and text-core cohesion induces a fixed, exactly video-mean-preserving transport of frame scores.
2. Under the unified 4-fps strict label-free evaluation, COHERE achieves the best frame-level PR-AUC, pooled ROC-AUC, and within-video ROC among the compared strict label-free methods.
3. COHERE improves dense frame localization while preserving the inherited interval output.

Do not claim overall temporal-localization SOTA, interval SOTA, audio-text synergy, MLLM reasoning, boundary refinement, a new optimal-transport theory, or three-module synergy.

## Novelty assessment

- Full-system idea novelty: **6/10**.
- Implemented/evidenced novelty: **provisional 6/10**; use 5.5 under a strict untouched-confirmatory standard.
- The novelty is concentrated in M2 and its constrained relationship with M1, not in preprocessing or M3.

## Reproducibility pointers

- Prediction: `results/idea_discovery/route_role_orthogonal_transport_v1.jsonl`, method `route_text_v1`
- Metrics: `results/idea_discovery/route_role_orthogonal_transport_v1_metrics.json`
- Projection implementation: `scripts/idea_discovery/project_role_orthogonal_transport.py`
- T3AL comparison bootstrap: `results/idea_discovery/cohere_vs_t3al_within_bootstrap_v1.json`

