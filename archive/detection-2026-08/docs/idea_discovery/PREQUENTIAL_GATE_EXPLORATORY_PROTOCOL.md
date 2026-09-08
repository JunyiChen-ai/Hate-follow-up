# Exploratory protocol: Prequential Cross-Modal Write Gate

Status: mechanism-development protocol. Test labels may be inspected while debugging and
selecting a stable region, as explicitly allowed by the user. Numbers from this phase are
**not** confirmatory benchmark results. A later leave-one-dataset-out run must freeze all
choices before scoring the held-out corpus.

## Hypothesis

T3AL fails when high visual similarity is written back as pseudo-positive evidence even
though that write only reinforces the selected frame. A candidate write is useful when a
one-step virtual update improves the visual margin near independently scored, timestamped
ASR evidence, while producing little far-field drift. Missing ASR is neutral rather than
negative.

For candidate frame `i`, virtually update the hateful prototype and measure

`J_i = local_ASR_weighted_improvement - lambda * far_field_abs_drift`.

Only candidates with positive `J_i` are admitted to the real update. This differs from
confidence, entropy, centroid distance, and synchronous modality-agreement gates: admission
depends on the *consequence* of the proposed update on held-out temporal evidence.

## Exploratory comparisons

- frozen CLIP/T3AL-like query curve;
- ungated top/bottom reservoir update;
- confidence gate at matched coverage;
- contemporaneous ASR-agreement gate at matched coverage;
- prequential gate at matched coverage.

Primary diagnostics are within-video macro ROC-AUC, pooled PR-AUC, and interval F1@0.5 on
each of HateMM, HateClipSeg, MHC, and MHC-ZH. Also report gate coverage, correlation of `J`
with raw confidence, and local-gain/far-drift components.

## Development and confirmation

Exploration may sweep prompts, learning rate, locality radius, drift weight, candidate
quantile, and ASR calibration. A mechanism survives only if there is a broad, interpretable
region rather than one isolated optimum. Confirmation uses leave-one-dataset-out: choose a
single configuration on three corpora and evaluate the fourth untouched, rotating all four.

Kill the central claim if prequential scores are essentially confidence (`Spearman > .9`),
if the gate admits almost all or almost no candidates, if local improvement does not exceed
far-field drift, or if matched-coverage confidence/agreement explains the gain.
