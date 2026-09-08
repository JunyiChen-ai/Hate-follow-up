# Pre-registration — Duplex Readout Probe (perception–report gap)

**Date frozen**: 2026-08-05, before any hidden-state extraction has been run.
**Status**: frozen. No probe direction, layer, or projection has been computed on this data.
**Prior**: `docs/duplex/PREREG_duplex_killtest.md` and `docs/duplex/KILLTEST_POSTMORTEM.md`.
Every Phase-0 figure quoted below was recomputed from the ten scored files under
`results/duplex_probe/ImpliHateVid/`.
**Data**: ImpliHateVid `train_clean`, 1283 videos, EX/IM/NH encoded in `Video_ID`. Test split is not touched.
**Model**: Qwen/Qwen3-VL-8B-Instruct, bf16, HuggingFace transformers. Qwen3-VL-2B-Instruct is an optional secondary contrast.

## Hypothesis

The kill-test settled two things about the 8B model. First, comprehension of
implicit hate is automatic: instructing the model to read only surface meaning
does not suspend the pragmatic inference, which is the IM fingerprint the
pre-registration called failure A. Second, the verbalized channel commits
bimodally on IM. On the frozen `prag` reader, 0.598 of IM videos are asserted
hateful with renormalized P(Yes) above 0.95, scored the way EX videos are
scored, while 0.239 are dismissed below 0.05, scored the way NH videos are
scored, and only 0.163 sit in the interior. Among videos the model commits on
at all, IM is 3.24 times more likely than EX to be committed low on that
reader, and between 1.96 and 3.24 times more likely on all five 8B readers.

The hypothesis is that the dismissal belongs to the assertion channel rather
than to perception. If so, the model's internal representations still separate
verbally-dismissed IM from verbally-dismissed NH in the stratum where the two
carry indistinguishable near-zero verbalized scores. The named mechanism is a
**perception–report gap**: the pragmatic inference happens during encoding, and
the report suppresses it.

The alternative is failure B restricted to the dismissed subpopulation, meaning
perception is absent there and the internals track the mouth. The kill-test
could not reach that alternative, because it read only the verbalized channel.
This probe can falsify it.

## Operationalization

**Sample.** All 1283 `train_clean` videos. The kill-test retained 1155 of them
and lost 128 to context overflow, every one at 1080p or larger, with the loss
correlated to the labels (EX 20.3%, IM 18.5%, NH 0.3%). The cause was that
`mm_processor_kwargs={"max_pixels": 100352}` never reached the resizer under
vLLM 0.11 with Qwen3-VL. This run enforces the same frame budget through the
processor parameters Qwen3-VL actually reads (`size["shortest_edge"]` and
`size["longest_edge"]`), targeting at most about 100,352 pixels per frame. The
128 HD videos therefore re-enter the sample and the resolution-label confound
is removed at the input, not by a downstream correction.

**Forward pass.** One forward pass per video, no generation beyond the scored
position. Hidden states are inaccessible in vLLM, so extraction runs through HF
transformers. The prompt is byte-identical to the kill-test `prag` reader: the
frozen `BINARY_PROMPT` skeleton with the `prag` block from
`src/duplex/score_duplex_probe.py::READER_BLOCKS`, 16 frames, transcript limit
300 characters. That reader is the frozen judge for this probe and no variant
of it is run.

**Two readouts, one pass.** Both are taken at the final prompt token position,
the position whose next-token distribution yields Yes or No.

1. *Verbalized*: the raw logit difference `z = logit(Yes) − logit(No)`,
   aggregated over the same Yes/No token-id sets as `build_binary_token_ids`.
   No renormalization and no clipping. The kill-test readout, a renormalized
   P(Yes) clipped at 1e-4, is retired: it tied 53.9% of the sample at the clip
   bounds and the postmortem shows the 8B P1 pass was an artifact of that
   censoring.
2. *Internal*: the hidden state `h_ℓ` at that position for every layer ℓ,
   stored fp16.

**Anchors, label-free.** Rank all videos by `z` from this same run. The top 5%
form the pseudo-hateful anchor set and the bottom 5% the pseudo-normal set.
Phase-0 measured the purity of exactly this rule on the prior run's scores:
top 5% 0.966 and bottom 5% 1.000, with the bottom quartile at 289/289 NH.
Robustness points k = 2% and k = 10% are pre-registered.

*Diagnostic abort.* Anchor purity is measured against gold labels for
diagnosis only. If it falls below 0.85 on either tail, the run is declared
invalid for prediction testing and redesigned rather than analyzed, because
contamination attenuates the direction by the factor `p_hi + p_lo − 1` and a
weak result would then be uninterpretable.

**Direction and probe score.** Per layer, the difference of the two anchor-set
means, unit-normalized. The probe score is the projection of `h_ℓ` onto it. No
trained parameters and no labels enter this step.

**Layer selection, label-free.** The primary layer is the one maximizing
held-out anchor classification accuracy under 5-fold cross-validation *within*
the anchor sets. The full layer sweep is reported as a diagnostic and never as
a selection surface for the predictions.

**Strata.** Because renormalized P(Yes) equals `sigmoid(z)`, the kill-test
commitment bands map onto `z` exactly. Verbally dismissed means `z < −2.944`
(probability below 0.05); verbally asserted means `z > +2.944` (probability
above 0.95). Both thresholds are fixed here. On the prior run these strata held
63 IM and 561 NH videos on the dismissed side, and 197 EX and 44 NH on the
asserted side.

## Predictions and failure lines

| # | Prediction | Pass | Fail |
|---|---|---|---|
| P1 | Core dissociation: within the verbally-dismissed stratum, the probe separates IM from NH where `z` cannot | AUC(dismissed-IM vs dismissed-NH, probe at primary layer) ≥ 0.65 AND ≥ same-stratum AUC via raw `z` + 0.10 | probe ≤ z-baseline + 0.03: internals track the mouth, perception is absent for the dismissed subpopulation (failure B) → **kill** |
| P2 | Gain asymmetry, the mechanism's signature: the probe buys more on implicit than on explicit content | Δ = probe AUC − raw-`z` AUC on the full sample, with Δ(IM vs NH) ≥ Δ(EX vs NH) + 0.03 | uniform gain: the probe is a better-calibrated readout of the same signal, not a dissociation → **kill** |
| P3 | Explicit sanity: the direction is a hatefulness direction | AUC(EX vs NH via probe) ≥ 0.90 (raw `z` reaches about 0.99, so the probe may cost a little) | below 0.90: the direction is not a hatefulness direction |
| P4 | Placebo: the effect is not anchor-set structure | probe AUC for P1 exceeds the 95th percentile of 200 permuted-anchor directions (anchor union, membership shuffled) | at or below that percentile: anchor-structure artifact → **kill** |
| P5a | Stability under anchor resampling | split-half anchor resampling, mean direction cosine ≥ 0.8 | below 0.8: the direction is sample noise |
| P5b | Resolution confound: the recovered HD videos do not carry the effect | P1 passes within the ≤720p stratum alone | fails there: a resolution shortcut, not a hate direction |
| P6 | Over-flagging side, secondary: internals recognize over-flagged benign content | among verbally-asserted videos, AUC(asserted-EX vs asserted-NH via probe) ≥ same-stratum raw-`z` AUC + 0.10 | below that: weakens the symmetric reading, does not kill |

P2 is the probe analogue of the kill-test's placebo discipline. The kill-test
died because a thoroughness-matched placebo reproduced the effect, which showed
the statistic measured effort rather than pragmatics. Here the corresponding
trap is a probe that improves every comparison because it reads the same signal
more cleanly. A gain that is flat across IM and EX is consistent with better
calibration and inconsistent with a perception–report gap, so P2 disconfirms the
mechanism even when the numbers improve.

P6 carries the Phase-0 surface-marker context. Among 97 over-flagged NH videos
against 97 controls, identity-group mention runs at 0.392 versus 0.052, a
factor of 7.6, and any surface marker at 0.639 versus 0.165, a factor of 3.9.
Degenerate transcripts are screened by a rule fixed here, namely a transcript
shorter than 40 characters or with fewer than 50% Latin letters, and P6 is
reported both with and without them. Failure of P6 weakens the symmetric
reading without killing the mechanism, because the over-flagged stratum mixes
topic-versus-stance confusion with ASR degeneracy, and those are different
mechanisms from the one under test.

## Kill rule

The perception–report mechanism is dead for this operationalization, and the
method is not to be built, if any of the following holds:

- P1 fails at the primary layer **and** at every layer in the sweep;
- P4 fails;
- P2 fails.

P3, P5, and P6 failures are reported and discussed before anything else
proceeds.

## Failure-mode fingerprints

Diagnostic, not pass/fail. If the probe separates interior IM but not dismissed
IM, perception is partial and is reported as partial. If dismissed IM is
indistinguishable from NH at every layer, the model lacks the knowledge for
that subpopulation, which argues for a knowledge-side successor rather than a
readout-side one.

## Confound-control map

| Confound | Control |
|---|---|
| Anchor contamination attenuating the direction | measured tail purity against gold, with the 0.85 abort line |
| Salient-feature capture (the Farquhar-style objection that the direction encodes a surface property) | P2 asymmetry, P3 explicit sanity, P4 permuted-anchor placebo |
| Resolution shortcut from the recovered HD videos | processor-level frame budget at the input, plus the P5b ≤720p stratum test |
| Better calibration mistaken for dissociation | P2 |
| Prompt-wording sensitivity | one frozen prompt, the kill-test `prag` block, with no prompt variation anywhere in the design |
| Censored readout destroying rank information | raw unclipped logit difference, the clipped P(Yes) readout retired |
| Layer selection as a hidden degree of freedom | label-free cross-validated selection within the anchor sets; the sweep is diagnostic |

## Label-use statement

Gold EX/IM/NH prefixes and labels are consumed by the analysis alone, as purity
diagnostics and as ground truth for evaluating the predictions. Anchor
selection, direction estimation, layer selection, and every scoring step are
label-free. The test split is reserved for the eventual paper.

## Post-run discipline

Results are reported for all pre-registered arms regardless of outcome. Any
deviation from this document, whether a new arm, a changed threshold, an edited
prompt, or a different extraction position, must be recorded here as a dated
amendment before the deviating run is submitted.
