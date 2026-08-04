# Pre-registration — Duplex Reading Kill-Test

**Date frozen**: 2026-08-05, before any probe job was submitted.
**Code**: `src/duplex/score_duplex_probe.py` (scoring), `src/duplex/analyze_duplex_probe.py` (analysis).
**Data**: ImpliHateVid `train_clean` split (~1282 videos; EX/IM/NH encoded in `Video_ID` prefix). Test split is not touched.
**Models**: Qwen3-VL-2B-Instruct and Qwen3-VL-8B-Instruct, bf16, vLLM 0.11.0, greedy, constrained Yes/No decoding, renormalized P(Yes) — identical extraction to the frozen stage-1 scorer.

## Hypothesis

Implicit/coded hateful videos are characterized by a separation between a
surface (literal) reading and a pragmatic (full) reading of the same
content. Operationalized: with z the logit of the renormalized P(Yes),

    D(v) = z_prag(v) − z_lit(v)

should be specifically elevated on IM videos relative to both EX (both
readings see explicit hate) and NH (neither reading sees hate).

## Reader roles

Five prompt blocks inserted into the frozen BINARY_PROMPT skeleton between
the policy rules and the question; everything else byte-identical across
readers. Full texts frozen in `score_duplex_probe.py::READER_BLOCKS`:

- `lit_v1` — surface reading, minimal phrasing (no naming of excluded inference types).
- `lit_v2` — surface reading + explicit exclusion clause.
- `prag` — full reading: explicit content plus conveyed meaning (coded references, wordplay/homophones, irony, title reframing, cross-modal joint effect, audience uptake).
- `prag_para` — semantically equivalent paraphrase of `prag` (noise-floor control).
- `effort` — thoroughness-matched placebo: maximal care, zero pragmatic content.

No prompt contains dog-whistle instances, group names, or lexicon entries —
category labels only. Coded knowledge must come from model pretraining.

## Predictions and failure lines

| # | Prediction | Pass | Fail (kill) |
|---|---|---|---|
| P1 | D right-shifted on IM vs both EX and NH | AUC(IM vs EX via D) ≥ 0.60 AND AUC(IM vs NH via D) ≥ 0.60 | either < 0.60 on both models |
| P2 | IM concentrates in (lit-low, prag-high) quadrant at pool-median cutpoints | IM share in that cell > its share in every other cell, and > EX and NH shares in that cell | IM occupancy flat across cells |
| P3 | D carries signal beyond s_prag where s_prag is ambiguous | in the middle tercile of s_prag, AUC(D: hateful vs normal) ≥ 0.55 | ≤ 0.50 (D is decoration) |
| P4a | lit reader still detects explicit hate | AUC(EX vs NH via s_lit) ≥ 0.70 | below → lit role broke the detector |
| P4b | lit reader suppresses implicit readout | median[z_lit − z_prag] on IM < 0 | ≥ 0 → lit reader leaked (over-reading not suppressed) |
| P5a | placebo does not reproduce P1 | AUC(IM vs EX via D_effort) at least 0.05 lower than via D_prag | placebo matches prag → effect is effort, not pragmatics |
| P5b | paraphrase noise floor is quiet | AUC(IM vs EX via z_prag − z_prag_para) ≤ 0.55 | above → "divergence" is prompt-sensitivity noise |

Kill rule: P1 failing on both models, or P5a failing, or P5b failing, or
P4 failing for both lit variants → the mechanism story is dead; do not
build the method. P2/P3 failing alone → weakened, discuss before deciding.

Failure-mode fingerprints on the IM subset (diagnostic, not pass/fail):
lit-high + prag-high → lit leaked (failure A); lit-low + prag-low → prag
lacks the knowledge (failure B; motivates the distributional-discovery
paradigm instead).

## Selection rule for the lit variant

Choose the variant passing P4a and P4b; if both pass, take `lit_v1`
(more minimal). If both fail on both models, the prompt-level
operationalization is declared failed; the fallback (description-bottleneck
literal reader) is a NEW operationalization requiring a fresh
pre-registration — no wordsmithing of the current blocks.

## Confound-control map

| Confound | Control |
|---|---|
| Directional prior (prag closing expands Yes-scope) | NH group differential (P1 uses IM vs NH, not raw D > 0) |
| Effort/intensity of the second reading | `effort` placebo (P5a) |
| Prompt-wording sensitivity | `prag_para` noise floor (P5b) |
| Ceiling behavior on explicit content | EX group in P1/P2 |
| 8B score compression (known: EN positives pile at <0.01) | logit-space D with clipping at 1e-4; rank-based AUC primary statistic |

## Label-use statement

EX/IM/NH prefixes and gold labels are consumed by the analysis script only,
as diagnostic ground truth for mechanism validation on the train split. No
scoring component, threshold, or prompt consumes them. The test split is
reserved for the eventual paper.

## Post-run discipline

Results are reported for all pre-registered arms regardless of outcome.
Any deviation from this document (new arms, changed thresholds, edited
prompts) must be recorded here as a dated amendment before the deviating
run is submitted.
