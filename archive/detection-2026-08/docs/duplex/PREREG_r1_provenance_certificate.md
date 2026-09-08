# Preregistration — R1 provenance certificate kill test

Frozen before any R1 model output is produced. Date: 2026-08-25.

## Question

Can deterministic, payload-preserving video-side interventions distinguish asserted/endorsed hostility from quoted/reported/countered mention before adaptation?

## Cohorts (evaluation labels only)

- HateMM: blinded audit category `D` is the asserted/policy-disagreement stratum; `Q` is quoted/reported/countered mention.
- MHC-ZH: protected `FN` cases coded `G` or `I` are the asserted/implicit-hostility stratum; `FP` cases coded `Q`, `R`, or `P` are mention/report/protected-topic controls.

The coding is never read by the scorer. It is joined only by the analyzer after all variant scores are complete.

## Automatic payload rule

- HateMM: the timestamped ASR chunk with the largest already-frozen isolated-chunk hate score; ties by earliest chunk.
- MHC-ZH: uploader title with HTML markup stripped when nonempty; otherwise the middle half of the frozen transcript.

No payload span is manually selected.

## Frozen views

1. `full`: original frozen title, transcript and 16 frames.
2. `payload_only`: payload retained; surrounding transcript/title context removed.
3. `no_payload`: automatically selected payload removed.
4. `source_shift`: original text retained; the 16-frame timeline cyclically shifted by 8 positions.
5. `nuisance`: payload retained; an equal-length substring is removed from the opposite end of the non-payload context.

All views use the frozen Qwen3-VL-8B duplex policy prompt. The scorer saves raw Yes/No margin and final prompt-position hidden states. No stance question is asked.

## Certificate

For video `i`:

`context = z_full - z_payload_only`  
`binding = z_full - z_source_shift`  
`nuisance = abs(z_full - z_nuisance)`  
`certificate = context + binding - nuisance`

This signed formula is fixed before scoring. `no_payload` is a deletion-only control and is not part of the certificate.

## Evaluation

Report per-corpus AUROC for asserted=1 versus mention=0 using certificate, context-only, binding-only, deletion-only (`z_full-z_no_payload`), and original `z_full`. Also report certificate AUROC after residualizing it linearly against `z_full`; fitting is evaluation-only leave-one-out and never affects scores.

## Decision

R1 survives only if both corpora have residual certificate AUROC >= 0.70 and neither deletion-only nor source-shuffled placebo matches it. Otherwise the CCPWG idea is killed before adapter or boundary work.
