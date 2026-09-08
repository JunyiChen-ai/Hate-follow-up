# AMBI: Active Multimodal Boundary Interrogation

## Status

- **Killed as a semantic/MLLM contribution by the equal-action control.**
  With exactly 115 endpoint edits, dense-only fused-transition selection reaches
  F1@0.7 `0.292700`, versus `0.254239` for AMBI authorization; it is also better
  at F1@0.3 and tied at F1@0.5.  The MLLM gate therefore removes useful geometric
  edits rather than explaining the gain.
- Implemented and evaluated on a 278-video development subset with common
  media, dense fields, and nonempty/empty baseline coverage.
- The earlier fixed-lattice MOSAIC candidate is killed: its cached V/T/A
  membership did not improve F1@0.5 or F1@0.7, and the lattice oracle ceiling
  was too low for a credible SOTA path.
- AMBI is a development candidate, not a confirmed SOTA result.

Data decoding, feature extraction, ASR loading, timestamp resampling, canvas
rendering, and cache construction are preprocessing and are not modules.

## Module 1 -- Scale-Orthogonal Evidence Transport

Hard timestamp support supplies the centered within-video temporal residual;
hard/triangular support consensus supplies only video-level propensity. Sparse
relations are qualified against circular transcript rotations and imposed by a
minimum-change projection. This module outputs the dense localization field and
a conservative tight event core.

## Module 2 -- Active Bilateral Multimodal Boundary Interrogation

AMBI does not ask an MLLM to rescore a full video. Each incumbent endpoint is
actively magnified into one local query containing:

- four visual frames sampled from the tight event core;
- eight ordered visual cells around the left or right boundary;
- timestamped transcript associated with those cells;
- independently derived visual, acoustic, and language residual bars.

The MLLM estimates each local cell's membership in the **same event as the
core**, rather than a generic hate probability. The largest directed membership
transition proposes the left or right boundary. This supplies endpoints outside
the inherited tight--midpoint--broad lattice.

## Module 3 -- Separation-of-Powers Boundary Execution

Three sources receive non-overlapping authority:

1. the MLLM membership trajectory may authorize an endpoint direction;
2. the dense fused field supplies the numerical endpoint coordinate;
3. the tight core vetoes any edit that would remove the anchor event.

An endpoint changes only when the MLLM and dense field agree on movement
direction and the dense coordinate preserves the tight core. Left and right are
decided independently; failed checks fall back per side to the current method.

This separation is load-bearing on the development subset. MLLM coordinates
alone lose F1@0.5; fused coordinates alone improve F1@0.7 but substantially hurt
F1@0.3. Semantic authorization plus dense execution retains much more coarse-IoU
coverage while improving precise localization.

## Development Evidence

### Frozen Stage-B 32 pilot

Relative to the midpoint baseline, core-preserving direction consensus left
F1@0.3 and F1@0.5 unchanged and raised F1@0.7 by 0.0682. Eight endpoint actions
were accepted. This cohort was used to select the separation-of-powers rule and
is not confirmatory.

### Expanded 278-video development subset

The rule was then run without prompt or parameter changes. Relative to the
current Scale-Orthogonal Transport lattice candidate:

| Metric | Current candidate | AMBI V1 | Delta |
|---|---:|---:|---:|
| Interval F1@0.3 | 0.383343 | 0.381846 | -0.001497 |
| Interval F1@0.5 | 0.310389 | 0.308892 | -0.001497 |
| Interval F1@0.7 | 0.217274 | 0.254239 | +0.036965 |

AMBI accepted 115 endpoint actions. The paired, dataset-balanced bootstrap for
F1@0.7 was `[-0.00554, 0.11428]`; it includes zero. Frame metrics are unchanged
because AMBI edits interval geometry without replacing the dense field.

### Matched controls

- Raw MLLM coordinates improve high-IoU localization on Stage-B but damage
  lower-IoU coverage.
- Fused transition plus the same tight-core veto reaches F1@0.7 `0.291203` on
  the 278 subset, but F1@0.3 falls to `0.343384`.
- AMBI-authorized fused coordinates reach F1@0.7 `0.254239` while retaining
  F1@0.3 `0.381846`.

The current evidence supports a precision--coverage mechanism, not universal
dominance.

## Next Confirmation Requirement

Before any SOTA or confirmed novelty claim:

1. freeze the prompt, sampling grid, event-core definition, direction rule,
   coordinate rule, and fallback hashes;
2. evaluate on untouched same-domain and cross-domain cohorts;
3. compare against raw MLLM coordinates, fused+core, current lattice, identical
   canvases without transcript/audio/visual bars, and modality rotations;
4. preregister F1@0.7 as the primary metric and F1@0.3 non-inferiority as the
   safety metric;
5. require positive effects beyond one dataset and a paired confidence interval
   excluding zero.

## Claims Not Yet Authorized

- SOTA;
- confirmed novelty of at least 6/10;
- statistically reliable improvement;
- balanced modality use merely because three evidence bars are displayed;
- superiority across all four datasets.

The method is retained as a negative result and must not be promoted as the
current candidate.
