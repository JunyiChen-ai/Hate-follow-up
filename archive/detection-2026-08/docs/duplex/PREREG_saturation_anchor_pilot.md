# Pre-registration — Saturation-anchor mechanism pilot

**Frozen:** 2026-08-08 before reading the anchor/tail statistics of the held-out
probe score file.  
**Compute:** CPU only; no new model calls.  
**Status:** mechanism gate, not a threshold-method result.

## Claim under test

For the frozen Qwen3-VL-8B binary moderator, the extreme raw-logit states are
model-owned saturation states while the middle score geometry is
corpus/input-owned. Small reader-block changes may change which videos occupy a
state, but should not substantially move the state locations. This is stronger
than ordinary score bimodality and weaker than claiming that every density
component is a semantic class.

The discovery analysis E7, which did not include the held-out file below,
suggested negative and positive anchors near `-18.1` and `+15.0`. Those values
are frozen here. A result on the same E7 cells has no confirmatory weight.

## Held-out material

`results/stance_gate/probe_scores.jsonl` contains three already-scored reader
conditions on identical per-dataset video cohorts:

- `stance_v1`
- `stance_para`
- `effort_ctrl`

The conditions cover ImpliHateVid train and HateMM test. They were generated to
test stance elicitation, not anchor geometry, and were excluded from E7. The
cohorts are label-selected diagnostics, so labels and classification metrics
are not used here. Selection changes occupancy by design and therefore makes a
useful stress test of the proposed separation between location and occupancy.

## Frozen readout

For each dataset × reader condition:

- negative anchor band: `z <= -13`
- positive anchor band: `z >= +13`
- interior: `-13 < z < +13`
- a band is identifiable only with at least five observations
- report its count, occupancy, mean, SD, and a 2,000-draw bootstrap CI for the
  mean (seed 20260808)

For each dataset, intersect video IDs across the three conditions. For every
video compute its range across conditions, `max(z)-min(z)`. Compare the median
range for videos that are in the same extreme band under all three conditions
against videos that remain interior under all three. This paired stability
test is descriptive and label-free.

## Frozen decision rule

The mechanism **PASSES** only if all clauses hold:

1. At least four of the six dataset × condition cells have an identifiable
   negative band, and every identifiable negative-band mean is within `2.5`
   logits of `-18.1`.
2. At least four of six cells have an identifiable positive band, and every
   identifiable positive-band mean is within `2.5` logits of `+15.0`.
3. In at least one dataset, extreme-state occupancy changes by at least `1.25x`
   across reader conditions while all identifiable anchor means still satisfy
   clauses 1–2. This is the location/occupancy dissociation.
4. For every dataset with at least five stable-extreme and five stable-interior
   videos, the median cross-condition range of stable-extreme videos is no more
   than `0.75` times that of stable-interior videos.

Clause 4 is **not evaluable**, rather than passed, for a dataset lacking the
minimum paired counts. The overall pilot then fails because all clauses must be
demonstrated.

## Interpretation boundaries

- Pass: licenses design of a separate saturation-anchored operating-point
  method and its own preregistration. It does not show that such a method
  improves F1.
- Failure of location clauses: the post-hoc E7 anchors do not replicate; retire
  the mechanism.
- Stable location but failed perturbation clause: the tails may be decoder
  clipping/quantization rather than robust latent decision states; do not claim
  the stronger mechanism.
- No labels, thresholds, transcripts, or video contents enter this pilot.

