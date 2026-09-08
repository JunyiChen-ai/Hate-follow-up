# Saturation-anchor mechanism pilot: result note

**Date:** 2026-08-08. **Verdict:** **FAIL** under the frozen rule.  
**Compute:** CPU only; existing scores, no new model calls.  
**Preregistration:** `docs/duplex/PREREG_saturation_anchor_pilot.md`  
**Machine-readable result:** `results/saturation_anchor_pilot/results.json`

## What was held out

E7 discovered apparent anchors near `-18.1/+15.0` without using the stance-gate
score file. This pilot tested those locations on three previously scored reader
conditions (`stance_v1`, `stance_para`, `effort_ctrl`) over identical
ImpliHateVid and HateMM diagnostic cohorts. Labels and classification metrics
were not read. The selected cohorts deliberately change score-state occupancy,
which stress-tests the proposed location/occupancy separation.

## Frozen clauses

| Clause | Result |
|---|---|
| At least 4/6 identifiable negative tails, all means within 2.5 of `-18.1` | **FAIL** (6/6 identifiable; 2 outside tolerance) |
| At least 4/6 identifiable positive tails, all means within 2.5 of `+15.0` | PASS (6/6) |
| Anchor location stable while extreme occupancy changes by at least 1.25× | PASS on ImpliHateVid |
| Stable extremes have at most 0.75× the reader sensitivity of stable interiors | PASS on both datasets |

The two failing cells are HateMM `stance_v1` (negative-tail mean `-15.07`) and
`stance_para` (`-15.27`), respectively 3.03 and 2.83 logits from the frozen
negative anchor. HateMM `effort_ctrl` is within tolerance at `-16.55`.

The positive-tail means range from `14.31` to `16.00`, all within the frozen
positive-anchor tolerance. Extreme occupancy changes materially despite this:
1.30× across ImpliHateVid conditions and 1.64× across HateMM conditions.

The paired perturbation signature also appears. Among videos remaining in the
same region under all three readers, the median cross-reader score range is:

| Dataset | Stable extreme | Stable interior | Ratio |
|---|---:|---:|---:|
| ImpliHateVid | 2.00 | 3.75 | 0.533 |
| HateMM | 2.50 | 3.50 | 0.714 |

## Interpretation

The strong symmetric story is false: this judge does not expose two reader-
invariant saturation locations at the preregistered precision. Therefore a
two-anchor latent-state threshold method is not licensed, and no F1 experiment
was run.

There is a narrower post-hoc observation: positive saturation replicated while
negative saturation did not, and both extreme regions were less reader-
sensitive than the interior. This is compatible with an asymmetric decoder or
safety-alignment geometry, where affirmative violation judgments enter a
stable commitment basin but acquittal remains prompt-sensitive. It is not yet
evidence for a method. A one-sided successor would need a new preregistration
and an untouched condition; it must also explain how a positive anchor alone
identifies a full operating point rather than merely a high-precision core.

## Decision

- Retire **symmetric Saturation-Anchored Latent Decision Geometry**.
- Do not run the proposed contamination-mixture threshold experiment.
- Preserve the one-sided positive-saturation observation as hypothesis-
  generating only. It is scientifically interesting but currently lacks a
  label-free rule for the non-saturated middle region, where most threshold
  errors live.

