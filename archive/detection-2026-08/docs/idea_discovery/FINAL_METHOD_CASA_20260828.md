# CASA: Cohort-Adaptive Selective Authority

## Status

CASA is the strongest current successor to VASTA.  It is label-free at
deployment and explicitly adapts modality authority to unlabeled cohort shift.
It currently improves the four-dataset development macro and the
prediction-frozen HateClipSeg p11 train+val result over VASTA, but the latter is
no longer an untouched confirmation for CASA because CASA was designed after
observing VASTA's failure there.

## Core claim

A negative modality should not have a fixed veto right under deployment prior
shift.  Agreement-only decisions from frozen visual experts provide an
unlabeled estimate of the deployment authority regime.  Transcript veto is
opened only when the consensus pool does not show statistically significant
positive dominance.

## Method modules

Frame decoding, feature extraction, ASR generation/alignment, proposal export,
and curve rasterization are preprocessing and are not modules.

### Module 1: Proposal-Set Temporal Support Recovery

The frozen A10/Vid-Group top-8 proposals after temporal NMS at tIoU 0.5 form a
connected support extent anchored at the rank-1 center.  This module owns
temporal boundaries; other modalities cannot move them.

### Module 2: Agreement-Only Cohort Authority Calibration

Frozen A08/VideoTGB and A12/TimeLens decisions partition videos into positive
consensus, negative consensus, and disagreement.  CASA uses only the two
consensus counts.  A one-sided exact sign test under equal positive/negative
consensus probability determines whether the unlabeled deployment cohort is
positive-dominant (`p < .05`).  Disagreement samples do not estimate the
regime, and no target label, dataset identity, or corpus transcript statistic
is used.

### Module 3: Regime-Conditioned Null Arbitration

- Positive-dominant regime: visual OR decides event existence; transcript has
  no veto authority.
- Other regimes: preserve VASTA's state-dependent rule.  Visual consensus is
  authoritative; in visual disagreement, a transcript veto is allowed only
  when its Yes-minus-No logit is below the same model's explicit empty-input
  response.

Text can only remove event existence in an open-authority regime.  It cannot
create an interval or alter temporal boundaries.

## Current evidence

### Four-dataset development cohort

| Method | F1@.3 | F1@.5 | F1@.7 |
|---|---:|---:|---:|
| VASTA | .33488 | .28138 | .17094 |
| CASA | **.33673** | **.28347** | **.17197** |

Detected regimes:

- HateClipSeg: positive consensus 46 vs negative 9, `p=2.17e-7`;
- HateMM: 57 vs 49, `p=.248`;
- MHC: 37 vs 62;
- MHC-zh: 27 vs 45.

CASA therefore changes only HateClipSeg, where F1@.5 improves from .10926 to
.11765; the other three datasets remain exactly VASTA.

### Prediction-frozen HateClipSeg p11 train+val cohort

After correcting the sealed proposal export to match v3's frozen NMS=.5:

| Method | F1@.3 | F1@.5 | F1@.7 |
|---|---:|---:|---:|
| A10 rank-1 | .21636 | **.10182** | .01273 |
| top-8 extent | .21091 | .10000 | .05818 |
| visual OR | .20075 | .09756 | **.05816** |
| VASTA | .19447 | .09342 | .05338 |
| CASA | .20075 | .09756 | **.05816** |

The cohort has positive consensus 114 vs negative 34 (`p=1.35e-11`), so CASA
correctly closes transcript authority.  It improves VASTA by .00414 F1@.5.
This cohort confirms the diagnosis of VASTA but is development evidence for
CASA, not untouched confirmation of CASA.

## Failed and retained variants

- Cross-video length-matched transcript warrants: killed; F1@.5 .27728.
- Native-frame proposal-vs-outside visual rescue: killed; .28137, no main gain.
- Proposal-scoped transcript veto: diagnostic only; dev .28025 and p11
  train+val .09595.
- Hard majority-saturation extent: main F1@.5 .29100 but F1@.7 collapses to
  .09677; not suitable as the final decoder.
- Continuous saturation extent: dev F1@.5 .28537 and p11 train+val .10507, but
  dev F1@.7 .07404; retained as a boundary ablation, not the canonical method.
- SCOPE proposal-disagreement routing: killed after discovering the apparent
  cohort separation was caused by a missing NMS flag in the first sealed
  proposal export.

## Integrity qualifications

- VASTA's sealed run failed its preregistered acceptance test.
- The initial sealed proposal bank omitted NMS=.5; corrected artifacts are
  versioned separately and are authoritative for comparison with frozen v3.
- Before A12 completed, the first 1000 bytes of the gold JSON were printed.
  They contained only an already-developed p11-test ID, not a sealed-cohort ID;
  this process deviation is recorded separately.
- CASA requires a new untouched cohort or dataset-family holdout before a
  generalization/SOTA claim.

## Current novelty position

The proposed contribution is not generic score fusion.  It is deployment-time,
agreement-only allocation of a modality's veto right under unlabeled cohort
shift.  A final novelty score and claim remain pending independent review and a
new untouched validation.
