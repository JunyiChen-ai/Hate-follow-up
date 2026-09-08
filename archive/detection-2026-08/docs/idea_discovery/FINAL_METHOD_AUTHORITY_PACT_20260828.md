# Authority-PACT: Evidence-Warranted Label-Free Hateful Video Localization

## Status — exploratory candidate rejected as the final method

This file preserves the strongest observed four-dataset point estimate, but an
independent integrity audit rejected PACT-v3 as the final method.  It violated
its preregistered edit-coverage and aligned-vs-shuffle gates, the gains over CCA
were not significant, and the repeatedly used development cohort is not an
untouched confirmation set.  The numbers below are exploratory and must not be
called SOTA or evidence that the contribution novelty exceeds six.

The method is label-free at target inference: it uses no video class label,
frame label, temporal interval, dataset identity in the decoder, or learned
target-domain router.  ASR, timestamp alignment, frame sampling, frozen
feature extraction, proposal extraction, and 4-FPS rasterization are data
preparation and are **not** counted as modules.

## Paradigm

Authority-PACT treats localization as **claim-specific multimodal write
permission** rather than score fusion.  A prediction can be deleted only when
the text modality has earned existence-level authority.  A boundary can be
amended only when a persistent visual claim receives a timestamp-aligned
semantic shell warrant.  Missing or conflicting evidence causes abstention
and exact fallback.

## Module 1: Consensus-Calibrated Existence Authority

Two frozen visual experts define an agreement jurisdiction.  Among consensus
cases where the text expert requests a veto, count text/visual negative
agreements `a` and positive conflicts `b`.  With a fixed `Beta(1,1)` prior,
text may veto a visual-disagreement case only when

\[
P(\theta>0.5\mid a,b)>0.95.
\]

Text can delete an uncertain event, but cannot create an event or move a
boundary.  This module is the inherited CCA existence controller; its
cohort-level transfer is reported as a limitation rather than presented as
ground truth.

## Module 2: Rank-Persistent Visual Proposal Filtration

Let `P1,...,P8` be frozen Vid-Group proposals in native rank order.  At each
rank, form the connected union-support component containing the rank-1 center:

\[
I_k=\mathrm{CC}_{P_1}\left(\bigcup_{i=1}^{k}P_i\right).
\]

After mapping endpoints to 4-FPS cells, collapse identical consecutive states
into `H1,...,HM`.  Visual persistence is the number of proposal arrivals for
which a state remains unchanged:

\[
\pi_v(H)=\#\{k:I_k=H\}.
\]

A non-fallback state is admissible only when
`pi_v(H) > pi_v(HM)`.  There is no learned or label-tuned numerical confidence
threshold.

## Module 3: Null-Anchored Native Semantic Shell Warrant

Every natural timestamped ASR chunk is independently scored by one frozen
Qwen3-VL-8B stance prompt with batch size 1 and left padding.  Its null surplus
is `q_c=z_c-z_empty`, where `z_empty` comes from the identical prompt with an
empty transcript.

At 4 FPS, overlapping chunk surpluses are averaged and ASR gaps are neutral
zero.  For candidate `H_j`, compare its evidence density with every incremental
visual annulus `A_r=H_{r+1}\H_r`.  Under the native timestamps and deterministic
minus/plus-one-chunk perturbations, require

\[
\mu^\delta(H_j)>0
\quad\land\quad
\mu^\delta(H_j)>\max_{r\ge j}\mu^\delta(A_r),
\qquad \delta\in\{-1,0,+1\}.
\]

The language model returns only a binary warrant.  It cannot propose an
endpoint or rank two warranted candidates.

## Module 4: Reciprocal Persistence Decoder

The feasible set is

\[
\mathcal C=\{H_j:j<M,\ \pi_v(H_j)>\pi_v(H_M),\ W_t(H_j)=1\}.
\]

The decoder chooses maximum visual persistence; tied candidates use the larger
extent, and an unresolved endpoint tie returns the closure `H_M`.  If
`C` is empty, or CCA suppresses existence, output falls back exactly.  Thus
vision owns legal endpoints and final scale, while text owns semantic
admissibility; neither score can compensate for failure of the other.

## Four-dataset result

All values are dataset-macro interval F1 on the existing 4-FPS evaluation.

| Method | F1@0.3 | F1@0.5 | F1@0.7 |
|---|---:|---:|---:|
| VASTA | .33488 | .28138 | .17094 |
| CCA-conf95 | .33452 | .28234 | .17301 |
| **Authority-PACT** | **.33452** | **.28352** | **.17361** |

Per dataset:

| Dataset | F1@0.3 | F1@0.5 | F1@0.7 |
|---|---:|---:|---:|
| HateMM | .27219 | .23077 | .15976 |
| HateClipSeg | .22118 | .12235 | .05647 |
| MHC | .41270 | .38095 | .22222 |
| MHC-zh | .43200 | .40000 | .25600 |

Relative to CCA, Authority-PACT changes 20 non-empty boundaries.  Using change
in best gold tIoU as a diagnostic, 8 improve, 7 worsen, and 5 tie; mean change
is positive (`+0.00882`).  The aligned method beats timestamp shuffle by
`+0.00118` F1@0.5 and `+0.00844` F1@0.7.  It also beats visual-only and
text-only controls at both high-IoU thresholds.

## Statistical and scope caveats

- Paired dataset-balanced bootstrap for the gain over CCA:
  - F1@0.5 `+0.00118`, 95% CI `[0, 0.00374]`, `p(delta<=0)=.363`.
  - F1@0.7 `+0.00061`, 95% CI `[-0.00460, 0.00622]`,
    `p(delta<=0)=.407`.
- HateClipSeg gains at F1@0.5 but loses at F1@0.7; HateMM provides the main
  F1@0.7 improvement.  The macro point estimate is therefore not evidence of
  uniform per-dataset dominance.
- The development cohort has been used repeatedly.  A new temporally annotated
  prospective cohort remains necessary for a confirmatory SOTA claim.
- The HateMM validation cohort was prospectively predicted, but it has no
  temporal ground truth and cannot supply this confirmation.

## Novelty assessment after integrity audit

The idea-level task novelty is approximately **6.0/10**, but the experimentally
supported contribution novelty is only **5.5--5.8/10** because 96.7% of outputs
equal the CCA fallback and the key mechanism gates fail.  PACT-v3 is therefore
an ablation/negative result, not the completed method.  A stronger score
requires a frozen prospective temporal cohort, complete controls, and a
load-bearing aligned multimodal effect.

## Reproducibility evidence

- Unified chunk scores:
  `results/idea_discovery/pact/unified_qwen3vl8b_chunk_scores_b1_fullcoverage.jsonl`
  (`sha256 dc2fa4331317c722b51a45ce788f35281bd060dab3a1bbd2086e2c1b3c209a9b`).
- PACT-on-CCA predictions:
  `results/idea_discovery/pact/pact_v3_on_cca_unified_b1_predictions.jsonl`.
- Metrics:
  `results/idea_discovery/pact/pact_v3_on_cca_unified_b1_metrics.json`.
- Bootstrap audits:
  `results/idea_discovery/pact/pact_v3_on_cca_vs_cca_bootstrap_f1_05.json` and
  `results/idea_discovery/pact/pact_v3_on_cca_vs_cca_bootstrap_f1_07.json`.
