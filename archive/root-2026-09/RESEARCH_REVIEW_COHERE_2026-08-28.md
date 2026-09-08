# Adversarial review of COHERE (2026-08-28)

Review route: Codex `gpt-5.6-sol`, ultra reasoning.  
Review independence: same-family.  
Acceptance status: provisional.  
Verdict: **1/10, strong reject, confidence 5/5**.

Full round-by-round trace: `.aris/traces/research-review/2026-08-28_run01/`.

## Fatal finding

The current text-cohesion gain is primarily a tie-order artifact. `rank01` uses stable sorting and assigns distinct chronological ranks to equal cohesion values. Transcript features contain extensive repeated rows, so this silently creates an increasing time prior.

| Variant | Macro within-video ROC | Delta vs base |
|---|---:|---:|
| Frozen base | .632969 | — |
| Current stable-tie COHERE | .645130 | +.012161 |
| Correct average-rank ties | .633066 | +.000098 |
| Transcript-free increasing time rank | .646071 | +.013102 |

The semantic mechanism is therefore falsified. Current COHERE must not be presented as a successful method.

## Module audit

1. **M1 is inherited and inelegant.** It combines CCA top-8 closure, a veto system, and T3AL proposals, then uses a global `shorter` heuristic. The core is read from a hard-coded external bank rather than emitted by the advertised base interface.
2. **M2 is falsified.** Correct tie handling removes nearly all gain; `g=.01` and the modality choice were selected on the official test cohort.
3. **M3 is inactive.** The implementation copies inherited intervals. The attempted cohesion endpoint decoder reduced F1@.7 from .247850 to .220549.

## Weakly supervised comparison

No SOTA claim is currently valid. MultiHateLoc (WWW 2026) reports HateMM frame mAP/AUC `.645/.799` and pooled MultiHateClip `.445/.750`, whereas COHERE reports HateMM `.436/.691`. Protocol and split differences prohibit a formal ranking; MultiHateLoc has no usable released code or predictions. Present it as a different supervision point, not a defeated baseline.

Pooled AP is also pathologically video-dominated: a perfect video-label broadcast with no temporal localization reaches `.675/.786/.853` AP on HateMM/MHC-EN/MHC-ZH under the neighbouring repository's audit. Primary claims must instead use within-video and event/boundary metrics, especially on native temporal benchmarks.

## Recommended redesign: TRIARC

### M1 — Reciprocal Orbit Calibration

For each video, calibrate V–T, T–A, and A–V temporal alignment against all non-zero circular shifts. Output per-sample lag and exact orbit-rank reliability only. Use tie-aware midranks; no dataset threshold, fixed top-K, or gain.

### M2 — Role-Disjoint Triadic Event Matching

Text owns hostile proposition/target/stance; vision owns entity/action grounding; audio owns carrier/source continuity. Generate candidates from all unique within-sample superlevel sets and form cycle-consistent V–T–A matches. Keep modality evidence separate and require the tri-modal arm to beat every unimodal and bimodal arm.

### M3 — Exact Max-Null Interval Scan

Score every legal interval by the weakest of its V/A/T support statistics. Repeat the entire search under circularly shifted null alignments, then retain only intervals whose factual score beats the strongest matched null. Select a non-overlapping interval set by deterministic interval scheduling; EMPTY is permitted. This removes fixed thresholds, span counts, smoothing gains, and endpoint lattices.

## Required kill gates

1. Exact manifest and curve-length equality; no silent ID intersection or truncation.
2. Factual multimodal alignment must beat shift, reversal, phase-randomized audio, scene-cut, and time-ramp controls.
3. Full tri-modal matching must beat the best bimodal arm by at least .01 with simultaneous CI above zero; every modality deletion must hurt its prespecified availability slice.
4. M3 must improve F1@.5 and F1@.7 by at least .02 with CI above zero, while F1@.3 loses no more than .005.
5. Dense within-video ROC gain at least .015 with CI above zero; pooled PR/ROC non-inferiority within .005.
6. Freeze the entire method before evaluating a genuinely untouched temporally labelled cohort. The current 611 videos remain development evidence only.

## Claims matrix

| Outcome | Allowed claim |
|---|---|
| M1 alignment controls fail | No multimodal temporal-binding claim; kill TRIARC |
| M1 passes, M2 tri-modal advantage fails | Per-sample alignment diagnostic only; no balanced multimodal localizer |
| M1/M2 pass, M3 interval gate fails | Dense multimodal reranker only; no localization paradigm claim |
| All gates pass prospectively | Target-label-free, per-sample-calibrated tri-modal interval localization with randomization-certified boundaries |

## Strongest possible story

A hateful event is a temporally bound hostile proposition, visually grounded target/action, and acoustic carrier—not a smoothed high-score region. Each modality earns authority through exact within-video randomization, and an interval is emitted only when joint factual support exceeds every temporally misaligned null.

