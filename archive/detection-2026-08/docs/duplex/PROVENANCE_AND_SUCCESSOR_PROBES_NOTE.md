# Provenance extraction and successor probes

**Date:** 2026-08-08. This note records the continuous goal-directed search
started by the speaker-provenance request.

## 1. Speaker-provenance extraction v1 — quality gate failed

The frozen probe is in `PREREG_speaker_provenance_probe.md`. WavLM speaker
verification windows plus YuNet visible-face evidence processed all 90 videos:

- 90/90 nonempty attributed transcripts;
- 83/90 classified as two-speaker;
- 840/840 ASR chunks assigned;
- 90/90 videos had visual-role observations.

The deterministic 20-video spot check nevertheless exceeded the maximum four
gross ownership failures. Long single-speaker monologues were repeatedly split
between PRIMARY and SECONDARY because channel, music, or delivery changes
formed acoustic clusters. These tags are not reliable provenance. Per the
quality gate, no MLLM judge calls were made. This fails the extractor, not the
provenance mechanism, which remains untested.

## 2. Generic-harm nuisance — signal exists, proposed mechanisms fail

Frozen SigLIP visual generic-harm scores separate HateMM original-valley FP
from TP at AUC **0.6685**, passing the signal bar. The preregistered global
linear residual does nothing on HateMM because the unlabeled OLS slope is
negative, and harms ImpliHateVid AUC by **0.0388**.

The predeclared 2-D successor raises HateMM macro-F1 from **0.6429 to 0.8108**,
but fails the novelty mechanism checks:

- raw-z AUC declines by 0.0139;
- z-only GMM already reaches 0.7974;
- diagonal 2-D reaches 0.8046;
- shuffled-g reaches 0.7934;
- full covariance beats the diagonal by only 0.0063, not the frozen 0.03.

The apparent performance is mostly an ordinary mixture operating point, not a
load-bearing generic-harm interaction. Do not promote it as a method.

## 3. Non-lexical delivery/prosody — no signal

The frozen SUPERB emotion encoder covers 89/90 videos. Mean angry posterior:

- AUC(TP vs FP): **0.5092**;
- marker-positive AUC(TP vs FP): **0.5022**;
- AUC(TP vs TN): **0.5138**;
- residual-prosody AUC(TP vs FP), after removing raw-z association: **0.5299**;
- duration placebo: 0.5368.

The channel is effectively random for the required distinction. No delivery
tags or MLLM calls are licensed.

## Decision and immediate next step

Generic visual harm is diagnostically real but did not produce a novel
load-bearing decision mechanism. Generic emotion/prosody is closed. Speaker
provenance remains the only candidate in this sequence whose hypothesized
information was never actually supplied to the judge.

The next implementation must replace ad-hoc per-video WavLM clustering with a
standard end-to-end diarization model that explicitly models speaker turns and
temporal continuity, then rerun the already frozen extraction-quality gate on
untouched videos. Only after that gate passes should the P/S/R judge arms run.

## 4. Standard-diarizer provenance probe — extraction passes, mechanism fails

The successor used NVIDIA's pretrained streaming Sortformer
(`nvidia/diar_streaming_sortformer_4spk-v2.1`) on the same 90 WAVs. It produced
39 one-speaker and 51 multi-speaker videos. Of 841 ASR chunks, 729 (86.7%)
received a speaker assignment. The same deterministic transcript/turn spot
check no longer showed v1's systematic monologue splitting, so the frozen P/S/R
judge arms ran. This establishes fitness for this probe, not diarization DER.

All arms used identical 16 frames, uncapped words, binary question, raw-z
readout, and Qwen3-VL-8B judge. Machine-readable results are in
`results/speaker_provenance/results.json`.

| TP-vs-FP AUC | F | P | S | R |
|---|---:|---:|---:|---:|
| value | .8728 | .8578 | .8489 | .8472 |

FP median P-F was +0.25 and no FP crossed down at the frozen threshold. TP
median P-F was -0.125. P-F AUC was -.0150 (95% paired-bootstrap CI [-.0689,
+.0389]); P-S was +.0089 and P-R +.0106. TN-vs-FP stayed at AUC 0 in every arm.

**Verdict: FAIL (one of seven clauses passes).** Once extraction quality is no
longer the limiting factor, literal speaker/onscreen tags remain non-load-
bearing evidence for the holistic judge. This closes tag injection, not merely
the first clustering implementation.

## 5. Next method-directed probe (corrected against the falsification map)

Do **not** pursue sparse-frame restoration. The completed HateClipSeg B1 audit
already kills that family: the exact uniform-16 sampler hits at least one
hate-bearing segment in 96.8% of offensive-union videos and 96.7% of
hateful-strict videos. Equal-budget shot covering is worse by 6--7.7 percentage
points. A change-point sampler would rerun a falsified premise.

The remaining visual information-restoration candidate is spatial, not
temporal: uniform-16 usually captures the relevant moment, but the frozen judge
receives only about 91 visual tokens per frame, potentially making burned-in
text illegible. The next gate is therefore the already ranked A2 OCR census on
native-resolution MHClip-EN frames. OCR is used only to measure text-box
prevalence and, if the gate passes, route a fixed pixel/token budget toward
text-bearing regions. Recognized strings never enter the prompt. Matched random
and anti-targeted pixel allocation are the causal controls.
