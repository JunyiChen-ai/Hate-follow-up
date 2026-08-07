# Idea-Generation Bundle — Label-Free MLLM Hateful Video Detection (Follow-Up)

You are a senior ML researcher brainstorming research ideas. Read the two context files first — they are mandatory context and contain hard constraints:

1. `/home/jehc223/Hate-follow-up/RESEARCH_BRIEF.md` — the project state: confirmed mechanism, held-out results, error budget, hard constraints.
2. `/home/jehc223/Hate-follow-up/docs/analysis/prior_falsification_map.md` — 22 falsified designs with death causes; the "hard lessons" section is binding.

## Research direction

The project has ONE confirmed mechanism: **channel restoration** — hateful evidence is often carried by speech that dataset transcripts missed or truncated; restoring the audio channel (fresh Whisper transcript, full length, degeneracy-gated) into a **frozen single-call MLLM judge** (Qwen3-VL-8B, raw logit z readout) causally flips missed hateful videos while leaving normals alone, and a **label-free threshold** (KDE valley of the corpus's own z distribution) matches the labeled oracle. Held-out test: ImpliHateVid macro-F1 0.8823 / AUC 0.9473 (supervised SOTA 91.75 F1).

We need the NEXT mechanism, same discipline: novel, label-free, mechanistically explainable, strong performance. The known behavioral law of the judge (three independent confirmations): **question-insensitive, evidence-sensitive** — changing how you ask does nothing (prompt re-readings correlate ≥0.97 with the baseline judgment and never beat an effort-matched placebo); changing what evidence reaches the model flips decisions causally. So viable ideas act on the INPUT/EVIDENCE side of a frozen judge, or on the score-distribution/readout geometry — not on prompting strategy, not on multi-call reasoning.

## Landscape summary (verified papers)

- Training-free competitor: MARS (2601.15115) multi-stage adversarial reasoning (breaks our ≤2-call budget; its design family — adversarial dual-stance — is a no-fly zone).
- MM-HSD (2508.20546): supervised fusion showing **on-screen text** carries hate cues other modalities miss.
- Failures to Surface Harmful Contents in VideoLLMs (2508.10974): sparse uniform frame sampling + spatial downsampling + fusion imbalance make VideoLLMs omit clearly-visible harmful content. Diagnosis only, no method, nothing label-free.
- Temporal Label Noise in Hateful Video Classification (2508.04900): hateful videos contain long non-hateful stretches (HateMM); supervised label-trimming remedy only.
- MultiHateLoc (2512.10408): weakly-supervised temporal localisation of hate segments (occupies localisation-as-task).
- VisualTextTrap (2604.17375): VLMs treat rendered on-screen text as privileged and let it override visual evidence (hazard for naive OCR injection).
- MME-VideoOCR (2505.21333): MLLM OCR degrades badly on video.
- Keyframe-selection line (AKS 2502.21271, LENS 2607.25125, FOCUS, KeyVideoLLM): query-driven frame selection for QA; none moderation-specific, none label-free.
- Label-free operating-point geometry for judge score distributions: unoccupied.

## Our located error budget (where performance is to be won)

- HateMM: judge ranking strong (AUC 0.923) but label-free valley picks a bad operating point (oracle gap 0.23) — its "normal" class is saturated with violent-but-not-hateful content, which both inflates FPs and bends the score distribution so the geometric valley diverges from the class boundary.
- MHClip-EN: judge ceiling itself low (AUC 0.78, threshold blameless) — evidence plausibly not reaching the judge (on-screen text at 91-token/frame resolution? temporal sparsity under 16 uniform frames?).
- Residual FNs: videos with no transcribable speech — evidence must come from a non-speech channel if at all.
- FP half everywhere: surface-cue misattribution (mention/quote/report/counter-speech read as assertion). Prompt-side fixes are DEAD (behavioral law). Input-side fixes unexplored.

## Hard constraints (violating any disqualifies an idea)

1. ≤2 MLLM calls/video, distinct named roles; single-pass preferred. No ensembling/self-consistency/multi-prompt pooling.
2. No external hate-specific resources (datasets, lexicons, retrieval over hate examples). General-purpose pretrained models fine (CLIP, Whisper, OCR engines, the MLLM itself).
3. Label-free end to end, including thresholds.
4. Each idea must have: phenomenon (specific to hateful video) → mechanism (falsifiable) → prediction (with a disconfirming outcome) → counterfactual ablation.
5. Do not rerun falsified designs (see map): observe-then-judge cascades; target×stance products/conjunctions; post-hoc fusion of narrowed calls; per-rule readouts; confidence gating (text/logprob/probe); prompt-difference statistics; boundary-rescue second opinions; multi-model judging.
6. Published no-fly zones: MARS, MATCH-HVD, Pro-Cap, LoReHM, ALARM.
7. Compute: one RTX 5090; pilot ≤2 GPU-hours; full runs ≤ a few GPU-days.

## Task

Generate **8–12 concrete research ideas**. For each:
1. One-sentence summary
2. Core hypothesis (what you expect and why — grounded in the phenomenon/mechanism discipline)
3. Minimum viable experiment (cheapest decisive test; specify data from: ImpliHateVid/HateMM/MHClip-EN/ZH with frames+audio+fresh transcripts local, plus HateClipSeg segment-level annotations usable as analysis-side gold only)
4. Contribution type: empirical finding / new method / theoretical result / diagnostic
5. Risk: LOW / MEDIUM / HIGH
6. Estimated effort: days / weeks / months

Prioritize ideas that: are testable on one GPU; produce a decisive answer either way; are NOT "apply X to Y" without a hateful-video-specific reason; are differentiated from every paper above; and extend the confirmed evidence-side family or the score-geometry family. Be creative but grounded — the best ideas here will make the judge SEE evidence it currently misses, or make the label-free operating point robust where it currently bends.
