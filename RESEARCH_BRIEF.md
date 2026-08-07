# Research Brief — Label-Free MLLM Hateful Video Detection (Follow-Up Project)

Date: 2026-08-08. Owner: Junyi Chen. This brief is the primary context for idea discovery. Detailed evidence lives in the referenced repo docs; this file stays under two pages.

## Problem statement

Detect hateful videos with a multimodal LLM using **zero human hate labels** at training, adaptation, and threshold-selection time. The deliverable is a method that is (a) mechanistically novel with an explainable, falsifiable story specific to hateful video, and (b) strong on held-out test splits across benchmarks. Paper writing is on hold; method discovery is the current phase.

## What is already established (do not re-derive)

**Confirmed mechanism — channel restoration** (the only mechanism to survive a pre-registered kill-test in two projects): hateful evidence is often carried by speech that the dataset transcript missed (ASR failure) or that a fixed transcript budget truncated. Restoring the channel (fresh Whisper large-v3, full length, degeneracy-gated) causally flips dismissed-hateful videos (flip asymmetry +0.486, p=1.3e-35) while leaving normals alone, with a clean three-way dissociation across starvation routes. Method form: **fresh full transcript → single frozen 8B judge call (Qwen3-VL-8B) → raw logit z = logsumexp(Yes)−logsumexp(No) → label-free KDE-valley threshold on the corpus's own z distribution**. One MLLM call per video.

**Held-out test results (all committed):**

| dataset | AUC | label-free macro-F1 | reading |
|---|---|---|---|
| ImpliHateVid (401) | 0.9473 | 0.8823 | method holds; supervised SOTA (IARE, SIGIR'26) is 91.75 F1 |
| HateMM (215) | 0.9232 | 0.6562 | ranking strong; valley picks a bad operating point (oracle gap 0.23) — its "normal" class is violence-saturated |
| MHClip-EN (161) | 0.7847 | 0.6962 | judge ceiling low; threshold blameless |
| MHClip-ZH (149) | 0.8547 | 0.7256 | restoration neutral — no starvation there (median transcript 76 chars), mechanism-consistent |

**Scale boundary condition:** the 2B judge gains ranking from restored evidence but its z distribution is near-unimodal (valley collapses; 2B label-free operating point unusable). Commitment bimodality emerges with scale and is what makes label-free thresholding work.

**Model behavioral law (three independent confirmations):** the 8B judge is **question-insensitive and evidence-sensitive**. Prompt-level re-readings (literal vs pragmatic; stance-directed probes) correlate ≥0.97 with the joint judgment and never beat effort-matched placebos; input-level evidence changes flip decisions causally. Any new mechanism must act on the **input/evidence side**, not on how the question is asked.

## Hard constraints (violating any of these disqualifies an idea)

1. **≤2 MLLM calls per video**, each with a distinct named role; single-pass preferred. No ensembling, no self-consistency sampling, no multi-prompt pooling, no CoT-sample-and-vote.
2. **No external hate-specific resources**: no auxiliary hate datasets, no hate lexicons, no retrieval over hate examples. General-purpose pretrained models (CLIP, Whisper, the MLLM itself) are fine.
3. **Label-free end to end**: no labeled dev set may tune anything, including thresholds.
4. Every idea needs the 4-part story: phenomenon (specific to hateful video) → mechanism (falsifiable) → prediction (with a disconfirming outcome) → counterfactual ablation.
5. **Falsified designs may not be rerun** (see docs/analysis/prior_falsification_map.md for all 22): observe-then-judge text cascades; target×stance products or boolean conjunctions; post-hoc fusion of deliberately narrowed calls (loses to the model's own cross-modal attention); per-rule readouts; self-assessed confidence gating at any level (text, logprob, activation probe); prompt-difference statistics that cannot beat a matched placebo; boundary-rescue second opinions; multi-model judging.
6. Published no-fly zones: MARS (adversarial dual-stance), MATCH-HVD (evidence agents + judge), Pro-Cap (attribute-probe captioning), LoReHM (re-ask with disclosed prior), ALARM (confidence + self-improvement + retrieval).

## Where the open performance headroom is (the error budget)

- **FP half (largest)**: surface-evidence/stance misattribution — the judge fires on hate-adjacent surface features in non-hateful context (IHV: 79% of FPs cue-carrying vs 53% of TNs, p=3e-7; HateMM: 70 FPs, all high-z, on violent-but-not-hateful normals that also bend the valley). Prompt-side fixes are dead (see behavioral law); an input-side fix is unexplored.
- **Judge-ceiling datasets**: MHClip-EN AUC 0.78 with threshold blameless — evidence needed by the judge is plausibly not reaching it (candidate channels: on-screen text/OCR at 91-token frame resolution; temporal sparsity under 16 uniform frames).
- **Residual FN**: videos with no transcribable speech (7 of 8 confirmed no-speech residuals stay FN) — evidence must come from a non-speech channel if at all.
- **Threshold geometry**: the KDE valley assumes the z-distribution's two modes align with the class boundary; on corpora whose normal class shares surface features with hate (HateMM), the geometric valley and the label boundary diverge. A label-free operating-point principle robust to this is open.

## Resources and constraints

Single RTX 5090 (currently occupied by the owner's own runs — **no GPU pilots without explicit scheduling**). All four benchmarks local with frames_16 + fresh Whisper transcripts + raw mp4s on B2. A 5th resource: HateClipSeg (395 videos, 11,714 segment-level annotations with timestamps, no splits) — usable as analysis-side ground truth for temporal-sparsity phenomenon validation only; its bundled hate lexicon is quarantined. Frozen instruments (scorer, extractor, gate, threshold recipe) must not be modified; new work wraps them.

## Non-goals

Supervised or few-shot methods; methods whose gain is inference-compute scaling; benchmark-specific prompt engineering; paper writing (on hold).
