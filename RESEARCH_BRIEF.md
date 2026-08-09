# Research Brief — Label-Free MLLM Hateful Video Detection (Follow-Up Project)

Date: 2026-08-09 (rev 2, post attribution program). Owner: Junyi Chen. This brief is the primary context for idea discovery. Detailed evidence lives in docs/duplex/ and docs/analysis/; this file stays under two pages.

## Problem statement

Detect hateful videos with a multimodal LLM using **zero human hate labels** at training, adaptation, and threshold-selection time. The deliverable is a method that is (a) **mechanistically novel** — the owner explicitly rejects further data-preprocessing-shaped contributions — and (b) strong on held-out test splits. Paper writing is on hold; method discovery is the current phase. The owner's standing question: **which constraint should be relaxed to buy real novelty, and how?**

## Established base (do not re-derive)

**Confirmed mechanism — channel restoration**: fresh full Whisper transcript → single frozen Qwen3-VL-8B call → raw logit z = logsumexp(Yes)−logsumexp(No) → label-free KDE-valley threshold. Causal flip asymmetry +0.486 (p=1.3e-35), three-way dissociation. One call per video.

**Held-out test (5 corpora, committed):** ImpliHateVid AUC 0.947 / valley macro-F1 0.882; HateMM 0.923 / 0.656; MHClip-EN 0.785 / 0.696; MHClip-ZH 0.855 / 0.726; HateClipSeg 0.754 union / 0.771 strict. Labeled-oracle mean over the first four = 0.8175 — threshold selection, even solved perfectly, gains ≈1 point over the base project (TRIAGE 0.808 at 1.73 calls/video).

**Behavioral laws (all preregistered):** (1) question-insensitive, evidence-sensitive — same-construct rewordings correlate ≥0.97; construct-swapped questions decorrelate slightly (0.935) but measure nothing new. (2) Answer-posterior saturation: 82% of videos sit at entropy ≈ 0 — any label-free training objective on the answer posterior has no gradient. (3) Layer-27 hidden states DO hold construct information the scalar readout discards (supervised probe separates no-target-offense from protected-target-hate at 0.837 where z scores 0.321) — but unsupervised access failed (PCA axes = z renamed or noise) and extreme-pseudo-label access yields only +0.04 and needs two-sided saturation occupancy that 3 of 5 corpora lack. (4) 2B judge: no commitment bimodality — label-free operating point unusable below ~8B scale.

**Five-corpus error attribution (autopsies + blind audits, completed):** no model-side blind spot survives controls. HateClipSeg "normal" class: 36% contains protected-group hostility/extremist content (blind-confirmed; byte-identical duplicate transcripts with contradictory labels exist). MHClip-EN positives: 69% carry no protected-group target (blind-confirmed) — construct mixing. MHClip-ZH: 42% of titles embed the harvester's offensive query keyword; keyword-bearing normals score median z +2.25 vs −6.5 for plain normals — collection artifact, and simultaneously a real judge failure mode: **surface keyword in title triggers z despite benign content (mention treated as assertion)**. Gender-stereotype content: judge ranks it 0.970 where annotators call it hateful; low elsewhere because corpora file it as insulting — construct disagreement, not deficit.

**Ill-posedness (proved):** same corpus, same scores, two label collapses → the optimal label-free rule flips (valley 0.652 vs anchored 0.547 under union; 0.448 vs 0.677 under strict). The decision boundary is annotation-owned; no z-only rule can be universal. Task/policy specification is the missing input.

## Falsified families (17 prior + 8 new; docs/analysis/prior_falsification_map.md + docs/duplex/*_NOTE.md)

Prior: observe-then-judge cascades; target×stance products; narrowed-call fusion; per-rule readouts; confidence gating (all levels); prompt-difference statistics; boundary-rescue second opinions; multi-model judging. New (all preregistered FAILs, 2026-08-08/09): saturation-anchored threshold (unconditional + occupancy-gated); commitment-state semantic interpretation; text-region pixel-budget rerouting (text already legible); construct-swapped second axis (dual-axis); unsupervised hidden-state axes (PCA); entropy/posterior-objective adaptation (no gradient); extreme-pseudo-label readout refitting (+0.04, insufficient, occupancy-limited). Also dead: temporal frame coverage (uniform-16 covers 96.8%); speaker-provenance tag injection; prosody channel; generic-harm 2-D fusion; evidence-volume account of implicit-content deficits (wrong sign).

## Constraint-relaxation menu (the owner's live question — ideas should target these)

R1. **Second call with a genuinely new role** — the ≤2-call budget has an unspent slot; every falsified second-call design reused the same scalar-judgment paradigm. What role is NOT in the falsified family?
R2. **Label-free training/adaptation** — charter-legal (only human labels are banned; target-corpus unlabeled data explicitly allowed). Posterior objectives are dead (no gradient); representation-level or cross-modal objectives are open.
R3. **Readout wider than one scalar** — the representation provably holds more (probe 0.837); the open problem is label-free access. Cross-model distillation (a general-purpose LLM supplies the supervision signal once, offline) would relax "no supervision" to "no human/hate-specific supervision" — owner ruling needed on whether this stays label-free.
R4. **Task-specification as first-class input** — the ill-posedness proof says the boundary must be injected. CAUTION: per-dataset definition PROMPTS are the base project's territory (policy conditioning); only structurally different uses (e.g., spec-conditioned readout geometry or spec-conditioned training) are candidate novelty.
R5. **Judge backbone** — currently pinned to Qwen3-VL-8B by execution constraint; relaxing it is engineering unless a scale/architecture phenomenon is the claim (the 2B boundary condition shows such phenomena exist).

## Hard constraints that stay

≤2 MLLM calls/video with distinct named roles; no ensembling/self-consistency; no external hate-specific resources (datasets, lexicons, retrieval); no human hate labels anywhere; 4-part story (phenomenon → mechanism → prediction → counterfactual) mandatory; falsified designs may not rerun; no-fly: MARS, MATCH-HVD, Pro-Cap, LoReHM, ALARM. **Owner veto: no more data-preprocessing-shaped contributions; the novelty must live in the model/mechanism layer.**

## Resources

Single RTX 5090, currently free. Five benchmarks local with frames_16, fresh transcripts, per-layer hidden states for all scored videos, raw mp4s on B2. HateClipSeg segment-level annotations available as analysis-side gold. Blind-audit corrected strata on disk for EN/HCS/ZH.
