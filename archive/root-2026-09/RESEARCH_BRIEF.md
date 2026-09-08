# Research Brief — Label-Free MLLM Hateful Video Detection (Follow-Up Project)

Date: 2026-08-10 (rev 3, post idea-discovery round 2 + owner rulings). Owner: Junyi Chen. This brief is the primary context for idea discovery. Detailed evidence lives in docs/duplex/ and docs/analysis/; this file stays under two pages.

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

**Round-3 pilot death (2026-08-10, preregistered):** illocution instrument transfer — speech act (author-committed vs attributed) is PERFECTLY linearly decodable from the judge's hidden states on text evidence (LOFO AUROC 1.0000, both languages, typography-invariant), but the direction transfers to video-evidence states at chance (HateMM 0.483) or inverted (MHClip-ZH 0.334). **Text-derived activation geometry does not transfer to video-evidence states.** Kills the entire researcher-authored-instrument family (round-3 candidates 1, 2, 4, 10, 8's text selector) and heavily discounts all policy-text-subspace designs (5, 6, 7). See ILLOCUTION_TRANSFER_NOTE.md.

**Round-3 pilot death 2 (2026-08-10, preregistered, owner-proposed):** speech-act second call — asking the judge directly "does the author assert or only quote/report the hostile content?" returns the hate scalar renamed (in-pool Spearman with z 0.936/0.918 = the dual-axis fingerprint; residualized AUROC chance or inverted; the model affirms author-endorsement for quoted hate). Verbal access dead alongside probe access: the assert/mention conflation is end-to-end in video processing, not a readout artifact. See SPEECHACT_SECOND_CALL_NOTE.md.

**Round-2 pilot deaths (2026-08-09, preregistered):** spec-displacement readout — hidden-state displacement induced by swapping the spec text is construct-orthogonal (cos ≈ 0.03 with the supervised construct direction); forced-answer contrast — PC1 of (forced-Yes − forced-No) contrast states reproduces z (0.983), no new axis. Scale-emergence measured and closed (2B unimodal → 4B+ bimodal commitment; construct probes 0.92–0.95 at all scales; 4B valley win on HateMM is a lucky cut that does not generalize, 4-corpus oracle mean 0.8017 < TRIAGE 0.808).

**Owner rulings (binding):** (1) Channel restoration is ruled a *preprocessing* contribution — it may carry the pipeline but cannot be the paper's mechanism claim. (2) **Cross-model supervision is VETOED** (another LLM blind-codes unlabeled videos to supervise a hidden-state readout): distillation has no hateful-video mechanism story; it answers no phenomenon→mechanism question. Do not re-propose in any disguise (teacher–student, "one-time offline calibration", judge-of-judge). (3) Negative-results/finding-shaped paper framing rejected; the deliverable is a working novel method.

## Anchor phenomenon for round 3 (the one replicated, model-owned failure)

**Assertion-vs-mention confusion.** The judge scores surface hostile content regardless of speech act: MHClip-ZH titles carrying the harvester's offensive query keyword push benign videos up by ≈8 logits (median z +2.25 vs −6.5 for plain normals); 11/70 HateMM valley false positives are quoted/reported hate; MHClip-EN counter-speech/critique videos score as hateful. Moderation targets *asserted/endorsed* hostility; web video is saturated with *mentioned* hostility (quoted, reported, sung, criticized). Same disease, three corpora. The illocution idea-cluster was parked by the round-2 jury (risk: role-instruction implementations collapse into prompt-side changes, which are question-insensitive). Any round-3 idea that fixes this must operate at the representation/mechanism level, not the prompt.

## Constraint-relaxation menu (the owner's live question — ideas should target these)

R1. **Second call with a genuinely new role** — the ≤2-call budget has an unspent slot; every falsified second-call design reused the same scalar-judgment paradigm. What role is NOT in the falsified family?
R2. **Label-free training/adaptation** — charter-legal (only human labels are banned; target-corpus unlabeled data explicitly allowed). Posterior objectives are dead (no gradient); representation-level or cross-modal objectives are open.
R3. **Readout wider than one scalar** — the representation provably holds more (probe 0.837); the open problem is label-free access. Cross-model distillation is VETOED (see owner rulings); five unsupervised access routes are dead (PCA, band-PCA, pseudo-label extremes, spec displacement, forced-answer contrast). Any new R3 idea must name an access mechanism outside these six.
R4. **Task-specification as first-class input** — the ill-posedness proof says the boundary must be injected. CAUTION: per-dataset definition PROMPTS are the base project's territory (policy conditioning); only structurally different uses (e.g., spec-conditioned readout geometry or spec-conditioned training) are candidate novelty.
R5. **Judge backbone** — currently pinned to Qwen3-VL-8B by execution constraint; relaxing it is engineering unless a scale/architecture phenomenon is the claim (the 2B boundary condition shows such phenomena exist).

## Hard constraints that stay

≤2 MLLM calls/video with distinct named roles; no ensembling/self-consistency; no external hate-specific resources (datasets, lexicons, retrieval); no human hate labels anywhere; 4-part story (phenomenon → mechanism → prediction → counterfactual) mandatory; falsified designs may not rerun; no-fly: MARS, MATCH-HVD, Pro-Cap, LoReHM, ALARM. **Owner veto: no more data-preprocessing-shaped contributions; the novelty must live in the model/mechanism layer.**

## Resources

Single RTX 5090, currently free. Five benchmarks local with frames_16, fresh transcripts, per-layer hidden states for all scored videos, raw mp4s on B2. HateClipSeg segment-level annotations available as analysis-side gold. Blind-audit corrected strata on disk for EN/HCS/ZH.
