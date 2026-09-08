# Idea-Generation Bundle v2 — Post-Attribution Phase (2026-08-09)

You are a senior ML researcher brainstorming research ideas. Mandatory context file: `/home/jehc223/Hate-follow-up/RESEARCH_BRIEF.md` (rev 2) — read it first; it contains the confirmed mechanism, four behavioral laws of the frozen judge, the five-corpus error attribution, 25 falsified design families, and the constraint-relaxation menu R1–R5. Do not propose anything on the falsified list.

## What changed since the last brainstorm (v1, three days ago)

Eight further preregistered kill tests all FAILED; the five-corpus error attribution completed. Hard new facts your ideas must respect:

1. The judge's answer posterior is saturated on 82% of videos (entropy ≈ 0) — any label-free objective on the answer posterior has no gradient.
2. Layer-27 hidden states contain construct information the scalar readout discards (supervised probe 0.837 where the scalar z scores 0.321) — but unsupervised PCA access fails (usable axes ≡ z, rest noise) and extreme-pseudo-label access yields only +0.04.
3. Construct-swapped questions do not create new measurement axes (Spearman 0.935 with the original, protected-target ranking bit-identical).
4. Label-free thresholding is PROVEN ill-posed w.r.t. annotation scheme (same scores, two label collapses, optimal rule flips). The boundary must be injected as a specification.
5. No model-side blind spot survives controls; remaining benchmark gaps are label artifacts. The judge, evidence-restored, is at the label-quality ceiling in the single-scalar-readout regime.
6. The owner VETOES data-preprocessing-shaped ideas (input restructuring, evidence routing). Novelty must live in the model/mechanism layer.

## Fresh literature landscape (verified this morning; [V] = abstract fetched, [S] = search-result only)

R1 (second call, new roles): LatentMAS 2511.20639 [S] — training-free hidden-state+KV handoff between agent calls, ICML26 spotlight. When Does an LM Commit 2605.06723 [V] — δ=S(yes)−S(no) theorized as "finite-answer projection", stabilizes pre-verbalization; text-only, single 4B model. Speculative verification via hidden-state classifiers 2510.02329 [S]. Counterfactual re-execution agents 2606.08275 [S]. GAP: nobody uses a second call to MANUFACTURE THE CONTRAST that defines a readout direction (CCS-style contrast pairs joined with latent handoff); nothing in moderation.

R2 (label-free adaptation, non-posterior objectives): CLIPTTA 2507.14312 NeurIPS25 [S] — entropy minimization is misaligned with contrastive pretraining, use soft contrastive loss; whole line is CLIP dual-encoder only. Survey 2508.05547 [V]. Subspace-alignment TTA 2601.08139 [S]. GAP: no representation-level corpus-adaptation objective exists for a GENERATIVE MLLM judge.

R3 (repE/steering/readout for safety): NExT-Guard 2603.02219 [V] — pretrained SAEs as training-free text-safety readout. Hidden-state discriminative readout from omni-modal LLM 2606.05713 [V] (supervised). FBHM steering vectors for hateful memes 2605.31349 [V] (supervised, 500 samples). SAEs on Qwen3-VL exist 2606.16193 [S]; SAE features transfer to toxicity zero-shot 2502.11367 [S]. Probe-OOD-failure counter-result 2509.03888 [S]. GAP: label-free location of a moderation construct direction in a VLM's activations is unclaimed.

R4 (spec-conditioned computation): Compliance2LoRA 2607.27594 [V] — policy text → hypernetwork → LoRA weights, generation-side compliance only, 2 weeks old. Text-to-LoRA 2606.06492 [S]. GMP benchmark 2603.01724 [V] — moderation degrades under dynamic rules (external support for ill-posedness). SafeWatch/OmniGuard/HaloGuard: prompt-level policy conditioning. GAP: spec-conditioned JUDGMENT READOUT GEOMETRY (the spec rotates/selects the decision direction on a frozen judge) is unclaimed. Time-critical.

R5 (scale phenomena): commitment-geometry line is text-only, single-scale (2605.06723 [V], 2607.12447 [V], 2606.29490 [V]); alignment collapses response diversity 2603.24124 [V]. Our 2B-no-bimodality / 8B-bimodality emergence is unpublished. GAP: scale-dependent emergence of judgment commitment geometry in VLMs is claimable.

Competitors: LELA 2602.09637 [V] — first training-free LLM hate-video localization, multi-stage prompting cascade, evaluated on HateMM+MultiHateClip (blows our ≤2-call cap; localization task). SafeLens 2605.17610 [V] — trained fast-and-slow video guardrail. Label-free meme self-improvement (KDD26) 2512.21598 [V]. The single-call label-free operating-point niche is still open but the neighborhood is filling.

## Task

Generate 8–12 concrete research ideas at the MECHANISM level. For each:
1. One-sentence summary
2. Core hypothesis (falsifiable; grounded in the phenomenon→mechanism discipline; must state what result would DISCONFIRM it)
3. Minimum viable experiment (cheapest decisive test on: ImpliHateVid/HateMM/MHClip-EN/ZH/HateClipSeg, all local with frames+audio+fresh transcripts+per-layer hidden states for every scored video; one RTX 5090; pilots ≤2 GPU-hours)
4. Which constraint it relaxes (R1–R5) and why the relaxation is principled, not a budget increase
5. Contribution type: new method / empirical finding / theory
6. Risk LOW/MED/HIGH and effort days/weeks
7. Why it is NOT in the falsified family (check against the brief's list — especially: no posterior-based objectives, no confidence gating, no observe-then-judge, no same-construct re-asking, no input restructuring)

Prioritize: (a) the R1×R3 empty cell (second call manufactures the contrast that defines a hidden-state readout direction — e.g., judge the video vs judge a model-constructed counterfactual/neutralized variant, read the direction between the two hidden states); (b) R4 spec-conditioned readout geometry (the ill-posedness proof demands the spec as input; make the spec act on the READOUT, not the prompt); (c) R5 scale-emergence of commitment geometry as a companion scientific claim. Ideas must not lose to TRIAGE (avg macro-F1 80.8 at 1.73 calls/video) on performance plausibility — say where the points come from.

Be creative but grounded. The owner is a demanding senior researcher who will reject anything that smells like data preprocessing, prompt engineering, or an ensemble in disguise.
