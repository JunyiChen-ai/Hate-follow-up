# Idea-Generation Bundle v3 — Round 3, Post-Veto (2026-08-10)

You are a senior ML researcher brainstorming research ideas for a label-free MLLM hateful-video detection project. Context: RESEARCH_BRIEF.md rev 3 (summarized below — the file itself cannot be attached).

## Project state in six facts

1. Base pipeline (frozen): fresh Whisper transcript + 16 frames + title → ONE frozen Qwen3-VL-8B-Instruct call → scalar z = logsumexp(Yes)−logsumexp(No) → label-free KDE-valley threshold. Five test corpora local (ImpliHateVid, HateMM, MHClip-EN, MHClip-ZH, HateClipSeg) with per-layer hidden states (37×4096, final prompt token) stored for every scored video. Held-out AUC 0.75–0.95; valley macro-F1 0.66–0.88. Labeled-oracle mean 0.8175 vs base project TRIAGE 0.808 — threshold perfection is worth only ~1 point. Single RTX 5090.
2. Behavioral laws (preregistered): question-insensitive/evidence-sensitive (prompt rewording changes nothing, ≥0.97); answer posterior saturated on 82% of videos (entropy≈0 → NO gradient for any posterior objective); layer-27 hidden states hold construct info the scalar discards (supervised probe 0.837 where z scores 0.321) but SIX unsupervised access routes are dead: corpus PCA (≡z, 0.992), band-conditioned PCA, extreme-pseudo-label heads (+0.04), spec-text displacement direction (construct-orthogonal, cos≈0.03), forced-answer contrast (PC1≡z 0.983), entropy TTA (no gradient). 2B judge lacks commitment bimodality; transition happens between 2B and 4B; 4B thresholdability is corpus-fragile.
3. Label-free thresholding is PROVEN ill-posed w.r.t. annotation scheme (same scores, two label collapses, optimal rule flips). The task specification must be injected somewhere.
4. Error attribution complete: no model-side blind spot survives controls EXCEPT one replicated real failure — **assertion-vs-mention confusion**: the judge scores surface hostile content regardless of speech act. Evidence: MHClip-ZH benign videos whose titles embed the harvester's offensive query keyword get z pushed up ~8 logits; 11/70 HateMM valley false positives are quoted/reported hate (news, critique); MHClip-EN counter-speech scores as hateful. Moderation targets ASSERTED hostility; web video is full of MENTIONED hostility (quoted/reported/sung/criticized). Three corpora, same disease.
5. OWNER VETOES (binding, do not violate): (a) no data-preprocessing-shaped contributions (input restructuring, evidence routing, transcript editing); (b) no prompt engineering (question-insensitivity makes it dead anyway); (c) NO cross-model supervision in any disguise (no other LLM supplies labels/supervision/calibration, no teacher–student); (d) no ensembling/self-consistency; (e) ≤2 MLLM calls per video, each a distinct named role; (f) no external hate-specific datasets/lexicons/retrieval; (g) no human hate labels anywhere; (h) negative-results paper framing rejected — deliverable is a working method beating/matching TRIAGE (avg macro-F1 0.808).
6. Every idea needs the 4-part story: named hateful-video phenomenon → falsifiable mechanism → advance prediction incl. disconfirmer → counterfactual ablation targeting the story.

## Fresh literature (verified 2026-08-10; [V]=abstract read, [S]=snippet only)

**Assertion-vs-mention / speech act:**
- 2404.01651 NAACL24 [V]: LMs fail use-mention distinction; counter-speech gets flagged; fix = prompting, text-only. Phenomenon named, mechanism unexamined. THE paper to beat.
- 2602.22787 [V]: whether a completion draws on contextual evidence vs parametric memory is linearly decodable (0.96) with a SELF-SUPERVISED label pipeline; transfers zero-shot.
- 2608.03035 [V]: "contextual truth of propositions" is a linear, causally steerable direction; a partner's assertion shifts internal representation.
- 2507.23221 [S]: single faithfulness direction; its lowest-activating extreme is QUOTED forum content — incidental evidence a quotation axis exists.
- 2605.31349 FBHM [V]: hateful memes, learnable steering vectors, +30 macro-F1 — but needs 500 LABELS. 2505.17760 JUSSA [V]: steering the judge at evaluation time is a legitimate published framing.
- Cautions: 2606.02907 probes may read surface format (quote marks) not illocution — minimal pairs must control lexical form; 2509.13450 steering entanglement — must show intervention isn't just globally lowering the hate logit.
- GAPS: use-mention failure in MULTIMODAL/VIDEO hate detection: unshown. Label-free representation-level fix: none anywhere. Use-mention direction in hidden states: never probed.

**Label-free representation access / spec-as-activation-object:**
- 2601.08139 SubTTA [V]: per-modality principal SUBSPACES aligned by chordal/Grassmann distance — CLIP only. Our dead spec test used a single residual direction; multi-dim spec SUBSPACE, attention-level injection (2605.10664), and input-conditioned steering FIELDS (gradient of learned scoring function) are all untried.
- 2410.12877 ICLR25 [S]: instruction vector = activation diff with/without instruction — used only to steer generation of surface constraints, never as a readout for judgment.
- 2606.05713 [V]: discriminative hidden-state readout from generative omni-modal LLM beats generative decoding — but SUPERVISED. Label-free version unclaimed.
- SAEs: 2606.16193 CSAE [V] trains SAEs on Qwen3-VL (feasible, published); Qwen-Scope [V] releases SAEs for text Qwen3 only (no VL — we'd train our own on unlabeled activations, 5090-feasible); 2506.01247 VS2 [V]: label-free SAE steering on frozen CLIP, names the reconstruction-vs-task-saliency gap; 2503.09446 [S]: SAE as zero-shot concept classifier by locating features for a NAMED CONCEPT; 2502.11367 [S]: SAE features solve toxicity classification but feature selection is supervised. 2606.29888 [S] warning: same concept sits in different SAE directions per modality.
- CCS family: 2312.10029 [S]: unsupervised consistency finds the most PROMINENT feature, not the target (this is a published law matching our PCA≡z failure); 2407.18712 [S]: cluster-normalization removes the prominent nuisance direction before CCS — cheap, untried on our states.
- GAPS: "written policy/spec → activation-space READOUT object (subspace/probe) on a frozen generative VLM judge, no labels" unclaimed. "SAE-widened label-free moderation readout on a VLM, features selected by activation on spec text" unclaimed.

**Competitors (niche state):** LELA 2602.09637 [V] and MARS 2601.15115 [V]: training-free hateful-video via multi-stage prompting, MANY calls/video. TANDEM 2601.11178 [V]: supervised RL, 0.73 F1 target-ID HateMM. NO competitor operates in activation space; none is single-pass. Our lane is uncontested.

## Task

Generate 8–12 concrete research ideas at the MECHANISM level. Prioritize these three anchors but do not limit to them:

(A) **The assertion-vs-mention direction.** Locate a speech-act (asserted vs mentioned hostility) direction/subspace in the frozen judge's activations WITHOUT labels and WITHOUT another model's supervision, and use it to correct the judgment. Key design question: how to manufacture the defining contrast label-free. Legal raw materials: neutral template text authored once by the researcher (like ActAdd contrast prompts — instrument construction, not corpus preprocessing); the corpus's own unlabeled videos; the judge's own activations. Illegal: rewriting corpus transcripts per-video at inference (preprocessing veto), another LLM coding videos (cross-model veto). Note the surface-format confound (quote marks) must be controlled by construction.

(B) **Spec → readout geometry, round 2.** The single-direction version is dead (construct-orthogonal). Untried: multi-dimensional spec SUBSPACE with principal-angle alignment; attention-head-level injection; input-conditioned steering field. The ill-posedness proof gives this family its story: the annotation boundary MUST be injected, and prompts can't do it (question-insensitivity).

(C) **SAE-widened readout.** Train an SAE on the judge's own layer-27 activations over unlabeled corpus videos (5090-feasible, precedent CSAE), then select features label-free (e.g., by activation on spec text / researcher-authored construct descriptions). Addresses VS2's saliency gap with a spec-driven selector. Cross-modal feature-split warning applies.

For each idea give:
1. One-sentence summary
2. 4-part story: phenomenon / mechanism (falsifiable) / advance prediction WITH disconfirmer / counterfactual ablation
3. Minimum viable experiment on our five local corpora (hidden states on disk; pilots ≤2 GPU-hours; CPU-only preferred where possible)
4. Why it is NOT in the dead family (check against fact 2's six dead access routes and the vetoes in fact 5)
5. Where the TRIAGE-beating points come from (which corpora, which error class)
6. Risk LOW/MED/HIGH + effort days/weeks
7. Contribution type: method / finding / theory

The owner is a demanding senior researcher who rejects anything smelling of preprocessing, prompting, ensembling, or distillation. Novelty must live in the model/mechanism layer. Be creative but concrete.
