# Label-free philosophy map vs this project's measured dead ends (2026-08-10)

Owner request: survey how the literature does label-free training in ANY
domain, and judge which philosophies could generalize here. Two parallel
scans (foundational families; LLM/VLM-era families), synthesized against
the project's preregistered deaths. Sources cited in the scan outputs;
key IDs inline.

## Organizing principle

Every label-free method replaces human labels with a substitute signal.
There are only ~7 distinct substitute signals in the literature. Judge
each against our measured facts.

| Substitute signal | Representative families | Status here |
|---|---|---|
| 1. Model's own confidence/posterior | pseudo-labeling (FixMatch, Noisy Student), entropy min (TENT), TTA, self-rewarding, TTRL/intrinsic-reward RL | **DEAD, measured.** Posterior saturated (82% entropy≈0): entropy TTA had no gradient; extreme-pseudo-label heads +0.04. Self-reward additionally needs generation diversity a binary judge lacks. |
| 2. Invariance under input augmentation | consistency reg (Mean Teacher, UDA), contrastive (SimCLR/MoCo), negative-free (BYOL/DINO), MI-max (CPC) | **EXCLUDED by constraint + principle.** Augmentation = input modification (owner preprocessing veto), and no label-preserving augmentation family is definable for hate (a crop/paraphrase can flip irony/target). |
| 3. Corpus geometry (clusters, kNN graphs, density, low-density boundary) | DeepCluster/SwAV/SCAN, label propagation, TSVM | **HALF-DEAD.** Our KDE valley IS low-density separation (works where annotation boundary = model construct, provably non-universal). Corpus PCA ≡ z (0.992). Published law 2312.10029: unsupervised structure finds the most PROMINENT feature — a graph propagates topic/speaker unless the embedding's dominant axis is the task axis. Cluster-Norm (2407.18712) untried. Cheap precondition test exists: are activation neighborhoods organized by harm or by topic? |
| 4. Internal logical consistency of activations | CCS (2212.03827), ICM (2506.10139) mutual predictability | **MOSTLY DEAD, one untried corner.** CCS-family = prominence law (matches our PCA≡z death). ICM differs: it scores a LABELING OF THE WHOLE CORPUS by mutual predictability + coherence, not per-item confidence — survives saturation by construction. Untried here. Must run on video-conditioned activations (not text pairs — bridge fractured) with cluster-norm hygiene. |
| 5. Predict withheld parts of the input itself | pretext tasks, masked prediction (BERT/MAE), next-token, TTT (1909.13231), DAPT (2004.10964) | **GENUINELY UNTRIED.** Next-token loss on the corpus's own videos+transcripts is label-free, needs no posterior gradient, no text→video bridge, no augmentation. Known risk (stated in advance): DAPT reliably moves representations without moving decision boundaries — the judge may become fluent about the domain and classify identically. Cheap to test. |
| 6. Cross-modal agreement/disagreement within one input | audio-visual correspondence (1705.08168), cross-modal agreement (2004.12943), redundancy/uniqueness/synergy formalism (2306.04539) | **GENUINELY UNTRIED as a training signal.** Adjacency disclosed: PP-v4 modality-split CALLS fused at score level died; round-2 jury killed cross-modal CCA as input-restructuring. The untried version: modality agreement as a TRAINING OBJECTIVE inside one frozen-architecture model, single unchanged call at inference. Phenomenon-matched: benign-visual/hostile-audio juxtaposition IS a hateful-video mechanism. Failure mode: agreement objectives collapse to what streams trivially share (scene/speaker identity); mismatch regime is exactly where hateful juxtaposition lives — cuts both ways. |
| 7. Text-defined concept as activation-space training target | RepE (2310.01405), circuit breakers (2406.04313) | **EXCLUDED by our own measurement.** The entire line assumes text-elicited directions align with real-input activations; we falsified that twice (spec displacement cos 0.03; illocution probe 1.0 on text → chance/inverted on video). |
| — | distillation / cross-model supervision | **VETOED by owner.** |
| — | weak supervision voting (Snorkel 1605.07723, LLM-era 2205.02318) | **BLOCKED practically:** with one model, labeling functions must be structurally independent (different modalities/layers), else correlated errors break denoising; prompt-variant LFs also hit the call cap and question-insensitivity. |
| — | depth dynamics as uncertainty (logit-lens trajectories) | **MEASUREMENT-ONLY for now:** saturated final posterior does not imply saturated trajectory (crystallization layer is a free unsaturated scalar), but no label-free training target defined; 2507.06722 warns trajectories may carry no signal. |

## Convergent conclusion (both scans, independently)

Three philosophies are alive AND untried in this project. All three read
the signal from the DATA or the CORPUS-LEVEL structure instead of from
the posterior, which is why the saturation death does not touch them:

1. **Within-video cross-modal agreement as training signal** — the only
   family whose substitute signal is itself a hateful-video phenomenon
   (juxtaposition). Engineering risk: extracting two separable
   modality readouts from one frozen stack; collapse-to-trivial-shared
   risk; adjacency to two prior deaths must be confronted in any prereg.
2. **DAPT / next-token continued pretraining on the target corpus** —
   cheapest, cleanest legality, sidesteps all three measured dead ends by
   construction; honest risk is "moves representations, not decisions";
   high-information either way.
3. **ICM-style corpus-labeling coherence (with cluster-norm)** — the one
   elicitation objective that survives saturation; faces the prominence
   law head-on and must carry its published defenses.

Precondition probe relevant to 1 and 3 (CPU, cheap): test whether
activation-space neighborhoods of the stored video states are organized
by harm or by topic/corpus nuisance — this decides whether any
graph/cluster-based signal can work before spending GPU.
