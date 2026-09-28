# Codex consultation on the mechanism, 2026-09-28

- Asked by the user: is the current performance close to what this paradigm can reach, and can the mechanism be made
  more interesting?
- Model: Codex, gpt-6-astra (`~/.codex/config.toml`), read-only sandbox.
- The sandbox could not run shell commands (`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`), so Codex
  did not read the repository. It received:
  - a brief of the task, the method `r6_bma` and today's headroom results;
  - the full text of `research-wiki/DIRECTIONS.md`;
  - the final-run ablations (`experiments/20260926_twolevel/README.md` §20.1);
  - excerpts of `docs/reviews/20260927_final_novelty_review.md` §0 and §6.3;
  - the gate rules of `RESEARCH_ITERATION_RULES.md` (8, 9, 13, 14).
- The check of its first proposal is in `experiments/20260928_headroom/README.md`, section "Codex consultation".

## Questions

1. Is the current performance close to the ceiling of this paradigm (MLLM reads plus a probabilistic time model
   outside the MLLM)? Is the ceiling in the reading or in the combination? What holes are there in the headroom
   analysis?
2. Propose 2–4 ranked mechanism redesigns. For each, give:
   - the component it changes;
   - why it can add information;
   - the closest past attempt;
   - its cost;
   - the cheapest falsification test.
3. Which ideas are rejected because DIRECTIONS.md shows they were tried?

## First reply (before the documents were pasted): holes in the analysis

- **The supervised comparison is asymmetric.** The fitted models hold videos out by cross-validation, but r6 was
  selected by development on the same test corpora. Cross-validation does not remove that selection advantage.
- **Ridge and boosted trees failing is not an information bound.** Small samples, compressed features and objectives
  that do not match pooled PR or within-video ranking can hide usable information.
- **Loader match of the ensemble comparison.** Codex first raised this. It withdrew the point once told that the
  reference is the 8B read through the same old loader.
- **Oracle gains show metric headroom, not information that can be recovered.**
- **The verdict is placed before the window questions.** This couples global and local evidence, so their agreement
  may partly come from the conditioning.

## Second reply (verbatim)

1. **Q1: probably close to the practical limit of the current scalar reads; not demonstrably close to the paradigm's ceiling.**

   The failed supervised combinations support diminishing returns from M2–M4 redesigns using the same inputs. They do **not** establish that those inputs contain no further usable information: the sample is small, the fitted model classes are limited, and development selection favours r6. The matched-loader ensemble result strengthens the diagnosis that **different measurements can supply additional information**. I withdraw the loader-confounding objection. [Source: `experiments/20260928_headroom/README.md`.]

   Distinguish three conclusions:

   - **Temporal coupling is necessary:** removing it costs .114/.057 within.
   - **The particular duration machinery is not yet necessary on both corpora:** replacing it with Gaussian smoothing costs .063/.006; only HateMM clears .01.
   - **Some sophistication is dispensable:** removing normal scores improves within by about .011 on both; key calibration does not clear the novelty gate. Consequently, "the combination is saturated" should not become "every component is justified." [Source: `experiments/20260926_twolevel/README.md` §20.1.]

   Three remaining holes matter:

   - **Representation versus readout:** failures on scalar reads do not settle what a different readout could extract from the backbone. Earlier hidden-state probes had acknowledged implementation problems. This is an unresolved opportunity, not evidence that a usable label-free direction exists. [Source: `research-wiki/DIRECTIONS.md` A2, "真标签上限"; B, "读出瓶颈".]
   - **Metric/objective mismatch:** check whether the supervised diagnostics explicitly model temporal sequences and optimise video-balanced localisation. A video classifier cannot test hate-share ordering adequately; an independent-window classifier cannot test structured decoding adequately. The supplied summary does not establish those comparisons.
   - **Oracle scope:** perfect video labels and hate-share ordering demonstrate metric headroom, not recoverable information. An additional useful diagnostic would preserve the actual 8-second observation support while supplying oracle window values, separating temporal-resolution limitations from semantic-reading errors. [Existing oracle results: `experiments/20260928_headroom/README.md`.]

   The stance turn's +.008/+.012 does not establish improved local reasoning, especially given the reported broad score shift. Treat it as conditioning whose mechanism remains uncertain. [Sources: `research-wiki/DIRECTIONS.md` A1; `docs/reviews/20260927_final_novelty_review.md` §6.3.]

2. **Q2, rank 1: replace clock-based persistence with persistence supported by observed discourse structure.**

   **Mechanism.** Redesign **M3**, retaining M1 measurements and M4. Replace the explicit-duration chains with inference over connections between windows: neighbouring windows receive strong coupling when their transcript evidence indicates continuation, weak coupling across a discourse break, and selected nonadjacent windows can connect when the same utterance or explicitly attributed quotation continues. Construct connections from timestamped transcript structure and a frozen text representation—not generated hate hypotheses or predicted hate scores.

   The current simplification is that proximity and duration determine persistence, although a speaker's continuing argument and an unrelated neighbouring statement need different treatment. This proposal replaces the persistence mechanism; it is not an extra correction after r6.

   **Why finding 2 does not exclude it.** The additional input is discourse continuity, not another transformation of the existing logits. The claim would be narrow and testable: *independently observed continuity determines where evidence should propagate*. It does not claim that discourse structure resolves endorsement by itself.

   **Closest precedents and difference.** This is close to the untested speaker-target binding proposal, and dangerously close to AMBI/MOSAIC/COHERE, whose content-based relations added nothing beyond temporal controls. RAVEL also failed through generated relation errors. The distinction is that relations control latent-state coupling directly and come from observable transcript structure, without generated semantic slots. That distinction is insufficient for novelty unless the content controls lose. [Source: `research-wiki/DIRECTIONS.md` C.] Graphical inference itself is established machinery; a new graph is not automatically a new contribution.

   **Cost.** Zero extra MLLM calls. Reuse current reads, transcripts and timestamps. Budget roughly **0.05–0.3 additional GPU seconds/video** for one small text encoder, plus short CPU inference; approximately **17–100 GPU seconds across 333 videos**. These are planning estimates, not measurements. Exclude new diarization initially.

   **Cheapest cached falsification.** Implement one fixed transcript-based coupling rule using the existing reads. Compare against temporal-only connections and shuffled transcript connections matched for edge count and temporal distance. Evaluate all three official metrics.

   **Kill it** if genuine connections do not improve within by ≥.01 on **both** corpora over the matched controls, or if pooled metrics regress beyond the permitted floor. Do not rescue failure with progressively elaborate relation extraction. This directly addresses the repository's repeated finding that content-free controls explain apparent gains. [Source: `research-wiki/DIRECTIONS.md` D.3.]

3. **Q2, rank 2: replace window verdicts plus smoothing with a dense reader trained from one frozen teacher and known temporal correspondences.**

   **Mechanism.** Redesign **M1–M3 together**. Use the current single-reader pipeline offline to provide soft interval targets on unlabeled videos. Train a compact temporal reader over timestamped transcript representations and existing visual features. Present temporally cropped or resampled views with known coordinate mappings; require corresponding moments to receive consistent predictions. At deployment, this reader directly produces dense scores, replacing repeated window questions and duration inference.

   The current simplification is treating one scalar per modality per eight seconds as the complete representation available for localisation. Here the reader sees temporally indexed content and learns a reusable mapping.

   **Why it could work—and why it might not.** Dense inputs contain information discarded by window logits, so finding 2 does not directly bound this model. However, teacher supervision cannot manufacture correct boundaries. Temporal consistency can preserve a wrong prediction, and a student may merely reproduce the teacher's smooth curves. The proposed mechanism must demonstrate that correspondence training contributes beyond ordinary distillation.

   **Closest precedents and difference.** Sparse-to-dense distillation is explicitly listed as proposed but untested. SDL/NGA instead adapted the MLLM using its own verdict-derived supervision; their failures make another pseudo-label loop unattractive. This proposal freezes the teacher, uses soft temporal targets, and trains a separate dense reader with correspondence supervision. [Source: `research-wiki/DIRECTIONS.md` A2 and C.] Distillation and temporal consistency are established techniques; the potentially publishable result is that they replace the external duration model while improving localisation.

   **Rule 14e.** Use **one teacher**, not the eight-model average, even during training. The deployed method is one dense reader. An ensemble teacher would retain an ensemble dependency and should not be presented as avoiding the rule.

   **Cost.** Zero additional MLLM calls on new videos; potentially zero MLLM calls altogether at deployment. Existing teacher reads and r6 outputs are reusable. Budget **2–8 RTX 5090 GPU-hours** for an initial small-reader training run, conditional on existing feature availability. A transcript/20-frame-feature reader might take **0.1–0.5 seconds/video**, excluding ASR and feature extraction; those extraction costs must be measured and included before claiming savings.

   **Cheapest cached falsification.** Train a small CPU prototype from cached transcript features and single-teacher soft targets, holding out videos; use GT only for evaluation. Compare ordinary distillation, temporal-correspondence training, and shuffled-correspondence training.

   **Kill this version** if it cannot retain r6 performance within the stated floors, or if correspondence training contributes <.01 within on either corpus. A negative transcript-only pilot does not prove a multimodal version impossible, but it removes the justification for an expensive implementation without further evidence.

   **For both proposals:** these are research hypotheses, not certified novel methods. Require the meaningful matched-control ablation, not merely removal of the entire temporal component. Also, preservation plus an ablation is insufficient for promotion: the written gate additionally requires ≥.01 improvement on both corpora. [Source: `RESEARCH_ITERATION_RULES.md`, promotion rule and 14g.]

4. **Q3: directions I would reject now.**

   - More semantic questions, speech-act labels or target subtraction: prior attempts mostly reproduce the same topic axis. [`research-wiki/DIRECTIONS.md`, A2/B/D.]
   - Generated explanations, evidence quotations or state sequences fed back into the reader: confirmation bias and continuation errors already dominate. [`research-wiki/DIRECTIONS.md`, A2.]
   - Pairwise window ranking: no meaningful advantage over existing score differences. [`research-wiki/DIRECTIONS.md`, A2, PWC.]
   - Deletion attribution, generic counterfactual masking or shifted grids: existing failures undermine another lightly modified version. [`research-wiki/DIRECTIONS.md`, A2/C.]
   - Further unsupervised duration fitting or probabilistic video-level recombination: repeated failures, with no new information source. [`research-wiki/DIRECTIONS.md`, A2.]
   - More frames or simply a larger reader: observed benefits do not justify their cost or clear both corpora. [`research-wiki/DIRECTIONS.md`, A2; `experiments/20260928_headroom/README.md`.]
   - Multi-reader averaging as the claimed method: informative diagnostic, but expensive and forbidden by the current rule. [`experiments/20260928_headroom/README.md`; `RESEARCH_ITERATION_RULES.md` 14e.]
