# Independent proposal review: Allocator

Date: 2026-10-03. Reviewer: independent agent `/root/m1_grounder_proposal_review`, same model as the parent session. **Decision: STOP under rule 4, first condition: prior target-task use of the source SHAP attribution mechanism.**

Reviewed declaration: `experiments/20261003_m1_allocator/README.md`, including its revisions for fixed background media, historical modality games, and the 100–200 GPU-minute estimate. This is the single proposal review. No Allocator implementation was written or run; no current visual-contrast performance or GT was read. The reviewer did not edit the declaration. The result is a repository-rule decision, not a claim that the proposed time-window game exactly duplicates a published implementation.

## Basis for STOP

The primary article [A comprehensive framework for multi-modal hate speech detection in social media using deep learning, Scientific Reports, 2025](https://www.nature.com/articles/s41598-025-94069-z), also available in [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12000576/), places SHAP in its proposed method, not merely its related work. Its “Contribution 1” and pipeline describe video frames, audio and text for hate/non-hate classification. “Contribution 3” explicitly uses SHAP to assess the contribution of features across modalities.

This is published target-task method use of the attribution family being transferred. Rule 4 covers hateful-video **detection or localization** and does not require the previous use to be a primary localization score. A new temporal player definition, an MLLM value function and Kernel SHAP estimation do not restore the declaration that the source mechanism has never been used in the target task.

**Evidence limitation:** that article gives little SHAP implementation detail, and its dataset table lists predominantly text datasets despite its multimodal narrative. I have not verified its computations or results, nor established use of temporal coalitions, actual subset re-encoding or the exact Kernel SHAP estimator. The decision relies on its explicit published method description. It must not be reported as a verified implementation of Allocator.

| Rule-4 ground | Decision |
|---|---|
| Source method already used in hateful-video detection/localization | **Triggered by the published SHAP application above.** |
| Pure ensemble | Not triggered: one frozen MLLM evaluates the subsets. |
| Pure calibration/postprocessing/smoothing | Not triggered: new raw-media subset evaluations produce information unavailable from saved native scores alone. Attribution is post-hoc explanation in the literature's terminology, but this is not merely postprocessing the existing output curve. |
| Pure engineering | Not independently triggered: the declared coalition estimand is a substantive change from grand-coalition leave-one-out. The source-use condition suffices for STOP. |

Expected accuracy, cost, approximation error and possible context-removal failures are not STOP grounds used here. No implementation or GPU experiment is required to substantiate this decision. Archive this declaration without launching it.

## Adjacent prior art and an access limitation resolved

- [Hate Speech Detection in Audio Using SHAP – An Explainable AI, ANTIC proceedings, 2024](https://link.springer.com/chapter/10.1007/978-3-031-64064-3_21): the publisher's abstract explicitly describes English/Kiswahili YouTube videos converted to audio, followed by SHAP explanations of audio-feature classifiers. Only the primary abstract was accessible. This is an audio application using video-derived material; it is not claimed here to implement joint audiovisual temporal localization and is not the sole STOP evidence.
- [Text or Image? What is More Important in Cross-Domain Generalization Capabilities of Hate Meme Detection Models?, Findings of EACL 2024](https://aclanthology.org/2024.findings-eacl.8/), [PDF](https://aclanthology.org/2024.findings-eacl.8.pdf), Section 3: Shapley values are computed for text tokens and image patches and summarized by modality. The task is static hateful memes. It establishes adjacent hate-attribution precedent, not the video STOP condition by itself.
- [An Interpretable Agentic Framework for Multimodal Hate Video Analysis with Explicit Evidence Attribution, WWW Companion 2026](https://doi.org/10.1145/3774905.3796488): previous reviews lacked full method access. This review found the [author's Zenodo software archive](https://zenodo.org/records/18208988), downloaded its `hvd-main.zip` into memory and inspected the README, evidence ranker and orchestrator source without executing them. `src/orchestrator/evidence_ranker.py` uses specified rule weights for extracted evidence, cross-modal agreement/conflict and temporal overlap; `llm_orchestrator.py` provides heuristic or optional LLM aggregation. The inspected released code does not perform coalition re-encoding or Shapley allocation. The complete published paper was not recovered; code inspection narrows, but does not eliminate, that access limitation.

The search was therefore not restricted to the term Kernel SHAP or to temporal localization titles. It distinguished actual method use, post-hoc explanation, static memes and audio from incidental mentions of videos.

## Source-method check

[A Unified Approach to Interpreting Model Predictions, NeurIPS 2017](https://arxiv.org/pdf/1705.07874), Theorem 2, gives the Shapley kernel and the endpoint constraints. The declaration uses that constrained weighted-regression construction. Its explicit physical deletion game is not the original paper's conditional-expectation definition of missing features; call these Shapley estimates **of the declared removal game**, not a uniquely correct or causal explanation of the original video.

[Play Fair: Frame Attributions in Video Models](https://arxiv.org/abs/2011.12372), [author repository](https://github.com/willprice/play-fair), already attributes video class scores to temporal elements using Shapley values in action recognition. It supports the temporal-allocation source lineage, not a new temporal-Shapley algorithm claim. Its ESV approximation and feature-based action models are not identical to the proposed Kernel SHAP estimator and raw-media MLLM calls.

[Explaining by Removing, JMLR 2021](https://jmlr.org/papers/v22/20-1316.html) distinguishes removal semantics, the explained output and the influence summary. Those distinctions explain why Eraser and Allocator need not be equivalent while remaining established removal-attribution methods.

## Comparison with archived project methods

The following code and declaration were inspected; none was executed:

| Prior method | Actual difference from Allocator |
|---|---|
| `archive/experiments/20261003_m1_eraser/README.md` | Each local effect removes one modality-window from the otherwise full input. Allocator's player is a joint window, and its estimand averages marginal contributions over coalitions rather than only the full coalition. |
| `archive/experiments/idea_discovery-2026-08/idea_discovery/run_shared_coalition_game.py` | The three players are already extracted visual/language/audio evidence views. Its eight values come from one LESS evidence model; no raw temporal-media subsets are re-encoded. |
| `archive/experiments/idea_discovery-2026-08/idea_discovery/run_interaction_coalition_game.py` | Its eight modality coalitions are marginals of an empirical-Bayes joint evidence model. It does not allocate a freshly evaluated global MLLM response across time windows. |

Accordingly, this review does **not** classify Allocator as a fully repeated Eraser experiment or a mere rerun of the August scripts. The OR-game example correctly distinguishes the estimands: two independently sufficient events can have zero full-coalition removal effect and nonzero Shapley shares. That mathematical example does not diagnose why Eraser failed on actual videos. The README now states both boundaries correctly.

## Sampling, cost and remaining wording checks

These findings record the requested proposal check; they do not authorize implementation after STOP.

1. **The proposed interior weights are consistent.** For interior subsets, sampling the size with mass proportional to `1/[s(M-s)]` and then sampling uniformly within that size gives subset probability proportional to the Shapley kernel. Consequently `W_int/64` per sampled row estimates the interior weighted objective without another kernel multiplier. Complement pairing preserves this marginal distribution but gives 32 dependent pairs, not 64 independent observations. Repeated masks must retain their combined statistical weight even when their value call is cached.
2. **Boundary weights and rank are correct.** A singleton or its complement has kernel weight `1/M`. With all singleton rows included, the constrained least-squares problem has full rank on the efficiency constraint space. For exhaustive coalitions, this recovers the game Shapley values in exact arithmetic. An unbiased sampled objective does not imply that its finite-sample minimizer is an unbiased Shapley estimator.
3. **Axiom tests need their stated scope.** Efficiency is imposed exactly. Structurally empty windows are excluded and assigned zero. Exhaustive small-game tests can check full symmetry/dummy properties; arbitrary finite sampled games need not preserve those properties exactly. Sampling error should not be misdiagnosed as an implementation bug or hidden by claiming exact Shapley values.
4. **The fixed-background correction resolves the reconstruction conflict.** Unassigned original media remain in every coalition. Thus `v(empty)` means the background-only value, not necessarily a no-transcript value. The allocation sums to `v(full)-v(background)`, not the full margin, and does not assign a share to background content. This is compatible with exact native reconstruction. Partial ASR segments retain coarse original timestamps, so this is not new word-level alignment.
5. **Joint players do not separately measure cross-modal synergy.** Their value can depend on both modalities and other windows, but allocating their combined contribution does not identify an interaction between vision and speech. Counts/availability prose also changes with subset reconstruction and belongs to the defined intervention. The declaration acknowledges both limits.
6. **The control question needs literal specification if ever revisited.** “Native joint query” should identify an existing exact prompt or predeclare its wording and conditioning context. It is otherwise insufficient to reconstruct the control from the two existing modality-specific queries. No such clarification rescues the present source-transfer novelty claim.
7. **Cost is counted as real inference.** The bound `U<=2M+66` for `M>6` is consistent with all singletons/complements, 64 interior rows and two endpoints; deduplication can reduce calls. Allocator deployment requires `2U` outer forwards with a fresh media prefix per unique subset. Paired dual/joint controls cost additional calls. The latest declaration gives at most 33,030 subset reads and 100–200 GPU minutes, based on the parent's no-GT input audit and earlier timings. I checked the formula and updated declaration, not the audit's raw records; runtime remains an estimate.

The proposed joint-query, joint leave-one-out, singleton and shuffled-allocation controls would distinguish multiple explanations if this were an eligible experiment. Efficiency and regression fit alone cannot show correct hateful-span localization or redundancy handling. None of these possible outcomes is being predicted by this review.

## Search record

Actual searches included the following query families, followed by the primary-source checks above:

```text
"hateful video" "Shapley"
"HateMM" "SHAP"
"hateful video" "coalition"
"hate video" "Shapley"
"HateMM" Shapley attribution
"hate speech" video "SHAP"
"hateful video" attribution Shapley explanation
"HateClipSeg" Shapley
"multimodal hate" "Shapley" video
"video hate" "SHAP" detection
"hateful" "video" "KernelSHAP"
"An Interpretable Agentic Framework" "Shapley"
"An Interpretable Agentic Framework" "Multimodal Hate" pdf
"hate speech detection" "video" "SHAP"
"hateful video" "Shapley values"
"multimodal" "hate" "video" "coalition attribution"
"Kiswahili" "SHapley" "video"
"Kiswahili" "Random Forest" "95.8%" hate speech
"A comprehensive framework for multi-modal hate speech" "SHAP" data
"Hate Speech Detection in Audio Using SHAP" pdf
"Play Fair: Frame Attributions in Video Models" arxiv.org
"Explaining by Removing" JMLR
```

**Final disposition: STOP and archive without implementation.** The reason is prior published use of the source attribution mechanism in the target task under rule 4. It is not an empirical rejection of coalition allocation, a claim of exact duplication, or a requirement to run more experiments before deciding.
