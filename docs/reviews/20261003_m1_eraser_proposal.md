# Independent proposal review: M1 Eraser

Date: 2026-10-03. Reviewer: independent same-model agent `/root/m1_grounder_proposal_review`.
Scope: one rule-4 review of `experiments/20261003_m1_eraser/README.md`, before implementation. No method code, declaration, predictions or GT were changed or generated. Attributor's pending performance was not inspected.

## Decision: PASS

None of the four permitted STOP conditions is established by the literature search. The candidate applies established input-occlusion / leave-one-out attribution to temporal, modality-specific parts of a hateful video's observed input. The necessary computation is a fresh model response after actual removal, not a request to imagine ignoring content or a transformation of existing output scores.

This is permission to implement and test under the repository's transfer criterion, not certification of publication novelty. Neither deletion attribution nor re-encoding after changing an input is new in general. The proposed contribution can only concern an effective transfer to this task, supported by the declared experiments.

| Rule-4 STOP condition | Finding |
|---|---|
| Source mechanism already used in hateful video detection/localization | Not established. I searched actual input occlusion, temporal erasure, LOO and perturbation attribution, including post-hoc explanation. Located target-task papers use rationales, retrieval, learned temporal scoring, global modality ablations or label-based temporal cropping; the checked sources did not use the input-removal response as per-input temporal evidence contribution. Access limits are recorded below. |
| Pure ensemble | No. Repeated evaluations of one frozen model under input interventions are not multiple independent models under rule 3. |
| Pure calibration / postprocessing / smoothing | No. New re-encoded inputs produce new global reads. Subtracting the unchanged global score cannot explain within-video reordering; the README correctly assigns the scientific question to the changed-input reads. |
| Pure engineering trick | No. Actual temporal input removal defines a different evidence estimator from the current local question. Prompt wording, encoder, resolution and decoder are fixed. Rebuilding dependent states is necessary implementation of that intervention, not a separate novelty claim. |

## Actual literature search and primary readings

Searches on 2026-10-03 combined `hateful video`, `hate video`, `video hate speech`, `HateMM` and `MultiHateLoc` with `occlusion`, `occlusion sensitivity`, `leave-one-out`, `erasure`, `temporal erasure`, `deletion`, `removal`, `masking`, `perturbation`, `counterfactual`, `attribution`, `LIME` and `SHAP`. Search results were followed to primary papers; unrelated keyword hits and bibliography citations were not treated as method applications.

The search scope included trained detectors and post-hoc explanations, not only training-free localization or the exact proposed difference formula.

### Established source methods outside the target task

- [Zeiler and Fergus, Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901): established input-occlusion sensitivity. Eraser must credit this family rather than claim deletion attribution as a new mechanism.
- [AttriBoT: A Bag of Tricks for Efficiently Approximating Leave-One-Out Context Attribution](https://arxiv.org/html/2411.15102v1), §2.2: attributes an LLM response through its likelihood change when a context span is removed. The paper studies context attribution, including open-book question answering. Eraser uses this established LOO principle; it does not adopt AttriBoT's full acceleration system or establish those efficiency claims.
- [Adaptive Occlusion Sensitivity Analysis for Visually Explaining Video Recognition Networks](https://arxiv.org/abs/2207.12859): already extends occlusion to video with spatiotemporal masks and output-score changes, evaluated on UCF101 and Kinetics. Therefore “first temporal/video occlusion attribution” is unavailable. This is not evidence of application to hateful-video detection.
- [Seeing Through VisualBERT: A Causal Adventure on Memetic Landscapes](https://arxiv.org/abs/2410.13488): relevant interpretability and causal-model work on offensive memes, not temporal hateful video. The search also surfaced occlusion-based analyses of hate memes. These constrain broad hate-content attribution claims but do not establish this repository's video-task STOP condition.

### Target-task sources and the relevant distinction

- [LELA](https://arxiv.org/html/2602.09637v1), §3.3–3.4: generates per-frame modality scores from captions and combines them with a maximum. This is direct local scoring, not input-removal attribution.
- [MultiHateLoc](https://arxiv.org/html/2512.10408v1), §3: trained modality-specific temporal representations, cross-modal interactions and MIL scoring. The checked method does not obtain temporal importance by re-evaluating a fixed predictor after input deletion.
- [MATCH, author-hosted full manuscript](https://jianlang.org/papers/MATCH.pdf), §III-C: retrieves spatiotemporal evidence for a verifier to check generated clues. Its ablations remove the evidence supply or use random evidence; these do not assign input contributions by per-unit removal-induced prediction changes. This is a close explainable hateful-video precedent, not evidence for the proposed source mechanism.
- [LEAF](https://aclanthology.org/2026.findings-acl.604.pdf), especially §4.5, and [IARE](https://arxiv.org/html/2606.11953v1), §4 and §6: generate and evaluate explanations using reasoning, grounding and training procedures. No input-occlusion attribution method was established from their checked accounts.
- [SAGE](https://aclanthology.org/2026.acl-long.817.pdf), Table 4, tests missing entire modalities; [HVGuard](https://aclanthology.org/2025.emnlp-main.456.pdf), Table 3, ablates modality/components. These demonstrate target-task input/component removal for aggregate robustness and ablation analysis. They do not by themselves establish per-example occlusion attribution. Ordinary accuracy changes under an ablation must not automatically be described as the same source attribution method.
- [Revealing Temporal Label Noise in Multimodal Hateful Video Classification](https://arxiv.org/html/2508.04900v1), §3 and §5: physically trims videos using annotated hateful spans, then compares full/trimmed training and testing settings. This is a relevant warning about context dependence. It does not derive temporal importance by removing each candidate span and measuring that video's predictor response.
- [CLARA](https://arxiv.org/html/2608.15905v1): clip-level modeling with VLM-derived rationales was checked as adjacent temporal evidence modeling; no temporal input-occlusion attribution procedure was established.

The SAGE, LEAF, HVGuard and MATCH PDFs were read through text extraction, including searches for occlusion, perturbation, erasure, deletion, masking, removal, Shapley, LIME, comprehensiveness and sufficiency, followed by inspection of matched passages. A missing keyword is not itself evidence of absence; the methodological distinctions above are the basis for the assessment.

### Access boundary

[An Interpretable Agentic Framework for Multimodal Hate Video Analysis with Explicit Evidence Attribution](https://doi.org/10.1145/3774905.3796488), WWW Companion 2026, is directly relevant. Its publisher-indexed abstract describes extracted evidence, deterministic ranking, agreement/conflict modeling and optional LLM reasoning. Full text remains unavailable in the access attempts documented by the preceding Attributor review; this round recovered the abstract again, not the full method. I cannot exclude an unobserved input-deletion analysis there. The abstract establishes prior explicit evidence attribution in hateful video, so a broad first-attribution claim is prohibited. It does not establish the specific STOP condition. This bounded search finding must not be represented as exhaustive proof of novelty.

## Relation to local experiments

`archive/experiments/20260912_cva/README.md` and the relevant `cva.py` scoring path were read. CVA constructs one full-media prefix, then asks exclusion questions against the same cache. It does not remove the alleged interval from input. Its negative result is relevant prior evidence but not a completed test of actual input occlusion.

Attributor gates contextualized cached values, leaving other previously computed state available. Eraser removes observed frames or transcript words and rebuilds dependent states. These interventions are substantively different. Neither their common attribution family nor Attributor's unknown outcome establishes a duplicate completed experiment. No previously completed exact Eraser run was identified in the reviewed repository records. The grounder/selector failures and Marginalizer's STOP are not reasons to block this candidate.

## Necessary precision and falsifiable diagnostics

The declared formal experiment can falsify the mechanism; retain the density, time-misalignment, shift-only and hypothetical-exclusion comparisons if a qualifying gain occurs. No speculative performance risk below is used as a STOP condition.

1. **Define the attributed output.** State explicitly that `F` is the native global Yes-minus-No logit margin. Positive erasure effects support the model's hate margin; negative effects oppose it. This is not attribution of whichever class happened to win, nor a posterior probability or a causal ground truth annotation.
2. **Verify input removal, including metadata.** Record removed frame paths/timestamps and ASR word indices, retained content, actual model token counts before/after, and fresh-prefill checks. Avoid stale image placeholders, altered timestamps or global answer turns carrying the original verdict into the altered input. Keep the original global question. This is the substantive distinction from CVA and cached-value interventions.
3. **Report the zero floor explicitly.** A frameless window has visual attribution zero by construction. With the declared maximum, a negative speech effect then becomes a raw score of zero. Report how often this happens and its effect on ties/ranking; do not credit this floor as discovered visual evidence. If a gain is concentrated there, evaluate the corresponding missing-frame zero assignment with native scores as a diagnostic control before claiming the erasure mechanism caused it. This can use saved native reads.
4. **Separate content sensitivity from deletion quantity and formatting.** Retain the frame/word-density controls and quantity-stratified results; model-token change is useful alongside word count. Actual deletion also changes sequence positions and can join nonadjacent words. Time rotation proves alignment dependence but alone does not rule out an alignment-dependent content-quantity effect. Case inspection should preserve and show the actual altered transcript, not only the original.
5. **Do not overclaim multimodal reasoning.** Removing one modality while retaining the other can reveal a conjunction already represented by the MLLM, but the score does not identify an interaction term or guarantee detection of redundant evidence. Compare cases and signed responses. Neither a large removal effect nor deleting the highest-scored unit validates GT localization by itself.
6. **Keep the cost visible.** The declared `2 + 2E` deployment forward count and fresh full prefixes expose the main cost. Replace the 60–120 GPU-minute estimate with smoke-based measurements, report both corpora and peak memory, and count later control calls separately. No gain may be attributed to a cheaper cached intervention if the actual method re-encodes inputs.

Proceed to implementation and the single independent rule-6 code review. Performance, localization improvement and the claimed explanation of any gain remain open experimental questions. The paper/current method should remain unchanged until the existing empirical gates pass.
