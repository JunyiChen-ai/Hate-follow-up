# Independent proposal review: M1 visual contrastive reading

Date: 2026-10-03. Reviewer: independent agent `/root/m1_grounder_proposal_review`, same model as the parent session. Decision: **PASS under RESEARCH_ITERATION_RULES.md rule 4**.

Reviewed declaration: `experiments/20261003_m1_visual_contrast/README.md`, including the clarification that identity checks use contrast coefficient zero and explicitly `abar=1`, not diffusion step zero. This is the single proposal review of this candidate. No implementation, run, GT, or pending Contraster performance was inspected. No method declaration or implementation was edited. PASS authorizes implementation and experiment; it does not establish effectiveness, promotion, or publication novelty.

## Decision against the four permitted STOP grounds

| Ground | Finding |
|---|---|
| Source method already used for hateful-video detection/localization | Not established by the search and primary sources inspected. A close **static multimodal safety** application already exists and must be acknowledged below. This limits the contribution substantially but does not establish the video-task STOP condition. |
| Pure ensemble | No. Clean and corrupted reads use the same frozen MLLM. There is no combination of multiple independent models; the source's use of “contrastive ensemble” terminology does not change the repository's definition. |
| Pure calibration/postprocessing/smoothing | No. The subtracted quantity requires a new image-dependent prefix encoding and new visual-window reads. It is not available as a transformation of the saved clean scores alone. Whether the benefit reduces empirically to rescaling or a video-wide offset remains an experimental question. |
| Pure engineering | No under the repository's transfer criterion. This transfers an existing input-intervention contrastive method and declares a falsifiable localization hypothesis. The changes are more than a prompt rewording, backbone swap, resolution change, or constant adjustment. The algorithm itself is not new. |

The previous candidates' outcomes are not grounds for stopping this candidate. Likewise, possible distribution shift, failure to improve ranking, or a constant corrupted response are not permitted proposal-stage STOP reasons.

## Source fidelity and the nearest prior application

### VCD

Primary sources inspected:

- [Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding](https://arxiv.org/html/2311.16922), especially Section 3.3 and Appendix A; [paper PDF](https://arxiv.org/pdf/2311.16922).
- [Authors' noise implementation, `vcd_add_noise.py`](https://raw.githubusercontent.com/DAMO-NLP-SG/VCD/master/vcd_utils/vcd_add_noise.py).

The source contrasts clean and distorted-image next-token logits as `(1+alpha)*clean-alpha*distorted`, with an adaptive plausibility constraint (APC). Its noise code uses 1,000 sigmoid-spaced beta values from the stated lower and upper bounds and a cumulative alpha product. The declaration's zero-based step 500 therefore includes factors indexed 0 through 500. The source uses step 500 for some evaluations and 999 for POPE; step 500 should not be described as a universal source setting.

The candidate makes two explicit departures: it contrasts aggregated Yes/No **class log-odds**, and it omits APC. These are accurately disclosed and must remain disclosed. CPU FP32 noise, a fixed per-video seed, and preserving repeated still-image temporal copies are additional model-specific choices, not a claim of bitwise source reproduction. The source evaluates image-language benchmarks; its limitations identify video as outside the reported evaluation.

Let `a=log(P_clean(Yes)/P_clean(No))` and `b=log(P_corrupt(Yes)/P_corrupt(No))`, aggregating the native token sets before taking these ratios. A binary distribution proportional to `P_clean(c)^2/P_corrupt(c)` has log-odds `2a-b`. Thus the declared class contrast is mathematically coherent. It is generally different from subtracting individual token logits and then applying class log-sum-exp. Avoiding token-wise ratios removes that particular route for rare spelling variants to dominate; it does not prove immunity to changes in class probability or confidence.

### MMSafeAware: essential adjacent prior art

Primary source: [Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs, ACL 2025](https://aclanthology.org/2025.acl-long.832/), [full paper](https://aclanthology.org/2025.acl-long.832.pdf). The full paper's Sections 3 and 4.3 and Table 6 were inspected.

This work already evaluates VCD for multimodal safety judgments, comparing original and noisy visual inputs. Its benchmark consists of 1,500 **image–prompt pairs**, with 29 safety scenarios; the SocietySafe category includes hate speech. This is a substantive prior application to harmful-content assessment, including hate, and cannot be dismissed because its main title says safety awareness. However, its reported input and evaluation unit is a static image–prompt pair, not a video; incidental mentions of videos do not constitute video evaluation.

Consequently, do not claim first use of VCD for harmful content, hate-related multimodal safety, or binary safety judgment. The remaining eligible claim under this repository's rule is effective transfer to hateful-video temporal localization, with evidence that the new visual-window reads contribute. Mere use of a video benchmark is not sufficient evidence for the proposed explanation of why it helps.

### Search co-occurrences that do not establish target-task use

[Visual Evidence Prompting Mitigates Hallucinations in Large Vision-Language Models, ACL 2025](https://aclanthology.org/2025.acl-long.205.pdf) was also checked in full text. It compares VCD on hallucination benchmarks and uses example images from Hateful Memes for open-world object/relation questions. That combination of terms does not establish application of VCD to hateful-video detection/localization, or by itself establish hate classification on those examples.

## Actual target-task search and access boundary

Queries executed in this review included:

```text
"hateful video" "visual contrastive decoding"
"hate video" "VCD"
"HateMM" "VCD"
"hateful" "visual contrastive decoding"
"hate video" "corrupted" visual
"hateful video" "noise" decoding
"HateClipSeg" "contrastive decoding"
"hateful video" "contrastive" "decoding"
"hate video" "contrastive" "noise"
"video hate speech" "visual contrastive"
"hateful" "VCD" decoding
"visual contrastive decoding" "meme"
"visual contrastive decoding" "hateful video detection"
"HateMM" "corruption"
"HateClipSeg" "noise"
"MMSafeAware" "video"
"hateful video" "distorted"
"hateful video" "Gaussian"
"hateful video" "clean" "corrupt"
"HateMM" "gaussian noise"
"VCD" "SafeWatch"
```

The target-task cross-check used the following primary method descriptions. LELA was reread during this review; the other listed descriptions had already been read in this review agent's preceding proposal searches and were retained as bounded prior-source coverage:

| Primary source | Mechanism inspected and relevance |
|---|---|
| [LELA](https://arxiv.org/html/2602.09637v1), Sections 3.2–3.4 | Converts modalities to language, combines descriptions with speech, and obtains rationale/modality hate scores. This is not clean/noisy-image contrastive inference. |
| [MARS](https://arxiv.org/html/2601.15115) | Adversarial hate/non-hate hypotheses concern reasoning prompts; they do not establish distorted-visual-input VCD. |
| [RAMF](https://arxiv.org/html/2512.02743v1) | Modality features and structured reasoning fusion. Related-work citations to contrastive decoding are not evidence that its method implements clean/noisy VCD. |
| [IARE](https://arxiv.org/html/2606.11953v1) | Its inspected SFT/DPO reasoning method is not the proposed input contrast. |
| [CLARA](https://arxiv.org/html/2608.15905v1) and [MultiHateLoc](https://arxiv.org/html/2512.10408v1) | Clip representations and learned contrastive/fusion mechanisms do not establish clean/corrupted test-time visual contrast. |
| [Revealing Temporal Label Noise](https://arxiv.org/html/2508.04900v1) | Temporal annotation noise and label-based trimming are different from pixel-noise contrastive inference. |

Access was not universal. The WWW 2026 explicit-evidence-attribution paper at [DOI 10.1145/3774905.3796488](https://doi.org/10.1145/3774905.3796488) remained unavailable in full text from preceding searches. A current search also surfaced *Learning from Incomplete Signals: Multi-View Knowledge Distillation for Robust Harmful Micro-Video Detection* at [the publisher](https://www.sciencedirect.com/science/article/pii/S1566253526006111), but full-page access failed; its title and search snippet are insufficient to adjudicate its full mechanism. Neither is treated as an inspected negative result. The conclusion is “no verified target-task use found in the accessible evidence,” not proof that none exists.

## Interpretation boundaries and necessary diagnostics

These items delimit claims and verify the declared experiment. They are not additional speculative proposal STOP rules.

1. **Identity controls are now correctly stated.** Diffusion step zero still has nonzero beta; setting the sampled epsilon to zero still attenuates the input by `sqrt(abar)`. Neither is a no-corruption identity. Use explicit `abar=1` to reproduce clean pixels and coefficient zero to reproduce the clean score. The latest README contains this correction.
2. **Verify the actual processor layout in implementation.** The parent reports inspecting the installed Qwen image processor: patch flattening places channel before temporal and spatial patch axes, and the still frame is copied along the temporal axis. Noise should be sampled once for each spatial pixel and expanded to both copies. This review did not independently execute that processor. Code review and smoke checks must verify equality of the repeated noise, unchanged text/grid, actual pixel changes, and deterministic reconstruction rather than infer them from a shape alone.
3. **Unchanged transcript tokens do not mean unchanged transcript hidden states.** Re-encoding corrupted images changes all subsequently image-dependent states. The native selected answer is held fixed as text, and the native speech scores are retained downstream, but the corrupted visual-query path still has full multimodal context. Do not describe the intervention as isolating a visual-only computational branch or removing all speech/verdict influence.
4. **Whole-prefix sensitivity is not temporal localization evidence by itself.** All sampled frames are corrupted. A window with no sampled local frame can still change through other frames and shared context. Report image coverage and behavior of those windows. A claim about temporally specific visual grounding requires the declared matched-noise own-window versus other-window intervention, with comparable affected input amounts. Such comparisons should reuse the same sampled noise field under different masks.
5. **The stated cheap controls can falsify simpler explanations.** Native, clean-only `2a`, within-video permutation of `b`, and an order-preserving common shift test whether subtraction and its window correspondence matter. Record changed assignments; equal-valued or single-window cases are not effective permutations. An additional zero-inference diagnostic is `2a_i-mean_j(b_j)` with native speech scores unchanged: this preserves a video-wide corrupted offset while removing its window variation. It is useful if the main run gains and is not a prerequisite for launching it.
6. **Report raw and decoded behavior together.** A visual rescaling can change which modality wins the maximum, while a video-level offset can change pooled ranking. Even fixed r6 code can produce different fitted states for different inputs. Therefore, improved final scores alone do not establish improved visual-window ordering. Preserve the declared raw/decoded results, modality dominance, verdict strata, and gain/loss cases on both corpora. A component must satisfy rule 14(g) before it is claimed as necessary novelty.

The controls are sufficient to make the central claim falsifiable: if the main method fails to improve, or the benefit is reproduced by clean-only scaling/video-wide shift or survives destroying window correspondence, the proposed window-dependent visual-evidence explanation is unsupported. That outcome should restrict or reject the mechanism claim rather than be renamed as semantic grounding.

## Cost and experiment scope

The declaration counts the new corrupted prefix and its global question/forced-answer extension, as well as the extra visual queries: native `3+B_visual+B_speech` outer forwards versus paired/deployed contrast `6+2*B_visual+B_speech`. Thus the increment is three outer forwards, including one full-prefix encoding, plus `B_visual` query forwards. There is no per-window prefix rebuild and no backward pass. Sequential cache use is a plausible memory-saving implementation, not free inference.

The 15–30 GPU-minute estimate for 333 videos is provisional and must be replaced by observed smoke/full-run time. Source-image caches and model weights can be reused; corrupted prefix states must actually be recomputed, with fresh position/rope state and no mutation of native inputs. Holding global scores fixed requires downstream provenance checks, not simply reusing the same answer string.

The declared paired full run, canonical evaluator, all three metrics on both corpora, and development-selected reporting remain the experiment gate. No pending Contraster scores were needed for this review. A supported outcome from an already-running candidate takes priority as declared by the parent.

## Required declaration correction and disposition

Before making a contribution claim, add MMSafeAware to the source discussion and explicitly exclude novelty claims for VCD-based static multimodal safety or hate assessment. Preserve the existing disclosures about class aggregation, omitted APC, and whole-prefix versus local intervention. The identity-test correction has already been incorporated.

**PASS.** Implement and evaluate the declared mechanism after the required code review. No one of the four rule-4 STOP conditions is established by the inspected evidence. Performance and the explanation for any improvement remain unverified.
