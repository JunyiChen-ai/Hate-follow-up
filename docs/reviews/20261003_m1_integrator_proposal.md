# Independent proposal review: M1 Integrator

Date: 2026-10-03. Reviewer: independent agent `/root/m1_grounder_proposal_review`, same model as the parent session. **Decision: PASS under RESEARCH_ITERATION_RULES.md rule 4.**

Reviewed `experiments/20261003_m1_integrator/README.md`, including the final clarifications on indirect information flow, future-text scope and the explicit-causal comparison. This is the single independent proposal review. No implementation was written, no GPU experiment was run, and no current visual-contrast performance or GT was read. The reviewer did not edit the method declaration. PASS permits implementation and the declared experiment, not promotion or an effectiveness claim.

## Four-condition decision

| Permitted STOP ground | Finding |
|---|---|
| Source mechanism already used in hateful-video detection/localization | Not established by the actual searches and inspected primary methods below. Multimodal fusion and bidirectional encoders already exist in this task, but they do not by themselves establish inference-time relaxation of visual rows in a frozen autoregressive MLLM's language-prefix mask. |
| Pure ensemble | No. One frozen MLLM supplies the native global decision and the modified local reads. There is no combination of predictions from independent models or averaging of candidate local branches. |
| Pure calibration/postprocessing/smoothing | No. The altered attention graph requires a new prefix encoding and new local reads. It cannot be reconstructed by rescaling existing native logits. |
| Pure engineering | No under the repository's transfer criterion. This transfers a specified attention mechanism with a testable representation hypothesis; it is not only a prompt rewording, backbone replacement, resolution change or scalar adjustment. The attention algorithm itself is prior work. |

Possible degradation, propagation of global context, distribution shift from the pretrained mask, or a benefit limited to video-level offsets are experimental questions, not additional STOP conditions.

## Source fidelity

Primary sources read:

- [Rethinking Causal Mask Attention for Vision-Language Inference, arXiv v1](https://arxiv.org/html/2505.18605v1), Section 3.1, Equations 7–9 and the discussion of the separate lightweight family.
- [FutureMask author repository](https://github.com/TerryPei/FutureMask), which identifies the ICLR 2026 version and maps mask/merge flags.
- [Author mask implementation](https://github.com/TerryPei/FutureMask/blob/main/LLaVA-mix_merge_v1/llava/model/kv_token_merge/modify_llama.py), `_make_causal_mask`, especially source lines 217–282, plus its attention call and decoder wrapper.
- [Author layer dispatch](https://github.com/TerryPei/FutureMask/blob/main/LLaVA-mix_merge_v1/llava/model/language_model/llava_llama.py) and [full-mask launch script](https://github.com/TerryPei/FutureMask/blob/main/scripts/llava-v1.5-7b/new_eval_future_7b.sh).

Equation 7 permits `j<=i OR i in V`. The author's `future` branch sets every visual query row to allow all supplied positions while preserving the lower triangle for other rows. The declaration matches this mask. The source dispatch installs its modified decoder across language layers. The full-mask script selects `attention_mask_mode=future` and `compress_future=original`; Integrator correctly excludes the pooling/merge variant. Source experiments already include temporal multi-image tasks, so general video or temporal use is not new.

Integrator transfers that mask definition, not the complete LLaVA software stack or its performance claims. The author code identifies fixed-size visual patch spans; Qwen needs its actual expanded image-token positions and native positional machinery. Native-global conditioning and limiting the intervention to the reusable media/policy prefix are project-specific adaptations. In particular, the later global/local questions are not visible during the altered prefix pass. Do not describe this as an exact reproduction of the source's entire inference protocol.

## Target-task search

Actual query families included:

```text
"FutureMask" "hate"
"Rethinking Causal Mask Attention" "hateful"
"hateful video" "bidirectional" attention
"HateMM" "causal mask"
"FutureMask" "HateMM"
"2505.18605" hateful video
"hateful video" "future-aware"
"hateful video" "causal attention"
"hateful video" "non-causal" attention
"hateful video" "future" "mask"
"HateClipSeg" "bidirectional" attention
"HateMM" "prefix" attention
"hateful video" "visual tokens" "mask"
"hate video" "non-causal"
"hateful video" "future-aware attention"
"FutureMask" moderation
"future-aware" "hate" attention
"visual tokens" "hateful video" attention mask
"hateful video" "bidirectional attention"
```

The following primary method descriptions were checked during this round rather than treating search-word co-occurrences as source applications:

| Source | Distinction relevant to this review |
|---|---|
| [RAMF](https://arxiv.org/html/2512.02743v1), Sections 3.3–3.5 | Uses modality encoders, local/global pooling and learned semantic cross-attention. Its displayed attention equation adds a causal mask; the described feature-fusion module does not open future input edges for visual tokens in the frozen VLM prefix. |
| [CLARA](https://arxiv.org/html/2608.15905v1), Sections 3.2–3.4 | Uses modality features, MoE clip fusion and a rationale-gated video Transformer. Its use of bidirectional text encoders, contrastive objectives or whole-video context does not establish the FutureMask intervention. |
| [LELA](https://arxiv.org/html/2602.09637v1), Section 3 | Uses modality descriptions and staged LLM assessment to produce local scores, rather than the inspected visual-row mask change. |

Earlier primary-source readings of SafeWatch/PEPE, HVGuard, LEAF, IARE and SAGE are documented in the Factorizer and Eraser proposal reviews. They provide adjacent context, not a new exhaustive survey here. In particular, SafeWatch's policy-block attention and existing audiovisual fusion prohibit broad claims that internal attention changes or multimodal integration are first introduced to video moderation. They do not establish the specific visual-row future-access source use from the inspected descriptions.

The bounded conclusion is that **no target-task adoption of the specified source mechanism was verified**, not that literature absence has been proved. A citation to FutureMask, a BERT encoder, or a generic cross-attention module alone would not establish adoption of its mask. Conversely, training or a detection-only output would not excuse an actual adoption if found.

## Difference from archived attention candidates

The original mechanism sections of these archived READMEs were inspected. No archived performance files were needed.

| Candidate | Prefix computation | Subsequent window queries |
|---|---|---|
| Grounder | Native causal prefix | Removes direct access to remote media/global verdict in the final quarter of query layers. |
| Factorizer | Removes cross-window media edges and prevents scaffolding from relaying media; retains causal ordering | Ordinary access to the complete modified prefix. |
| Integrator | Adds future visual/text access for visual prefix rows; all native causal edges remain | Ordinary access to the complete modified prefix. |

Integrator therefore changes a different computation from Grounder and the opposite edge operation from Factorizer. Neither archived candidate tested this graph. Their failures do not prove that opening these edges will help, and are not evidence for a missing-fusion diagnosis.

The actual prefix construction in `src/mllm_judge.py:153` was read: frame content precedes the transcript/policy tail. This supports the stated directional restriction in the native language-prefix encoding. It does not imply that the original later questions cannot combine modalities; the declaration correctly rejects that stronger claim.

## Accepted clarifications and claim boundaries

1. **Direct masks versus information flow.** Text rows retain causal direct edges, but deeper text states can receive later input through already updated visual states. The latest declaration now says this explicitly. All such information belongs to the supplied input; the modified pass contains no answers. Ordinary suffix decoding does not turn this offline, whole-video method into an online-causal localizer.
2. **Future text is broader than speech.** The complement of the visual token set includes ASR, policy, timestamps and scaffolding. Full versus visual-to-visual therefore tests all added future-text access, not ASR specifically. The latest README incorporates this boundary. A speech–vision mechanism claim would require an ASR-specific source control; it is not justified by these two masks alone.
3. **Cached states outside visual positions can also change.** Keeping their direct mask rows unchanged does not freeze their representations. Both local branches and the global-question/forced-answer extension consume the altered prefix. The controlled quantity is the native global margin and selected answer text, not every global-context hidden state or the decoder's final video-level key.
4. **A numerical implementation comparison is essential here.** The parent reports prior BF16/backend drift between native implicit-causal attention and an explicit causal mask. The latest declaration appropriately retains both full native and explicit-causal arms, does not demand their bitwise equality, and requires the main comparison against both plus the current method. Exact native/restored-native parity remains a separate check. An improvement shared by the explicit-causal control is not evidence for the new edges.
5. **Effective masks, not only requested masks, must be checked.** Qwen may use different kernels for implicit and explicit masks. The declared actual-model witness and code review should verify that no additional causal restriction silently intersects the opened mask, that expanded image-token membership is correct, and that only the prefix pass receives the intervention. This is verification of the stated method, not a new design or tuning request.

The final declaration already contains the material wording corrections communicated during this review. No additional proposal round is required for them.

## Falsifiability and cost

The native/explicit-causal comparison removes the new edges through matched plumbing. If the main method qualifies, the declared visual-to-visual and visual-to-text masks distinguish the two categories of added edges. These comparisons support edge-type claims, subject to the text-source limitation above; they do not independently prove semantic grounding or cross-modal synergy. Report raw ordering and both local branches alongside decoded results. Unchanged global input scores do not exclude downstream video-level shifts from changed local-score distributions or separately fitted r6 states.

The outer-forward counts are consistent: native `3+B`; deployment `5+B` for a native prefix/global read followed by a fresh modified prefix, global-question read, forced native answer and local reads. Three paired full arms can share the native verdict while costing `3*(3+B)` outer forwards as declared. Dense-mask storage and loss of a faster kernel can add substantial cost beyond this call count. The source's merge-variant speed claims do not apply. The 12–20 minute deployment and 30–45 minute paired estimates for 333 videos remain provisional until the declared smoke measurement.

**Final disposition: PASS.** Proceed to implementation, then the required independent code review and full paired experiment. Novelty is limited to an effective, empirically supported target-task transfer of a known mask mechanism; no source novelty or performance conclusion is established by this review.
