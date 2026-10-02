# M1 Amplifier — independent proposal review

Date: 2026-10-03. Reviewer: `/root/m1_grounder_proposal_review`, independent of the proposing/implementing agent, using the current parent-session model. Scope: the single rule-4 review of `experiments/20261003_m1_amplifier/README.md`. No implementation, GPU execution, GT access, pending-run performance inspection, or hash calculation was performed. The proposer incorporated the factual clarifications below during this review, before implementation.

## Decision: PASS

None of the four rule-4 STOP conditions is established. This permits implementation and the declared experiment; it does not establish effectiveness, a localization mechanism, or publication novelty. The bounded search below found no verifiable use of PAI's attention amplification plus text-only token-logit reference in hateful-video detection/localization. Absence from this search is not proof of exhaustive absence.

| STOP ground | Assessment |
| --- | --- |
| Source method already used in hateful-video detection/localization | Not established by the actual searches and primary-source checks below. |
| Pure ensemble | No: both reads use the same frozen MLLM, under the repository's definition. |
| Pure calibration/postprocessing/smoothing | No: the candidate changes attention computation inside the language model and obtains a fresh language-only reference. It does not merely transform existing scalar window scores. |
| Pure engineering | No: it transfers two inference mechanisms from a published method. Constants, Qwen compatibility, and omitting APC are implementation/transfer choices, not the claimed contribution. |

## Source fidelity

Primary paper: Liu, Zheng and Chen, *Paying More Attention to Image: A Training-Free Method for Alleviating Hallucination in LVLMs*, ECCV 2024 ([paper](https://arxiv.org/html/2407.21771v1), [proceedings PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10933.pdf)). Sections 4.1–4.2 and 5.1 were checked. Equation 3 adds `alpha * abs(logit)` to image-key attention logits in the last query row before softmax. Section 5.1 uses alpha 0.5 for LLaVA, 0.6 for Shikra, 0.2 for the resampler model, and gamma 1.1. The paper studies image hallucination, not hateful videos. Equation 4 uses probability notation; the implementation below resolves the candidate's exact arithmetic.

Author code was read directly, not inferred from a third-party implementation:

- [attention.py](https://github.com/LALBJ/PAI/blob/master/attention.py), lines 67–99: mask application, last-row image-span intervention, then softmax. Lines 120–128 configure the layer range. The [author README](https://github.com/LALBJ/PAI) example uses `start_layer=2`, `end_layer=32`. Extending this range through Qwen layer 35 is a declared transfer assumption, not an experimentally established source setting for Qwen.
- [CFG.py](https://github.com/LALBJ/PAI/blob/master/CFG.py), lines 28–58: log-softmax both branches, temporarily disable amplification for the reference branch, and contrast with gamma. It additionally masks tokens whose amplified probability is below 0.1 times the largest amplified probability. The candidate explicitly omits that adaptive plausibility constraint (APC), so it is not an exact reproduction of the author's complete decoder.

The revised declaration correctly identifies its token-logit arithmetic as matching the code's contrast term, up to a token-independent normalizer. Without APC, that normalizer cancels between the Yes and No logsumexp terms. Contrast must occur **before** class aggregation; subtracting already aggregated class margins is generally different. Omitting APC can amplify low-probability answer variants; this is an empirical diagnostic and claim limitation, not a rule-4 STOP ground.

## Actual target-task search

Queries included the exact title with `hateful`, `HateMM`, and `hate video`; `"PAI" "hateful video detection"`; `"HateClipSeg" "PAI"`; and target-task combinations with `attention amplification`, `image-centric`, `visual attention`, `attention intervention`, `classifier-free guidance`, `text-only logits`, and `contrastive decoding`. Broad searches were supplemented with arXiv/ACL-domain searches. Many results were unrelated uses of “attention,” “PAI,” or “contrastive”; these were not counted as adoption.

Primary target-task checks included:

- [SAGE, ACL 2026](https://aclanthology.org/2026.acl-long.817.pdf): its method uses modality-specific experts, shared-context refinement, and learned expert arbitration. The similar motivation that dominant benign modalities obscure hateful cues is already present. Its feature/expert mechanism is not the frozen-MLLM last-row PAI intervention plus language-only logit contrast. Thus the motivation cannot be presented as newly discovered here.
- [HVGuard, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.456.pdf), sections 3.2–3.4: multimodal extraction, CoT-derived rationale, and subsequent feature fusion. This is not PAI-style internal attention intervention or its text-only logit reference.
- [LEAF, ACL 2026 Findings](https://aclanthology.org/2026.findings-acl.604.pdf), sections 3.1–3.4: self-grounding CoT and staged distillation, followed by model inference. This does not establish adoption of PAI.
- [MMSafeAware, ACL 2025](https://aclanthology.org/2025.acl-long.832.pdf): a relevant safety precedent evaluates visual contrastive decoding on static image-prompt pairs. It is neither a hateful-video PAI application nor evidence that visual grounding interventions in safety are broadly new.

These primary texts were also searched for the source title and mechanism terms; simple substring hits inside unrelated words were not treated as citations. No positive target-task adoption evidence was found. This conclusion does not reclassify static hateful memes, text hate detection, or generic video hallucination as the target task.

## Controls, interpretation, and necessary clarifications

The native and matched-alpha-zero arms are appropriate: promotion against native alone could conflate intervention effects with eager/SDPA arithmetic changes. The cached 2-by-2 removals—neither, attention only, reference contrast only, and both—can test component necessity without additional model reads. Every claimed component must still meet the repository's ablation gate; source reputation does not exempt it.

Three boundaries must remain explicit:

1. **Cache wording:** “all cached states stay unchanged” means the reused native prefix/global-question/answer cache. The intervened suffix last-token hidden state, and potentially its deeper-layer K/V entries, change. Correct cropping/restoration is needed between independent window queries. This is a wording correction, not a redesign.
2. **Support control:** amplification covers all image keys. Permuting membership/order inside that same set does not change the support mask. The proposer has corrected this. A later temporal-grounding control must change actual media/support or alignment and document that the intervention is nontrivial. The 2-by-2 component ablation alone establishes necessity, not correct temporal evidence use. A wrong-modality support test would likewise not independently prove temporal alignment.
3. **Reference meaning:** the no-image prefix also removes image timestamps and changes introductory wording. It retains ASR, policy and the native forced verdict. It is a language-only conditional reference, not a pure pixel intervention, nor a removal of verdict anchoring. Attention boost across all frames does not enforce target-window evidence or restore bidirectional visual-speech encoding.

Raw visual-window ordering, final within-video ordering, native-verdict strata and pooled metrics are therefore useful discriminators already in the declaration. All-three-metric reporting and the fixed r6 interface remain necessary. An increase in attention mass by construction is not evidence that semantic localization improved. None of these cautions is a speculative-performance STOP.

The candidate also differs materially from the earlier Grounder/Selector restrictions and Integrator prefix re-encoding: it preserves native prefix encoding and access, increases image-key logits during the last query-row read, and adds a separate text-only reference. It is not simply another name for a masked-prefix or noisy-image experiment. No previous-run performance was read to make this distinction.

## Cost check

With V visual-window queries, S speech-window queries and B = V + S, the declared counts are internally consistent:

- Native deployment: `3 + B` outer forwards.
- Candidate deployment: `6 + B + V` = `6 + 2V + S`; overhead over native is `3 + V`.
- Paired collection of native, matched, amplified and text-only readings with shared speech: `6 + B + 3V` = `6 + 4V + S`.

These count outer calls, not equal-cost operations. Dense attention may increase the visual-branch cost, and the language-only branch still requires a new prefix encoding. The 20–35 GPU-minute estimate for the paired 333-video run remains an estimate to replace with the declared input-only smoke measurements. No GPU time was measured in this review.

**Disposition:** PASS for the declared standalone experiment, with the cache wording clarified before implementation. APC omission, the paper/code arithmetic distinction and the invalid all-image-set permutation have already been acknowledged in the proposal. Any eventual contribution claim remains conditional on complete paired results and the declared mechanism controls.
