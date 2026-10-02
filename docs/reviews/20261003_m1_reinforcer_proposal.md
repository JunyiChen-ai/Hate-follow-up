# M1 Reinforcer — independent proposal review

Date: 2026-10-03. Reviewer: `/root/m1_grounder_proposal_review`, independent of the proposing/implementing agent, using the current parent-session model. Scope: one rule-4 review of `experiments/20261003_m1_reinforcer/README.md`. No implementation, GPU execution, GT access, or Recycler/Integrator R4 performance inspection was performed. No hashes were calculated or used.

## Decision: PASS

None of the four permitted STOP grounds is established. The candidate may proceed to implementation and the declared experiment. This is not a claim of effectiveness, temporal-grounding mechanism, or publication priority.

| STOP ground | Finding |
| --- | --- |
| Source method already used in hateful-video detection/localization | No verifiable VISTA/VSV/SLA adoption was found in the actual searches and primary checks below. This is a bounded search conclusion. |
| Pure ensemble | No. Native/reference passes and intermediate-layer projections belong to the same frozen model. |
| Pure calibration/postprocessing/smoothing | No. VSV changes residual states within the network; SLA obtains additional internal-layer readouts. The full candidate is not merely a scalar transformation of cached predictions. |
| Pure engineering | No. It transfers a published inference mechanism, with explicit model/scope adaptations. Borrowed constants or a backbone replacement are not presented as the contribution. |

## Paper and author-code verification

Primary paper: Li et al., [*The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering*, ICML 2025](https://arxiv.org/html/2502.03628v2). Sections 2.1, 2.3–2.4 and 3.1 were read. Equations 4–6 use the difference between positive/no-image last-token residuals, add the layer-specific direction, and restore the original residual norm. Equations 7–8 average the five preceding layers' logits and mix them with final logits. The declared lambda .17 for LLaVA and gamma .3 match the stated defaults. The source selected lambda/gamma on 100 MSCOCO validation images; “no scan” describes this proposed transfer, not the original paper. The source studies image hallucination, including generation and visual questions, not hateful-video localization.

The official [VISTA repository](https://github.com/LzVv123456/VISTA) was read directly. The proposal accurately separates its paper-defined method from the released implementation:

- [llm_layers.py](https://github.com/LzVv123456/VISTA/blob/main/llm_layers.py), lines 15–32 and 132–150: VSV is attached after the MLP, uses normalized directions and activations with a cosine-dependent gain, restores the input norm, and returns FP16. This differs from adding an unnormalized direction to the complete post-block residual and casting back to BF16.
- [steering_vector.py](https://github.com/LzVv123456/VISTA/blob/main/steering_vector.py), lines 86–129, and [myutils.py](https://github.com/LzVv123456/VISTA/blob/main/myutils.py), lines 114–137: last-token layer activations are captured; the contrast is passed through a PCA procedure. Direct subtraction in the proposal follows the paper and is not a claim of identical released-code behavior.
- [pope_eval.py](https://github.com/LzVv123456/VISTA/blob/main/pope_eval.py), lines 29–37, and [run_pope.sh](https://github.com/LzVv123456/VISTA/blob/main/run_pope.sh): POPE uses lambda .01 and defaults to layers `25,30` for augmentation.
- [llava_llama.py](https://github.com/LzVv123456/VISTA/blob/main/llava/model/language_model/llava_llama.py), lines 92–103: the range is inclusive, hence six layers, and intermediate states go directly through `lm_head`. The proposal instead defines five layers and explicitly applies the final RMSNorm before each projection.

There is no requirement under rule 4 to reproduce every released-code choice. Keeping these differences explicit avoids attributing the resulting experiment to a nonexistent exact source reproduction.

## Target-task search and nearest relevant prior

Actual searches combined the source title, `VISTA`, `Visual Steering Vector`, `Self-Logits Augmentation`, `visual information steering`, `activation steering`, and `residual steering` with `hateful video`, `hate video`, `HateMM`, and `HateClipSeg`. Searches used both the broad web and primary-domain restrictions to arXiv, ACL Anthology and OpenReview. Unrelated projects named VISTA and nontechnical uses of “steering” were excluded.

Primary target-task texts checked for these mechanisms included [SAGE](https://aclanthology.org/2026.acl-long.817.pdf), [LEAF](https://aclanthology.org/2026.findings-acl.604.pdf), [HVGuard](https://aclanthology.org/2025.emnlp-main.456.pdf), [IARE](https://arxiv.org/html/2606.11953v1), and [MARS](https://arxiv.org/html/2601.15115v1). Their method sections had already been read in this independent review sequence; this round additionally checked the source title and steering/SLA terminology. No VISTA adoption was found. The result is not an exhaustive guarantee of absence.

A relevant positive neighboring precedent is [FBHM: Functional Benchmarking and Steering of VLMs for Hateful Meme Detection](https://arxiv.org/html/2605.31349v2), whose section 4 was read. It learns layerwise steering vectors and coefficients from labeled static meme examples. This is actual activation steering for multimodal hate, but not hateful-video detection/localization and not the proposed per-input label-free residual contrast plus SLA. It does not trigger the target-task STOP rule. It does preclude an unrestricted claim that activation steering for multimodal hate is itself new.

No third-party summary was used to establish source mechanics or prior adoption.

## Transfer definition and component boundaries

The proposed computation is sufficiently specified for a code review:

- Directions come from unsteered native and no-image reads of the same window question. The no-image condition also removes image timestamps and changes introductory wording; it is not a pure image-semantic causal effect.
- Native prefix/global-QA representations are retained. Every suffix row is steered at all 36 post-block outputs, with FP32 update/norm computation followed by BF16 casting. This query-only scope is an adaptation. It must not be described as the source's full-prefill intervention.
- For a zero-based final block 35, outputs 30–34 are exactly the five preceding blocks. Applying the same final RMSNorm and head to each is an explicitly declared Qwen transfer. The final head must not receive an unintended second RMSNorm.
- Computing only the existing answer-token columns is algebraically sufficient: both averaging and mixing act independently on vocabulary columns, and the shared softmax normalization cancels in the Yes/No margin. Combining already aggregated class margins would be a different computation.

The paper calls SLA a logits ensemble, but it does not combine independently trained models here. Under repository rule 3 this is same-model internal readout, not the prohibited ensemble category. Its use also differs from the earlier DoLa candidate's selected-layer contrast; positive averaging is not a renamed contrast operation. PAI modifies attention logits and subtracts a text-reference logit distribution, whereas this candidate modifies hidden residuals and uses intermediate-layer augmentation.

The source's long-generation information-loss explanation does not establish why a one-token local Yes/No read fails. That transfer hypothesis must be tested. A changed hidden direction or its increased visual association is not, by itself, temporal-localization evidence. Full-video context and the forced native verdict remain present; there is no direct target-window support restriction or removal of verdict error propagation.

## Cost and experimental checks

With V visual questions, S available speech questions and B = V + S, the declared full deployment and paired native/full collection both require:

`(3 + B) native reads + (3 + V) no-image reads + V steered replays = 6 + B + 2V`.

Native positive reads are required to obtain directions on a new video. They are therefore correctly charged rather than treated as free cached data. Cached VSV-only and SLA-only removals can reuse the stored steered final and native intermediate logits, respectively.

The proposer clarified the initially ambiguous cache schedule during review: prefills execute sequentially, but both unmodified KV caches coexist on the GPU until the reference queries finish; then the reference cache is deleted and the original native cache is reused for steering. No native rebuild or CPU offload is assumed. Under that schedule, the formula is consistent. Rebuilding the native prefix/global/answer would require additional calls and cannot be silently folded into the same count. Both KV caches, mRoPE state restoration, direction transfers and hook overhead must be included in the smoke measurements. The 20–35 GPU-minute and under-32-GiB estimates remain unmeasured forecasts.

The declared primary gate—within-video gain of at least .01 on both corpora versus native/current, without pooled loss over .005—matches the stated objective. Component claims require the declared removal effects on a common main metric in both corpora. Keeping the full candidate fixed before inspecting component variants avoids silently choosing the best ablation as the main result. Rule-9 revision/archive handling is stated.

The proposed within-video direction shuffle can challenge query specificity after a qualifying main result. Its exact fixed-seed mapping must be declared before replay, as already promised. Record whether vectors actually change: single-window cases, fixed points and identical directions do not supply evidence of a successful misalignment intervention. A shuffle tests query matching; it does not independently establish a purely visual causal mechanism.

**Disposition:** PASS for the standalone, paper-defined VSV+SLA transfer. Source-code differences and the dual-cache cost schedule are explicit. Implementation correctness, complete paired results and the declared component/misalignment controls remain prerequisites for any effectiveness or mechanism claim.
