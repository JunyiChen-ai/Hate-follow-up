# M1 Recycler — independent proposal review

Date: 2026-10-03. Reviewer: `/root/m1_grounder_proposal_review`, independent of the proposing/implementing agent and using the current parent-session model. Reviewed `experiments/20261003_m1_recycler/README.md` under rule 4. No candidate implementation, GPU execution, GT access, or Integrator/Amplifier performance inspection was performed. No hashes were calculated or used.

## Decision: PASS

None of the four permitted STOP grounds is established. This decision allows the declared experiment; it does not establish that the proposed Qwen3 adaptation detects semantically redundant tokens, improves localization, or supports a paper contribution.

| Rule-4 STOP ground | Assessment |
| --- | --- |
| Source method already used in hateful-video detection/localization | No verifiable adoption was found in the actual searches and primary-source checks below. This is a bounded search conclusion. |
| Pure ensemble | No: one frozen model and one modified visual-query read. |
| Pure calibration/postprocessing/smoothing | No: redistribution changes attention and subsequent hidden representations before the answer is read. |
| Pure engineering | No: activation-defined sink detection, content-dependent head selection and budget redistribution transfer an existing research method. The constants or backbone change are not the claimed contribution. |

## Source paper and official implementation

Primary source: Kang et al., [*See What You Are Told: Visual Attention Sink in Large Multimodal Models*, ICLR 2025](https://arxiv.org/html/2503.03321v1). Sections 4.1, 5.1–5.2, 6.1 and Appendices A.1/B.2 were read. The paper defines an RMS-normalized activation detector on backbone-specific channels, selects heads using image attention and the non-sink image fraction, then transfers a fraction of sink attention proportionally to non-sink image keys. The declared tau 20, image mass 0.2, transfer fraction 0.6, hallucination rho 0.5, and last-layer exclusion follow the paper. Other task categories use different rho values; 0.5 is a transfer choice, not a universal setting. Source experiments concern image tasks, not hateful-video detection/localization.

The author's official repository **was located**: [seilk/VisAttnSink](https://github.com/seilk/VisAttnSink). Its README identifies the same paper and author; this is not the unrelated visual-autoregressive VAR project. The proposal's statement that no author implementation was located needs updating.

The following author files were actually read:

- [src/logic/constants.py](https://github.com/seilk/VisAttnSink/blob/main/src/logic/constants.py)
- [src/logic/logic.py](https://github.com/seilk/VisAttnSink/blob/main/src/logic/logic.py)
- [src/model/llava.py](https://github.com/seilk/VisAttnSink/blob/main/src/model/llava.py)
- [A_exps/lv1.5_7b.yml](https://github.com/seilk/VisAttnSink/blob/main/A_exps/lv1.5_7b.yml)

The current public code and paper are **not identical**:

| Item | Author implementation | Recycler declaration |
| --- | --- | --- |
| Transfer fraction | `logic.py:196–201` retains `p` and transfers `1-p`; config `p=.6` therefore transfers .4. | Transfers .6, following the paper's definition. |
| Sink threshold/RMS | `logic.py:99–109`: RMS epsilon `1e-6`, strict `> tau`. | Paper-style `>=20`, zero-RMS special case. |
| Detection layers | `logic.py:50–52` starts at layer 2. | Layers 0–34. |
| Head criterion | `logic.py:125–141`: sink-image fraction `<=rho`, denominator epsilon; no selected heads when no visual sink exists. | Non-sink-image fraction `>=.5`; text-only sink mass can still be recycled. |
| Detection timing | `llava.py:313–316` scans block inputs during prefill. | Native cached-token masks plus current query-token masks at each layer. |

At rho .5 the complementary fraction tests agree in exact arithmetic when applicable, but the code's epsilon and empty-visual-sink branch prevent blanket equivalence. These differences are not STOP grounds. The implementation should explicitly follow the declared **paper-based adaptation**, rather than silently switch definitions or claim exact author-code reproduction. The author code also places redistribution after masked softmax and before value mixing (`llava.py:132–150`), consistent with the proposed intervention location.

## Qwen3 channel adaptation

The checked official code has fixed LLaMA channel lists and no Qwen3 implementation or automatic channel-discovery procedure. The paper supplies Qwen2-VL channels `{458, 2570}` in Appendix A.1, but no Qwen3 list. Those indices cannot justify a Qwen3 implementation.

Consequently, choosing each layer's top-two absolute channels of the first prefix-token residual remains an explicitly new transfer assumption. It is **not** an author-validated detector for Qwen3. The source's fixed backbone-wide channels and this layer-dependent rule are different. There is no verified source rule available in the checked material that should replace it automatically.

The proposal correctly states what needs testing: activation magnitude is not proof of semantic irrelevance. Record the first token/position, selected channels, sink rates, selected-head rates and actual redistributed mass. This establishes that the declared operation was applied; it does not by itself validate the semantic hypothesis. Preserve the native prefix/global-QA state and restore both cached masks and KV state between window branches. Earlier rows of the current visual query are intentionally allowed to change.

## Actual target-task novelty search

Searches included:

- Exact source title with `hateful`, `hate video`, and `HateClipSeg`.
- `"Visual Attention Redistribution"` with `hate`, `HateMM`, and `hate video`.
- `"VisAttnSink" "hateful"`.
- `"hateful video" "attention sink"`, `"sink token"`, and `"attention redistribution"`.
- `"HateMM" "sink"` and `"HateClipSeg" "redistribution"`.

Broad searches were supplemented with arXiv, ACL Anthology and OpenReview domain searches. Results for unrelated uses of VAR, social-media attention, or sink tokens in unrelated language tasks were not treated as target-task adoption.

The primary target-task PDFs [SAGE](https://aclanthology.org/2026.acl-long.817.pdf), [LEAF](https://aclanthology.org/2026.findings-acl.604.pdf), and [HVGuard](https://aclanthology.org/2025.emnlp-main.456.pdf) were checked again for the source title, redistribution and sink terminology. Their method sections were already read in the immediately preceding independent review: expert arbitration, self-grounding/distillation, and rationale-assisted multimodal fusion, respectively. No VAR adoption was found. The broader motivation of preventing dominant modalities from obscuring hateful evidence is already present in this literature; it is not a new contribution by itself.

This search supports provisional target-task eligibility under rule 4, not an exhaustive “first ever” assertion. No third-party summary was used as positive or negative adoption evidence.

## Distinction, falsification and cost

Recycler differs materially from archived Selector: the latter selects heads by local-versus-other-window affinity and removes other-window access. Recycler uses sink/non-sink visual mass, preserves temporal access and reallocates probability mass. It also differs from PAI/Amplifier: post-softmax selective redistribution replaces neither all-image logit amplification nor its text-only reference. Unlike Integrator, it leaves prefix encoding causal and native. These are computation differences, not conclusions drawn from the other candidates' pending results.

The matched dense no-redistribution arm can distinguish the operation from attention-backend drift. Removing head selection and replacing detected sinks with matched-count random masks can challenge the stated components if the main result qualifies. Exact random-mask replay must be declared before its run, including eligibility, causality, type/count matching, and whether the resulting redistribution actually changes. The proposal already acknowledges that count matching alone does not isolate semantics. Since all image keys are eligible recipients, component necessity alone cannot establish temporal grounding or visual-speech fusion. Raw and final within-video ordering are necessary evidence; mechanically increased visual attention is insufficient.

The outer-call arithmetic is consistent: native and candidate deployment each use `3+B`, with `B=V+S`; paired native/matched/recycle collection uses `3+B+2V`. The candidate adds real work despite adding no model calls: block-input RMS/channel inspection during native capture, mask storage, and dense per-row/head attention redistribution during visual queries. Attention storage is proportional to `H*Q*(P+Q)` for prefix length P and query length Q, plus roughly `L*P` sink-mask storage, rather than a new dense prefix attention matrix. The 20–40 GPU-minute estimate remains provisional and must be replaced by the declared smoke time/memory observations.

**Disposition:** PASS. Update the source-code availability statement and explicitly document the paper/code differences before implementation. Keep the Qwen3 channel rule and query-only scope labeled as adaptations. Performance and mechanistic claims remain subject to complete paired results and the already declared controls.
