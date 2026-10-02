# Independent proposal review: M1 Grounder

Date: 2026-10-02. Reviewer: separate agent instance
`/root/m1_grounder_proposal_review`, inheriting the main-session model.
Scope: one proposal review under `RESEARCH_ITERATION_RULES.md` rule 4;
not a code review or a prediction of experimental success.
Proposal reviewed: `experiments/20261002_m1_grounder/README.md`, initial
five-arm version. No experiment or method code was changed by this reviewer.

## Decision: PASS

None of the four permitted STOP conditions is established. Implement and test.
This decision does **not** certify a novel general attention technique, a
successful mechanism, or publication readiness. The research contribution, if
supported, is the application and explanation of restricted direct evidence
access in frozen-MLLM hateful-video localization.

| Rule-4 STOP condition | Finding |
|---|---|
| Source mechanism already used for hateful video detection/localization | Not found in the actual searches and primary papers below. There are close methods in general VLM reasoning and safety alignment; scope the novelty accordingly. |
| Pure ensemble | No. One frozen MLLM; the declared arms are separate controls, not averaged predictions. |
| Pure calibration/postprocessing/smoothing | No. The intervention changes attention during the computation producing local logits. The existing temporal decoder is unchanged. |
| Pure engineering or prompt/backbone/constant change | No. The proposal changes which evidence is directly accessible at specific model depths and states a falsifiable localization hypothesis. The fixed depth fraction is an implementation constant, not the entire idea. |

Low expected success, indirect information leakage, and potential degradation
are **not** STOP grounds under rule 4. They belong in the experiment and claim
boundaries.

## Literature search and closest primary sources

Actual search performed 2026-10-02. Queries included:

- `hateful video localization attention masking large multimodal models layers`
- `"hateful video" "attention" "mask" MLLM`
- `"hateful video" "layer" "context" MLLM`
- `"hateful video" "attention steering"`
- `"hateful video" "PASTA"`
- `"MultiHateLoc" arxiv`; `"HVGuard" attention method`
- `site.aclanthology.org "LEAF" "Hateful Video"`
- `site.aclanthology.org "SAGE" "Hateful Video"`
- `"The Ebb and Flow of Multimodal Focus" arxiv`

Search summaries were used for discovery only. Comparisons below use the
primary papers. For ACL PDFs, text was also retrieved directly from the
publisher and parsed in memory after the browser's PDF section lookup failed.

**Existing general mechanisms.**

- [PASTA, Tell Your Model Where to Attend](https://arxiv.org/html/2311.02262v1)
  intervenes on selected attention heads to emphasize user-specified text
  during inference. Attention steering without parameter updates is therefore
  established; neither freezing the backbone nor editing attention alone is a
  defensible general novelty claim.
- [AutoPASTA](https://arxiv.org/html/2409.10790v1) identifies important context
  and steers attention for open-book QA. It differs from a fixed temporal
  window and a prescribed depth schedule, but is relevant prior art for
  explicit evidence emphasis.
- [TRACE, The Ebb and Flow of Multimodal Focus](https://arxiv.org/html/2607.11436v1),
  especially Sections 3 and 4.1, is the closest conceptual antecedent: it
  controls visual evidence use across depth and subsequent decoding, using
  lightweight learned modules. Its reported tasks are image grounding,
  document/visual QA, and reasoning; its listed benchmarks do not include
  hateful-video detection/localization. The present proposal has a fixed
  schedule, known temporal support, no learned controller, and local Yes/No
  reads. It must not claim to originate depth-dependent evidence scheduling.
- [GuardAlign](https://arxiv.org/html/2602.24027v1), Section 3, changes
  instruction-to-safety-prefix attention in middle layers. Its task is
  multimodal generation safety rather than localizing hateful segments in
  input video. This is adjacent safety-related attention intervention and
  should be acknowledged; the shared word “safety” does not establish the
  rule-4 prior-use condition.

**Hateful-video papers checked.**

| Primary source | Relevant method and distinction |
|---|---|
| [MultiHateLoc](https://arxiv.org/html/2512.10408v1), Section 3 | Learned modality-specific temporal encoders, cross-modal contrast/fusion, and video-label MIL. Does not implement frozen-MLLM branch attention restricted by depth and target-window support. |
| [HVGuard](https://aclanthology.org/2025.emnlp-main.456.pdf), Section 3 | MLLM rationales combined with multimodal representations through a learned MoE classifier. Not the proposed internal per-window attention intervention. |
| [RAMF](https://arxiv.org/html/2512.02743v1), Section 3 | Adversarial rationale generation, local/global feature fusion, and semantic cross-attention. “Local/global” refers to its representation fusion; it is not the proposed early-global/late-local MLLM access schedule. |
| [CLARA](https://arxiv.org/html/2608.15905v1), Section 3 | Utterance-aligned clips, MoE encoding, local/global contrastive learning, and a rationale-gated Transformer for video classification. Global rationales guide learned aggregation rather than being blocked at selected frozen-MLLM layers. |
| [LEAF](https://aclanthology.org/2026.findings-acl.604.pdf), Section 3 | Self-grounding CoT produces explanation supervision for staged distillation. “Grounding” is not the same computation as temporal attention support restriction. |
| [SAGE](https://aclanthology.org/2026.acl-long.817.pdf), Section 3 | Modality experts exchange information through global deliberation and adaptive arbitration. It does not establish prior use of the particular frozen-MLLM depth schedule proposed here. |

This search did not establish prior use in the target task. That is a bounded
search conclusion, not proof of absence from all literature. Under the
repository's explicitly permitted cross-task transfer criterion, it supports
PASS. A stronger public novelty claim would require the usual later review.

## Prior work inside this repository

Read `docs/reviews/20260928_codex_mechanism_review_round2.md`, the corresponding
section of `experiments/20260928_headroom/README.md`, and searched experiment,
archive, review, and wiki Markdown for layer-dependent and late-layer masks.

The **same conceptual candidate was suggested on 2026-09-28 but not directly
run**. That record explicitly distinguishes the proposed GPU intervention
from its CPU check of completed full/reduced-context scores. The cached-read
mixtures failed to establish the premise; they did not execute the candidate.
Window-only inputs, no-verdict inputs, context subtraction, and isolated
branches also change different computations. No previously completed exact
experiment was found. Attribute the earlier suggestion rather than call this
a newly invented idea.

## Can the experiment explain an improvement?

The existing controls make important claims falsifiable:

- `late` versus `base`: whether the intervention helps at all under identical
  media, prompts, global judgement, and decoder procedure.
- `late` versus `verdict_only`: whether limiting remote media contributes
  beyond removing direct access to the global Q/A turn.
- `late` versus `shifted`: whether the selected prefix support matters beyond
  the number of accessible tokens. Since copied local speech stays in the
  branch query, this is only a wrong-**prefix**-evidence control.
- `late` versus `all_local`: whether retaining early direct access helps
  compared with restricting the whole branch computation.
- Raw local ranking versus final ranking: whether the read itself becomes
  more temporally discriminative instead of merely moving the downstream
  unsupervised fit into another regime.

**One necessary correction for the proposed depth-specific explanation:**
`all_local` changes both *which layers* and *how many layers* are restricted.
It cannot by itself establish that late restriction is better because it is
late, rather than simply because fewer layers are restricted. Before outcome
inspection, add an `early` control with exactly the same token mask and exactly
`ceil(L/4)` restricted query layers, placed at the beginning of the stack;
leave the later layers unrestricted. Compare `late` and `early` using the
already declared both-corpus effect threshold. This adds one evaluation arm,
not extra calls when deploying the selected method. Alternatively, retain the
five arms but explicitly abandon the stronger claim about timing of access.
This is a claim/experimental-design correction, **not** a fifth STOP category.

Necessary interpretation and reporting boundaries:

1. Treat “early context interpretation, late decision” as a hypothesis, not a
   known decomposition of this backbone. Even a successful early/late contrast
   supports differential effects of the intervention, not exclusive cognitive
   roles for layer groups.
2. Retain the proposal's qualification: cached K/V states, query residuals,
   copied speech, and contextualized scaffolding can retain remote information.
   Neither `late` nor `all_local` creates complete local-only representations.
3. For `shifted`, report actual overlap with the true mask and cases with
   no usable change. Token-position rotation may retain partial frame/text
   units; do not describe it as substituting a coherent distant video window.
4. Document raw Reader ranking changes, gain/loss cases, and subgroup results
   before claiming improved local evidence discrimination. A change in global
   score scale or final metrics alone is insufficient. Reusing the same
   decoder algorithm does not mean its fitted parameters are numerically fixed.
5. A mask's zero attention to blocked keys is an implementation sanity check,
   not independent evidence that the remaining information is semantically
   correct. Use the declared controls and both-corpus raw/final outcomes.

The proposal already predeclares constants, evaluation gates, coverage checks,
failure reporting, and costs. Preserve these commitments, add the timing
control or narrow the intended claim, and proceed to implementation/code
review. No extra general proposal-review round is needed.
