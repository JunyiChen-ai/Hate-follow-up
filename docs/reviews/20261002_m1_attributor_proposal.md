# Independent proposal review: M1 Attributor

Date: 2026-10-02. Reviewer: independent same-model agent
`/root/m1_grounder_proposal_review`.
Scope: one rule-4 review of `experiments/20261002_m1_attributor/README.md`.
No declaration, method code, or experiment was changed. Grounder and Selector
reviews are not reopened.

## Decision: PASS

None of the four permitted STOP conditions is established. This is a narrow
permission to implement and test a known attribution method in a specific
internal representation space. It is not certification of publication novelty
or evidence that the mechanism will improve localization.

| Rule-4 STOP condition | Finding |
|---|---|
| Source mechanism already used in hateful-video detection/localization | No located primary source establishes this cached-value-path integrated-gradients mechanism in the target task. General explainability in hateful video and integrated gradients for hateful text already exist; see the limits below. |
| Pure ensemble | No. One frozen MLLM; path evaluations numerically integrate one function rather than combine independent models or decisions. |
| Pure calibration/postprocessing/smoothing | No. The local signal requires differentiating the model under internal value interventions; it cannot be obtained by rescaling completed video/window scores. Calling IG a post-hoc explanation in the XAI literature does not make it pure output-score postprocessing under this rule. |
| Pure engineering trick | No. The declared mechanism replaces independent window questions with signed temporal attribution of a joint decision under an explicit reference path. Quadrature, caching, and speed are implementation details, not the scientific contribution. |

Prior failures or concerns about indirect information, saturation, sparse
frames, or attribution quality are empirical risks, not additional STOP
categories. The endpoint-gradient comparison will determine whether path
integration deserves a substantive claim.

## Actual literature search

Queries executed on 2026-10-02:

- `"hateful video" "integrated gradients"`
- `"hate video" "integrated gradients"`
- `"HateMM" "integrated gradients"`
- `"hateful video" "evidence attribution"`
- `"hate video" "gradient" attribution`
- `"HateMM" "attribution"`
- `"hateful video" "Grad-CAM"`
- `"hate video" "Sundararajan"`
- The full WWW Companion 2026 paper title, its DOI, and title searches with
  `integrated`, `ranking`, and `filetype:pdf`.

Primary sources were opened where available. Search results pointing only to
references to HateMM were not treated as evidence that the method was evaluated
on hateful video.

### The explicitly requested WWW 2026 paper

[Yadav and Singh, An Interpretable Agentic Framework for Multimodal Hate Video
Analysis with Explicit Evidence Attribution](https://doi.org/10.1145/3774905.3796488),
WWW Companion 2026, pages 464–472, published 2026-05-28.

**Access obtained:** the publisher's indexed abstract and bibliographic record
were returned by search. Direct full-text, PDF, and ePDF access failed; no
accessible author manuscript was located. This review therefore does not claim
to have inspected the full methods section.

The abstract describes object/OCR/ASR/text-toxicity/entity extraction and offline
retrieval, a shared evidence structure, deterministic prioritization with
cross-modal agreement/conflict, and deterministic or constrained-LLM decision
orchestration. It evaluates HateMM, ImpliHateVid, and YTHate. This establishes
prior **explicit evidence attribution in hateful-video analysis**, so no broad
first-attribution claim is permissible. The accessible account does not
establish internal cached-value integrated gradients or temporal localization
through that computation. Its title alone is insufficient for rule-4 STOP.
Full-text verification remains an explicit limitation before any strong public
novelty claim; the current evidence does not establish the STOP condition.

### Other antecedents and scope

- [Sundararajan et al., Axiomatic Attribution for Deep Networks](https://proceedings.mlr.press/v70/sundararajan17a.html)
  introduces integrated gradients. Attributor applies this established method
  to its gate variables and a declared zero-value reference. Neither path
  integration nor completeness is new.
- [Chefer et al., Transformer Interpretability Beyond Attention Visualization](https://arxiv.org/abs/2012.09838)
  is relevant Transformer-attribution work. It does not establish that
  Attributor's contextualized cached-value baseline corresponds to removing
  semantic evidence from the original video.
- [Astorino et al., Integrated Gradients as Proxy of Disagreement in Hateful
  Content](https://aclanthology.org/2023.clicit-1.7.pdf), CLiC-it 2023, uses
  integrated gradients of language models to identify textual constituents
  involved in hate and disagreement predictions. This is direct prior art
  against “first IG for hateful content,” but its text task does not satisfy
  this repository's specific hateful-**video** prior-use STOP condition.
- [Causal Intersectionality and Dual Form of Gradient Descent for Multimodal
  Analysis: a Case Study on Hateful Memes](https://arxiv.org/html/2308.11585v2)
  analyzes gradient-based attention/causal interpretations in hateful memes.
  It constrains broad multimodal-attribution novelty claims, but does not
  establish the proposed temporal video method.
- [IARE, Decoding Multimodal Cues: Unveiling the Implicit Meaning Behind Hateful
  Videos](https://arxiv.org/html/2606.11953v1), Section 4, uses multimodal
  information augmentation and preference-based reasoning improvement to
  generate decision rationales. This is relevant explainable video detection,
  not cached-value-path attribution.

The same-day primary readings of LELA, MultiHateLoc, HVGuard, RAMF, CLARA,
LEAF, and SAGE are recorded in the earlier Grounder/Selector review files.
They did not establish this particular method either. The result is a bounded
search finding, not exhaustive proof of novelty.

## Prior experiment inside the repository: acknowledge it

This is **not an unexplored high-level family**. The result note was read;
associated preregistration/report/code locations were identified:

- `archive/detection-2026-08/docs/duplex/TEMPORAL_ATTRIBUTION_PILOT_NOTE.md`
- Its identified preregistration:
  `archive/detection-2026-08/docs/duplex/PREREG_temporal_attribution_pilot.md`
- Report location:
  `archive/detection-2026-08/docs/duplex/reports/temporal_attribution_pilot.json`
- Code locations: `archive/detection-2026-08/scripts/duplex/` files
  `temporal_attribution_pilot.py`, `temporal_attribution_probe.py`,
  `temporal_attribution_analyze.py`, and associated launch/ASR/cohort scripts.

The old result note reports unsuccessful grad-times-input and late-attention
temporal attribution, with token density accounting for the apparent signal.
It also records substantial exclusions due to the old ASR/prompt reconstruction.
Those legacy numbers must not be merged with the current 4-fps evaluation.
The historical raw-output path named by that note is
`results/temporal_attribution_pilot/`; its present location was not verified.

The current declared computation is different: detached contextualized prefix
K/V, a shared value gate across layers, signed integration along an explicit
gate path, and current temporal mapping. No completed identical computation
was found. Record the family history and this difference before implementation;
do not describe switching to IG as sufficient evidence that the old failure
has been overcome. The user-authorized new candidate may be tested under the
current four-condition review rule.

## Necessary diagnostics and claim boundaries

1. **Token-density control.** Predeclare a CPU control using the identical
   modality/window membership weights with unit token contributions, including
   the same `N_windows` scaling, original global margin, and r6 procedure.
   Compare raw ordering and final outcomes if a qualifying gain appears.
   This directly addresses the repository's observed failure mode without new
   MLLM calls. Parent confirmed this declaration was added before implementation.
2. **Convergence of the local signal.** A small completeness residual is not
   sufficient numerical evidence: errors in individual token/window terms can
   cancel. The declared 16/32/64 smoke should report per-window signed
   contribution changes and rank stability, not just their total. Keep the
   integration rule based only on model outputs, and flag failures at the cap.
3. **What is conserved.** The sum explains `F(1)-F(0)` for this *internal
   intervention*, not the full original margin, a probability of hate, or a
   decomposition of all semantic evidence. Report the unaltered baseline
   margin `F(0)`, temporal remainder, signed modality totals, and mapping
   conservation. Shared gates must attribute all their layer uses jointly.
4. **Actual effect versus location.** Keep the wrong-time, sign, endpoint,
   raw-rank, and shift-only controls. A positive deletion test supports model
   sensitivity under value ablation; it does not establish GT correctness or
   a causal statement about raw-video removal. For highest-window versus
   half-offset deletion, report removed token/frame counts and empty controls;
   unequal evidence amounts can otherwise explain different margin drops.
5. **Joint computation is not proven joint semantics.** One global forward
   includes visual/speech context, but does not alone establish that the model
   correctly uses cross-modal interactions. The contribution is only as broad
   as the demonstrated localization gains and controls. Global wrong-verdict
   cases remain essential counterexamples.
6. **Cost is real.** The proposal correctly counts query backward passes and
   cumulative retries. Report actual smoke and corpus runtime/memory, including
   failed numerical attempts. “Frozen,” “cached,” and “no per-window calls” do
   not imply negligible inference cost.

Proceed to implementation and the separate rule-6 code review with the narrow
attribution claim and these diagnostics. No second proposal review is needed.
