# Independent proposal review: M1 Selector

Date: 2026-10-02. Reviewer: independent agent
`/root/m1_grounder_proposal_review`, same model as the main session.
Scope: the single rule-4 review of the new candidate
`experiments/20261002_m1_selector/README.md`. This does not reopen the archived
Grounder proposal review. No method code or experiment was changed/run.

## Decision: PASS

No permitted STOP condition is established. Selective attention intervention
is already a research mechanism in other VLM tasks; the proposed contribution
can only be its specific label-free temporal-support selection and useful
transfer to hateful-video localization. Whether that transfer works remains
an experimental question.

| Rule-4 STOP condition | Assessment |
|---|---|
| Source mechanism already used for hateful-video detection/localization | Not established by the actual searches and primary papers below. The closest sources concern visual grounding, narration, VQA, or robot actions. |
| Pure ensemble | No: one frozen model with interventions inside its forward pass. Controls are separate predictions, never combined. |
| Pure calibration/postprocessing/smoothing | No: masking changes the computation that creates local logits. The shift-only output is an explanatory control, not the deployed method. |
| Pure engineering trick | No: content-dependent head selection and selective temporal support restriction form a testable inference mechanism. A new kernel or fewer model calls is not the proposed contribution. |

Selecting heads by affinity may fail to identify useful information routes.
That uncertainty is precisely what the declared wrong-head/support controls
test; it is not an additional STOP ground.

## Actual search and primary sources

Searches performed on 2026-10-02 included:

- `"hateful video" "attention head"`
- `"hateful video" "head selection"`
- `"hateful video" "selective attention"`
- `"hateful video localization" "attention"`
- `"hateful" "LocalizationHeads"`
- `"vision language" "head selection" grounding attention intervention local context`
- `"Your Large Vision-Language Model Only Needs A Few Attention Heads"`
- `"Towards Training-free Multimodal Hate Localisation" arxiv`

Primary papers were opened; findings below do not rely on secondary summaries.
Search absence is a bounded finding, not a proof that no unpublished or
unindexed target-task work exists.

**Closest source: [Gaze Heads: How VLMs Look at What They Describe](https://arxiv.org/html/2606.14703v1).**
Its Sections 4–6 identify heads through attention changes across image-panel
queries and steer selected heads toward a chosen panel. The intervention
retains text-token access and includes random/non-gaze and all-head controls.
It studies narration/visual QA, including Qwen3-VL-8B, rather than hateful-video
localization. Selector differs by using the current branch's local-versus-remote
per-token affinity online at every layer, temporal frame/transcript support,
and no offline top-K head discovery. This is a close antecedent, not a distant
inspiration: cite it explicitly. Do not claim that sparse attention support
intervention or the wrong-head/all-head experimental logic is newly invented.

Other relevant primary sources:

- [Localization Heads](https://arxiv.org/html/2503.06287v1), Sections 3–5,
  uses final-prompt-token attention and attention concentration criteria to
  identify a few heads for training-free visual localization. It mainly reads
  attention as a localization signal; Selector instead modifies the original
  model's Yes/No computation using known temporal support.
- [IGAR](https://arxiv.org/html/2603.06001v1), Section 4.2, selects head-query
  pairs using attention criteria and reallocates attention within the forward
  pass for robot instruction grounding. Thus automatic per-input head
  selection for attention intervention is also established. Its sink-based
  criterion, instruction target, and manipulation task differ from Selector.
- [Retrieval Heads Meet Vision](https://arxiv.org/html/2608.27417v1), method
  and Appendix E, studies visual retrieval heads, reference-region attention,
  and head ablation on grounding and VQA benchmarks. Selector must not describe
  a positive affinity test as sufficient proof of a head's causal role; its
  matched intervention controls must supply the evidence.

**Target-task checks.**

- [LELA](https://arxiv.org/html/2602.09637v1), Sections 3.2–3.4, is directly
  relevant training-free hate localization. It combines modality captions,
  staged LLM prompting, and frame scores; it does not implement online
  selection and temporal masking of internal MLLM attention heads.
- [RAMF](https://arxiv.org/html/2512.02743v1), Section 3.5, appears in the
  attention-head target-task search. Its learned cross-head convolution and
  structural mixing operate in multimodal feature fusion. They are different
  from selecting heads in a frozen MLLM using target-window affinity and
  restricting their cached evidence support.
- The same-day primary-paper readings of MultiHateLoc, HVGuard, CLARA, LEAF,
  and SAGE are recorded with direct sources in
  `docs/reviews/20261002_m1_grounder_proposal.md`. None supplies the specific
  head-selection/intervention mechanism here. This record is reused as
  literature evidence, not as a reopened review of Grounder.

Under the repository's explicitly permitted cross-task transfer novelty
criterion, these sources support PASS. Public novelty wording must acknowledge
the close Gaze Heads and IGAR precedents.

## Difference from the failed candidate and previous experiments

Read `runs/20261002_m1_grounder/r1_full_analysis/summary.json` as development
evidence, and searched experiment/archive/review/wiki Markdown for prior
head-selection or selective-head interventions. No completed identical
experiment was found.

Grounder imposed a fixed support restriction across every head of selected
layers and also removed direct global Q/A access. Selector chooses heads
from each branch's current affinities, operates across depths, and retains
global Q/A and every unselected head. These are substantive computational
changes, not a renamed layer-depth sweep. Grounder's failure motivates but
does not establish Selector's explanation.

## Minimal clarifications before interpreting results

The proposed base, all-head, replayed wrong-head, replayed wrong-support,
raw-ranking, and shift-only comparisons are capable of falsifying the intended
explanation. No new general review round or mandatory extra model arm is needed.
Make the following small declarations/reporting additions:

1. **Affinity timing.** “Ordinary affinities” must mean the current layer's
   affinities before its new support mask, along the current intervention
   trajectory. Later layers already receive states changed by earlier layers;
   these are not counterfactual affinities from an untouched baseline model.
   Selection cannot use already masked local/remote attention, which would
   make the criterion self-fulfilling.
2. **Informative wrong-head comparison.** Save selected-head identities/counts
   by layer and branch. Report overlap with the half-rotated set, plus empty
   and all-head selection rates. A half rotation can reproduce much of the
   original set, sometimes all of it. If overlap dominates, a null comparison
   is inconclusive about head identity; do not describe it as a strong wrong-
   head test. The declared replay is important: recomputing selection inside
   the control would confound changed head identity with changed routing.
3. **Routing scope.** Choosing a mask using the last query token and applying
   it to all query positions is whole-query-conditioned routing. Maintaining
   the ordinary causal key mask is necessary, but does not make earlier query
   states independent of the later question text through this routing rule.
   That is compatible with scoring a fully observed question and is not label
   leakage. Describe the computation accurately; no mechanism change is
   demanded for the current use.
4. **Claim scope.** Local-versus-remote affinity is a selection statistic, not
   proof of semantic relevance, sparse specialization, or removal of verdict
   error propagation. Keep the existing qualification that global Q/A and
   contextualized representations remain accessible. A positive result should
   connect improved raw temporal ordering to head-identity/support controls;
   decoded gains alone or mask visualizations are insufficient.
5. **Decision and cost.** Apply the continuation/promotion gate to the final
   decoded method; report raw scores as diagnostics. The declared two-arm
   screening and conditional controls fit rule 9. Report actual per-video
   runtime and mask overhead even though model-forward count is unchanged.

Proceed to implementation and the separate rule-6 code review. No outcome is
predicted or endorsed by this proposal PASS.
