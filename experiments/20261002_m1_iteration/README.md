# M1 autonomous iteration, 2026-10-02

User instruction: 进入自主迭代，修改第一个模块，要能够做出为什么涨点的机制。
Scope: modify frozen-MLLM reading, hold r6 inference/evaluator fixed. Continue
according to RESEARCH_ITERATION_RULES, with genuine mechanistic controls; no
paper polishing substitutes for empirical improvement. No new dataset or ensemble.
All development decisions are development-selected. Test GT is used only after
scoring for evaluation/error analysis, never fitting/routing/threshold selection.

Completed candidates:
1. `archive/experiments/20261002_m1_grounder/README.md`: late restriction had no
   qualifying gain; correct/wrong support almost indistinguishable.
2. `archive/experiments/20261002_m1_selector/README.md`: selective-head restriction
   changed routing but reduced final within by .0418 / .0150; no qualifying gain.
   Full baseline reproduction exact; controls beyond smoke not run per funnel.

3. `archive/experiments/20261003_m1_marginalizer/README.md`: proposal STOP;
   MARS/RAMF already use opposite hate/non-hate assumptions on the same media.
   No implementation or GPU run.

4. `archive/experiments/20261002_m1_attributor/README.md`: complete uniform FP32
   run, native parity/numerics passed; final within drops.1296/.0695 and every
   main metric drops. No control performance evaluated after failed primary gate.

5. `archive/experiments/20261003_m1_eraser/README.md`: actual media deletion and
   full re-encoding, native parity passed; all final metrics fell, within−.1312/
   −.1311. Raw ordering also worse. Archived with no extra controls.

6. `archive/experiments/20261003_m1_factorizer/README.md`: full native/explicit-
   causal/factor experiment complete; no qualifying gain. HCS finalwithin−.0700
   and raw−.0654; paired native exact. No further controls.

7. `archive/experiments/20261003_m1_contraster/README.md`: full333 paired run,
   native exact. Final within+.0001/+.0002; all13939 branches selected layer2.
   No qualifying gain; no additional controls or layer-pool revisions.

9. `archive/experiments/20261003_m1_allocator/README.md`: proposal STOP; published
   SHAP use in target video detection task; no implementation/GPU run.

8. `archive/experiments/20261003_m1_visual_contrast/README.md`: eighth candidate,
  clean/corrupt visual-window contrast with native speech/global retained.
  Proposal/code PASS; five-video GPU plumbing passes (292 exact native branches).
  Full333 R1 missed promotion because of HMM PR; cached matching controls
  supported localization only on HCS. R2 local-image corruption had no qualifying
  main gain and degraded visual raw ordering. Archived after that revision; no donor
  control or further schedule/contrast tuning. R1 best numbers remain recorded.

Running candidates:
- `experiments/20261003_m1_integrator/README.md`: tenth candidate, future-aware
  visual prefix memory from FutureMask; R1 full333 complete on sc448960, native
  exact. HMM within improves but pooled losses prevent promotion; both visual
  raw orderings fall. HMM gains are concentrated in two unchanged-raw-order cases.
  Numbers and sources remain in the candidate README and STATUS, not a third table.
  R2 future text keys only full333 complete, HMM within gain but pooled fails;
  HCS visual ordering improves without combined improvement. R3 declares
  new visual with native speech; cached complete-branch test complete: HMM pooled restored, HCS fails.
  R4 last modification declares same-window future ASR with native speech.
  Independent of VCD, not combined.
- `archive/experiments/20261003_m1_recycler/README.md`: candidate12, paper-defined VAR
  with explicit Qwen3 sink-channel adaptation; independent proposal/code PASS,
  real-input GPU smoke PASS, full333 complete; no qualifying gain, eleventh archive.

11. `archive/experiments/20261003_m1_amplifier/README.md`: candidate11, tenth
  archive. Complete333 PAI transfer had no qualifying main gain; HMM PR drops.
  Native reads exact, matched eager near baseline, no component controls after
  failed main gate.

Cumulative archives11 (nine performance/mechanism failures, two proposal novelty STOPs).

Initial candidate rationale:
Reason: all previous score-subtraction, hypothetical counterfactual questions,
generated hypotheses and latent common-offset decompositions failed. A previously
suggested layer-dependent attention intervention has not actually been run.
Read-only prior sources: STATUS; 20260928 headroom and its round-2 review;
20260927 error analysis; CVA archive; 20261002 verdict analysis and revisable-prior
archive. Distinguish failed scalar-cache proxies from an untested internal
computation. Each candidate gets a separate declaration, review and result record.

Acceptance: same metrics/gates on both main corpora; baseline-matched reading and
unchanged downstream inference; claimed mechanism must pass component ablation
and an intervention capable of falsifying its explanation. Each new reading cost
and all failures are recorded. Current paper remains unchanged during search.

Maintenance2026-10-03: independent Allocator proposal review encountered a
content-derived tie-breaking seed in the archived August
`project_shared_shapley_rank_transport.py`. Replaced it with fixed seed0 under
CLAUDE prohibition; script not executed, historical results not reinterpreted.

Unselected possibilities inspected while full jobs run, not additional candidates:
- Soft expected-answer embeddings (SoftThinking source previously noted) would
  differ mainly near uncertain native verdicts. Read only native z_video from
  `runs/20260926_glr/base_gridA/predictions.jsonl`, no GT: smaller Yes/No probability
  exceeds.05 in18/215 HMM and14/118 HCS, and its median is about2e-6 in both.
  This is an input-distribution observation, not a performance screening result;
  no temperature, routing threshold, method or experiment was selected.
- PAI, [arXiv2407.21771](https://arxiv.org/abs/2407.21771), is another known
  input-attention/contrastive-decoding source. Only abstract/source discovery
  was inspected at this point; no claim of target-task novelty or implementation.
  At that time the running work was visual-contrast R2 and Integrator R1;
  visual contrast has since been archived. PAI has now been fully source-read
  and separately declared as candidate11; SoftThinking remains unselected.
- VAR was inspected during Integrator R2 / Amplifier R1 collection and is now
  separately declared as candidate12: `archive/experiments/20261003_m1_recycler/README.md`.
  Independent proposal review PASS; paper/code differences and Qwen3 channel
  adaptation declared before implementation, no full-run performance read.

Candidate13 proposal: `experiments/20261003_m1_reinforcer/README.md`, VISTA
paper-defined residual steering + preceding-layer logits, Qwen local-query scope.
Author-code discrepancies are declared explicitly. Independent proposal review
PASS; implementation and own actual-model checks complete, independent code review underway. VTI/PTI were only inspected
as adjacent steering sources, not declared as candidates or run; no external data
has been added. Pairwise Ranking Prompting was located but not yet method-read.

Candidate14 proposal: `experiments/20261003_m1_projector/README.md`, paper-defined
ACG transfer, orthogonal attention-output correction with same-pass masked
reference. Explicit per-head Qwen3 adaptation and matched last-row eager control;
proposal review underway. This is distinct from residual steering, sink transfer
and image-logit boosts. PRP was not selected after finding the prior PWC failure.
