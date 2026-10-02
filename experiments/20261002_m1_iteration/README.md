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
  Full333 completed: within+.01961/+.01066, HMM PR−.01410; retained under rule9,
  not promoted. Four cached controls complete: matching supported only on HCS.
  R2 local-image corruption final within+.00296/−.00669, visual raw ordering
  degrades;44.34min actual. Archived after this unsuccessful revision; no donor
  control or further schedule/contrast tuning. R1 best numbers remain recorded.

Running candidates:
- `experiments/20261003_m1_integrator/README.md`: tenth candidate, future-aware
  visual prefix memory from FutureMask; R1 full333 complete on sc448960, native
  exact. HMM within+.01604 but PR-.01263; HCS within-.00130 and ROC-.00718.
  Both visual raw orderings fall; HMM two cases with unchanged raw within account
  for.01476 of the.01604 gain. R2 declared: future text keys only, preserving all
  other settings; implementation review underway. Independent of VCD, not combined.
- `experiments/20261003_m1_amplifier/README.md`: candidate11, PAI attention
  amplification plus language-only token-logit reference, independently declared
  before Integrator evaluation. Proposal review underway, no implementation yet.

Cumulative archives9 (seven performance/mechanism failures, two proposal novelty STOPs).

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

Shared post-scoring report helper added while both main runs were still in progress:
`scripts/analysis/m1_branch_diagnostics.py`, independent narrow review PASS
(`docs/reviews/20261003_m1_branch_diagnostics_code.md`). It calls the canonical
within metric on each branch's available frames, reports different speech counts,
and separates single-read-window final gains from multiwindow gains. Uses original
window boundaries and canonical shared GT/prediction extent; no evaluator change.
Only completed VCD R1 raw/decoded records and its test GT were read to validate it;
all branch n/means agree with its existing control report within1e-12. No new method
or constant was selected from this check. Run this diagnostic after each complete
new evaluation, with all declared arms; keep it separate from primary metrics.
