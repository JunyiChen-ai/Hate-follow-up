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

Running candidates:
- `experiments/20261003_m1_visual_contrast/README.md`: eighth candidate,
  clean/corrupt visual-window contrast with native speech/global retained.
  Proposal/code PASS; five-video GPU plumbing passes (292 exact native branches).
  Full333 completed: within+.01961/+.01066, HMM PR−.01410; retained under rule9,
  not promoted. Four cached controls complete: matching supported only on HCS.
  R2 local-image corruption passes independent code and five-video GPU checks;
  complete333 running on sc474399, estimated35min.
- `experiments/20261003_m1_integrator/README.md`: tenth candidate, future-aware
  visual prefix memory from FutureMask; proposal/code reviews PASS, CPU actual
  model witnesses passed; preparing GPU smoke. Independent of VCD, not combined.

Cumulative archives8 (six performance failures, two proposal novelty STOPs).

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
