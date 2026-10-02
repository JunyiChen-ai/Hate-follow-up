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

Running candidates:

- `experiments/20261003_m1_factorizer/README.md`: sixth proposal, prefix
  temporal-block encoding with ordinary global query access. Independent proposal
  and code reviews PASS; five-video GPU smoke passed, including largest prefix.
  Complete primary includes native/explicit-causal/factor arms to separate kernel
  drift from the proposed computation. No performance read yet.

- `experiments/20261003_m1_contraster/README.md`: seventh candidate, one-forward
  language-depth contrastive readout. Proposal PASS; CPU plumbing passed and
  independent code review PASS. Five-video GPU checks passed; full333 paired run started on sc448960 at03:01NZ.

Cumulative archives5 (four performance failures, one proposal novelty STOP).

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
