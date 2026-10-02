# Revisable prior independent code review — PASS

Date: 2026-10-02. Independent reviewer instance under `RESEARCH_ITERATION_RULES.md` rule 6.
Reviewed implementation: `experiments/20261002_revisable_prior/model.py`, shared kernel `src/temporal.py`, and its compatibility wrapper in `experiments/20260926_twolevel/twolevel_r2.py`.

**PASS for the declared cached pilot.** No bug that changes the intended experiment or invalidates its observations was found. This is a code-integrity decision, not a claim of effectiveness or promotion. The reviewer did not alter method code, parameters, or the proposal.

## Findings

- **The mechanism reaches the final score.** `terms` includes the shared offset in the global and both local likelihoods, then marginalizes the offset and video regime. Its conditional local probability and posterior video log odds feed `score_curve` directly at model lines 226–238. With local inputs and parameters fixed, a synthetic change in the global observation changed conditional local probabilities by up to 0.4638; the corresponding change is zero in the independent arm. This checks implementation of dependence, not whether that change improves localization.
- **Joint EM sufficient statistics are retained.** Global observations enter once per video. Local state-zero responsibilities include V=0 and the conditional state-zero mass under V=1; state-one responsibilities include only V=1. The six statistics retain observation, node, squared-node and observation-node cross-moments. Chain start statistics are weighted by V=1 responsibilities. Means/tau are updated conditional on the previous variances and variances are then updated conditional on those coefficients: a valid conditional-maximization sequence, with likelihood decrease detection.
- **Tau reflection is consistent.** Variance residuals use the signed least-squares coefficient before its sign is reflected. Symmetric Hermite nodes/weights make positive and negative tau the same marginal model. A deliberately negative-coefficient test matched an independent explicit weighted least-squares solution to 1.33e-15. Direct sign reflection left evidence, regime odds, conditional/unconditional probabilities and posterior offset mean unchanged to 1.11e-16.
- **Modality OR precedes integration over the common offset.** The per-node duration-averaged chains are fused conditionally, then averaged with the posterior node weights normalized within V=1. Unconditional probabilities equal the conditional probabilities multiplied once by the posterior regime probability. There is no repeated regime factor and no OR of already offset-marginalized probabilities. The author's separate explicit enumeration over regime, node, both chain paths and duration choices matches this calculation to 1.11e-15; its test source and output were inspected.
- **Time and video alignment are preserved.** All 215 HateMM and 118 HateClipSeg records match `data/omsl_v6_inputs/manifests/all_test.jsonl` exactly, without extra/missing IDs. The new loader's 4-second cells, one/two-cell window assignment and 4-fps output mapping equal the existing runner on every record. The real caches include truncated final windows in 215 and 116 videos respectively; these pass the same mapping checks. All transformed observations are finite. Input keys are `(dataset, video_id)` and output inherits them from the same record.
- **GT is isolated.** Loading, normal-score transforms, initialization, EM and inference use cached MLLM reads only. The module invokes `src.eval.evaluate_four_datasets` after prediction writing is complete. Its default split is test, and datasets are passed explicitly. No copied evaluator or label-dependent threshold is present. Review checks used cached predictions, test-manifest identifiers and synthetic values; they did not read GT arrays or compute performance metrics.
- **The shared-kernel extraction preserves the old runner.** The syntax trees of `build_chain`, `fb`, `bma_fb` and `length_grid` are identical to the corresponding source in the staging index before extraction. The wrapper continues to pass the old runner's dynamically selected modalities. The newly added batched routine agrees with scalar model averaging across all four chain kinds, with maximum error 4.44e-15 over 240 comparisons. This comparison used direct source syntax, not content digests or Git identifiers.

## Execution evidence

Reviewer command:

```text
/home/jehc223/miniconda3/envs/HateVideo/bin/python runs/20261002_revisable_prior/review/checks.py
```

Artifacts:

- `runs/20261002_revisable_prior/review/checks.py`: independent reviewer checks.
- `runs/20261002_revisable_prior/review/checks.json`: numerical and complete-cache alignment results.
- `runs/20261002_revisable_prior/review/coverage.json`: exact test-manifest coverage.
- `runs/20261002_revisable_prior/review/em.log`: ten synthetic EM iterations for each of full, independent and no-global arms; all likelihood sequences are nondecreasing.
- `runs/20261002_revisable_prior/checks/checks.json`: author's separate joint path-enumeration and smoke-test results, inspected alongside `experiments/20261002_revisable_prior/checks.py`.

Additional checks passed: tau=0 matches the candidate's independent arm; no-global inference is invariant to the global observation; missing speech produces finite posteriors and empty speech observation statistics.

## Interpretive limits to retain in the run report

1. **Missing speech is a missing observation, not a forced off state.** The candidate keeps an unobserved speech chain in 14 HateMM and 3 HateClipSeg videos and includes its prior marginal in the OR. This matches current r6 behavior and the declared two-chain generative model, so it is not a new implementation bug. Do not claim that no-speech videos reduce exactly to a visual-only model.
2. `ordered_means`, `converged` and the frozen-parameter quadrature diagnostic are logged rather than automatically stopping evaluation. The run analyst must enforce the declared semantic-failure and 7→15→31 numerical-refit decisions before interpreting performance. Seven-node quadrature and ten-iteration synthetic smoke checks are not proof of adequate full-corpus integration or convergence.
3. Comparisons to r6 use the older matched Reader family. The independence arm isolates the shared random effect within this candidate; it is not expected to reproduce every historical r1/r6 choice numerically. The current paper and main method remain unchanged unless empirical gates pass.

No repair was required and no second general review is requested.

## Numerical repair confirmation: continuous quadrature and sparse forward–backward

Later on 2026-10-02, the parent requested confirmation of two specific numerical changes after the original fixed-node fits failed their declared integration checks. This follow-up does not reopen the mechanism review. **PASS for both numerical changes.** The reviewer changed no method code or settings and ran no full fit or performance evaluation.

Reviewed changes:

- `adaptive_quadrature` in `model.py` encloses the means of every possible path-conditional Gaussian posterior of u, adds eight common conditional standard deviations at both ends, and uses trapezoidal spacing of at most one such standard deviation. Its density weights are `phi(u) * du`, with half weights at endpoints, **without renormalizing the retained prior mass**. The bounds use the same observation counts and variances as the likelihood, including omission of the global observation in the no-global arm. They are correct for the nonnegative tau maintained by the M-step. This is a posterior-width-aware trapezoidal quadrature, not an adaptive-subdivision or single-mode Laplace approximation.
- `src/temporal.py` keeps the scalar and NumPy references, and optionally JIT-compiles sparse forward/backward recursions with `cache=False`. The union of nonzero transition edges across duration choices is used with the corresponding weights for each choice. Both transitions' direction and backward scaling are correct. Duration posterior averaging and state normalization are unchanged.

Independent executable evidence is in `runs/20261002_revisable_prior/review/adaptive_checks.py` and `adaptive_checks.json`:

| Check | Coverage | Maximum absolute error |
|---|---|---:|
| Sparse versus NumPy kernel | 36 cases; all four chain kinds; k=1,2,4; n=1,3,17 | 2.85e-14 |
| Continuous step 1 versus step .25 | 128 cases; k=1,2,4; n=1,3,9; tau=0,.1,.8,2.5; full/no-global; complete, missing speech, partly missing and both missing observations | conditional posterior 1.20e-9; sufficient statistics 1.06e-6 |
| Step 1 versus exact continuous path integration | 96 small-chain cases; all path likelihoods integrated analytically before posterior averaging | log evidence 5.36e-9; conditional posterior 1.20e-9; sufficient statistics 4.23e-7 |
| Complete inference with sparse versus NumPy kernels | Integrated probabilities, regime odds, state marginals, start statistics and all six joint emission statistics | 7.11e-15 |
| Frozen real-data step 1 versus .25 | Three deterministic observation-count-selected videos per corpus, including a 250-cell video; q15 full parameters | conditional posterior below 7e-10; sufficient statistics below 3.37e-7 |

The analytic test enumerates both regimes, both modality paths and duration choices. For each component it integrates the Gaussian product exactly, retaining its conditional first and second u moments, then accumulates all six EM statistics, initial phase statistics, full augmented-state marginals, and the conditional/unconditional OR. It therefore independently tests the continuous posterior and cross-moments, rather than merely comparing two resolutions of the same implementation. Neither conditional OR nor EM logic was changed by the new numerical integration.

A separate tail-interval test puts the entire retained interval at u in approximately [4.055, 5.846], where the retained prior mass is approximately 2.5e-5. Evidence still matches analytic path integration to 2.52e-9, confirming that the interval was not spuriously normalized to prior mass one. Its prior-density quadrature mass alone need not be highly accurate: the numerical spacing is designed for the narrower posterior product. The jointly weighted likelihood integral is the quantity checked.

The six real-data tests also preserve the final frame-rank contribution exactly between step 1 and .25; results are in `runs/20261002_revisable_prior/review/adaptive_rank_checks.json`. These are numerical checks only and do not use GT or performance metrics. Full-run convergence and final frozen-parameter step-1-versus-.5 checks remain necessary at the fitted solution; coarse GH convergence did not establish continuous-model convergence.

## Numerical acceleration confirmation: PX-EM

The parent subsequently requested a targeted check of the continuous-model PX-EM acceleration after ordinary EM remained slow. **PASS.** This is a parameter-expanded solver for the same observed-data model, not another mechanism arm.

The E-step collects `E[u]` and `E[u²]` once per video, summing the joint posterior over both V regimes. Division by the video count gives expanded prior mean c and variance s². After the existing regression and residual-variance updates, the reduction is `mu_active += tau_signed*c`, `tau = abs(tau_signed)*s`. Substitution `u=c+s*u_standard` establishes the marginal-model equivalence; a negative coefficient can then be reflected under the symmetric standard-normal prior. The code correctly shifts all six active means in the full arm and only the four used local means in the no-global arm, retains the signed coefficient until after the shift, and leaves observation variances, regime weight and chain starts unchanged by reduction. The second moment includes posterior uncertainty, not just variance of posterior means. The independent arm is unaffected. `fit` now asserts that PX uses continuous quadrature; the same equivalence must not be assumed for a fixed discrete GH approximation.

Independent tests in `runs/20261002_revisable_prior/review/px_checks.py` and `px_checks.json` establish:

- Sixteen expanded/reduced cases cover both arms, positive and negative regression coefficients, nonzero prior means and different prior variances. Evidence and conditional/unconditional posterior discrepancies are at most 3.56e-15; joint sufficient statistics after the corresponding latent-coordinate transformation differ by at most 1.43e-14.
- The implemented M-step reduction exactly matches the expected signed shift and scaling, including negative coefficients. Residual variances, regime weight and chain-start probabilities are unchanged by reduction.
- Twelve synthetic PX iterations in each arm are monotonic: full likelihood rises from -114.1215 to -96.3105 and no-global from -99.1792 to -84.2068. These are synthetic likelihoods, not dataset performance metrics.
- Five iterations of the independent arm are identical with PX enabled or disabled.

No full corpus fit, GT inspection or metric-based selection was performed by the reviewer. The production continuation must retain the declared likelihood stopping rule, per-iteration monotonicity check, and final continuous-quadrature accuracy check. A faster solver does not itself establish empirical benefit or justify reporting unconverged runs.
