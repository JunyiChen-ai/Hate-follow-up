# M1 Attributor independent code review

Date: 2026-10-02. Reviewer: independent same-model agent instance
`/root/m1_grounder_code_review`. Scope: rule 6, the Attributor reader,
attribution engine, analysis, and launch scripts. No method code was changed,
no GPU job was started, and no true GT or localization performance was read by
the reviewer.

**Code decision: PASS.** No unresolved gradient, cache-isolation, temporal
mapping, label-leakage, or evaluator bug found. **Numerical readiness remains
separate:** the initial GPU smoke exposed local integration instability, and
the revised 16/32/64/128/256-node smoke must establish that its accepted output
is sufficiently stable before the full performance experiment is interpreted.

## Gradient and cache checks

Reviewed `attributor.py`, `measure.py`, `analyze.py`, `launch/`, the shared Judge
and token-region helper, and the installed DynamicLayer/SDPA implementation.
Independent executable checks are saved at
`runs/20261002_m1_attributor/independent_review/check_attributor.py` and
`check_attributor.json`.

- A single fp32 leaf gate per prefix token is reused across every language
  attention layer and KV head. Multiplication happens before the ordinary SDPA
  value aggregation. Non-media gates are exactly one; their leaf gradients
  are zero. All layer uses contribute to the same gate gradient. Query hidden
  states, subsequent query K/V, and the fp32 Yes/No output head retain the
  necessary backward paths even though model parameters are frozen.
- On a real small Qwen3 text model in fp32, the summed shared-gate derivative
  agrees with an independent central finite difference to **7.36e-6**. A
  separately computed 32-node integral has completeness residual **3.42e-7**.
  These checks exercise the actual shared-gate computation, not a mock gradient.
- Both fp32 and BF16 CPU tests produce finite nonzero media gradients, exactly
  zero non-media gradients, and exact unit-gate/native output equality.
  Repeated points give identical gradients and scores. Frozen parameter
  `.grad` fields remain empty. BF16 finite gradients do not themselves prove
  numerical integration convergence; the GPU refinement addresses that issue.
- `cache_branch` makes new cache/layer objects while sharing only initial
  detached tensors. DynamicLayer updates assign newly concatenated K/V tensors;
  gate multiplication and value concatenation also allocate new tensors. The
  original cache length and every original K/V element remain unchanged in
  independent checks. The helper explicitly rejects other layer types and
  attached initial tensors rather than silently assuming they are safe.
- Prefix encoding/global native baseline run with the attention wrapper
  inactive. Attribution branches contain the original global question, without
  appending the selected answer. Native attention masks and the SDPA function
  are retained; only cached prefix values are gated. The final margin uses
  exactly the Judge's fp32 label-token log-sum-exp calculation.

The independent CPU environment was Torch 2.7.1 / Transformers 4.57.6. The
returned deployment smoke uses Torch 2.11.0 / Transformers 5.15.1 on sc474399.
The reviewer read `runs/20261002_m1_attributor/r1_smoke/checks.jsonl`: all four
videos have endpoint/native difference zero and exact cache immutability.
Thus deployed BF16 backward/native compatibility is exercised, not inferred
solely from the older-version CPU tests.

## Temporal accounting and scoring

- The shared token-region implementation checks exact processor expansion and
  uses the established frame timestamps and proportional ASR-word mapping.
  Keys and detached contextualized prefix encoding remain unchanged.
- Each media token's contribution is divided by its number of temporal
  memberships. Independent hand-calculated signed examples verify shared
  timestamp splitting, modality separation, the common `N_windows` multiplier,
  and the unassigned remainder. Omitting a speech readout leaves its signed
  contribution in the reported remainder rather than assigning it to visuals.
- `token_sum` and `assigned_sum` are in unscaled attribution units. The emitted
  modality reads are multiplied by the number of windows. Their sum is not
  claimed to equal the original margin; the underlying sum approximates
  `F(1)-F(0)` under the declared internal value intervention, with a temporal
  remainder. The max over observed modalities is a separate declared readout.
- Empty visual support gives zero visual contribution; empty original speech
  windows omit speech. Output construction retains native `z_video`, the
  established branch schema, and the original 8-second/4-fps mapping. No
  gradient magnitude, label, fitted threshold, or external model is introduced.

## Numerical revision within this review

The original smoke's `non_hate_video_4` passed total completeness only at
64 nodes but had window Spearman 0.90 for 16 versus 64 and 0.70 for 32 versus
64. Its 32/64 token relative L1 difference was 0.1424. The other three videos
had 16/64 token relative L1 below 0.017 and window correlations at least 0.9995.
These are numerical findings without GT, not evidence for or against method
quality. The initial total-sum check alone was insufficient.

The implementation agent revised only the declared numerical procedure;
checked the final code and corresponding README/config updates:

- Evaluate 16, 32, 64, 128, 256-node Gauss–Legendre grids in order.
- Accept the first grid with completeness error at most
  `max(.25, .05 * abs(F(1)-F(0)))` **and** relative L1 change from the preceding
  complete token-attribution vector at most `.05`. The earliest acceptance is
  therefore 32 nodes. Comparisons use model outputs only.
- Full inference stops at first acceptance. At the cap, a failure is recorded
  and stops the run before admitting predictions for that video; it is never
  treated as a reason to exclude a difficult video. Attribution is not rescaled
  to force the completeness sum.
- Smoke retains every grid, uses the same first-acceptance output, and reports
  token L1 plus max/mean window changes and rank correlation against 256 nodes.
  Global sum and vector convergence are complementary checks; smoke window
  stability still matters for the localization signal.
- Forward/backward accounting correctly includes all lower-grid retries and
  the endpoint backward. At earliest acceptance the attribution method uses
  51 forwards including prefix and 49 backwards. Smoke executes all 496
  quadrature nodes regardless of early acceptance; deployment cost must use
  `numeric.deployed_seconds`, not its full `attribute_seconds`. Full-run timing
  and deployed counts then refer to the same executed grid sequence.

The refined GPU smoke is pending at this report entry. No code-level defect
requires a method change, but its numerical result should determine readiness
for a full run; this review does not certify convergence from the new guard
alone. Absolute completeness tolerance can dominate when `F(1)-F(0)` is small,
so report the actual residual and baseline margin, not an exact decomposition.

## Analysis and launch path

- Preparation checks full manifest coverage, matched raw arms, unchanged
  globals versus the current reader, rates/lengths/finite scores, identical
  window boundaries, and speech-branch availability. Baseline window
  discrepancies are recorded against `base_gridA`.
- The report additionally requires complete paired decoded outputs and
  matching global/rate/length/finite values. It rejects incomplete numerical
  checks or any video lacking numerical acceptance.
- Both raw and final metrics come from the canonical evaluator. The decoder
  CLI exactly matches current r6: `--noleak --transform nscore --key calib
  --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`.
  Per-arm fits remain label-free; GT enters only evaluation or reporting.
- Confidence intervals resample paired video-level differences, seed 0,
  2000 resamples as declared. The continuation gate tests any final main-metric
  gain of `.01`; the stricter performance gate requires within improvement
  `.01` on both corpora, with no `.005` pooled / `.01` within drop, versus both
  the paired base and current r6. `mechanism_supported` remains false pending
  the declared explanatory controls.
- Separate CPU arms write separate logs, raw metrics, and decoded directories.
  Preparation completes first, both process exit statuses are checked, and
  report executes only after both finish successfully. Cost and numerical
  summaries retain forward/backward calls, integration node counts, remainder,
  endpoint error, and observed peak memory.

No part of these checks licenses claims about raw-video causality, semantic
cross-modal interaction, or correctness of an attribution merely because it
explains the model. Those remain the declared empirical tests and claim limits.
