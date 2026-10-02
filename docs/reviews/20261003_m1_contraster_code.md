# Independent code review: M1 Contraster

Date: 2026-10-03. Reviewer: `/root/m1_grounder_code_review`.
Scope: one rule-6 review of `experiments/20261003_m1_contraster/{contraster,measure,analyze,selfcheck}.py` and `launch/`, against the README/proposal. No candidate implementation was edited, GPU or benchmark evaluation was launched, or pending Eraser/Factorizer predictions, performance or GT were read.

## Decision: PASS for the declared numerical/cost GPU smoke

One blocking analysis bug was found and corrected by the author: `report` used the nonexistent `r['erase_raw']` when computing `raw_delta`. It now uses `r['contrast_raw']-r['base_raw']`. I inspected the correction and exercised the complete report with isolated synthetic labels and stubbed metric records; it finishes and records the expected raw paired differences. These fixture values are not benchmark results. No other observation-changing bug requiring a fix was found.

## Readout and native-state preservation

Hooks attach to decoder blocks indexed `n-1` for declared depths `2,4,...,34`, giving 17 candidates on the 36-layer model. They are enabled only for the existing cached local-query forward and capture a detached clone of its last token's post-block residual state. They neither read a generated answer token nor intervene during prefix/global/answer computation. Although Qwen3-VL has DeepStack injections during media prefilling, the cached query call supplies no image/deepstack features; the captured query states therefore have the declared post-block meaning.

The model's final RMSNorm is applied once to the candidate-state batch. The mature state returned by `_step` already passed that norm and is not normalized again. Candidate and mature rows use the same frozen FP32 LM-head copy and full vocabulary. Standard JS is calculated as the two forward KL terms against the mixture, with stable log-softmax/logaddexp arithmetic. Selection uses the complete vocabulary and first `argmax`, which implements the shallowest exact tie. It is not the reversed-KL expression in the source implementation or a Yes/No-only divergence.

The selected token logits are subtracted before Yes/No variant aggregation. Shared log-normalizers cancel in that margin; it is generally different from subtracting two already aggregated class margins. The saved alternative calculation records finite-precision log-probability/logit equivalence error. Batched full-head mature projection may differ numerically from the native restricted-row projection; `measure` records this drift and retains the **original restricted-row margin** for the native arm, rather than replacing it with the batch value.

Both output arms use the same untouched native global margin and forced answer. They share the original forward, and branch suffixes crop back to the original prefix/global/answer length. Hooks return no replacement output and do not change parameters or KV values. Visual/speech prompts, missing-speech behavior, raw maximum, window boundaries and 4 fps expansion are unchanged. No label or GT enters scoring or layer selection.

## Logging, evaluation and cost

Each branch records window index/modality, native and contrastive margins, all 17 candidate depths and JS values, selected depth/index, 18 rows of Yes/No logits, 18 full-vocabulary log-normalizers and explicit label-token IDs. This preserves enough information to reconstruct fixed L18, the declared premature-window permutation and mature/premature variant probabilities without another transformer pass. JSON round-trip and fixed-L18 reconstruction were independently checked.

`prepare` requires complete paired manifest/check coverage, exact native global/window parity versus `base_gridA`, finite 4 fps curves, matched windows/speech availability, branch counts and recorded forward counts. It reconstructs selected contrast margins from logged token logits. Analysis calls the sole evaluator and unchanged r6 (`--noleak`, `nscore`, `calib`, BMA duration, length prior, minimum 2 windows, grid 6, `m2`), with independent arm outputs/fits. It does not duplicate ROC/PR/within metric formulas. Reporting checks decoded coverage, finite values, rate, length and original global before reading GT; bootstrap resamples paired video differences with seed 0 and 2,000 draws. The corrected raw difference, final comparison directions and gates match the declaration. Mechanism support remains false pending controls.

Outer transformer work is `3+B` in deployment and in the paired full measurement, with no backward pass; smoke adds B ordinary reference queries for `3+2B`. Each branch still pays for 17 intermediate and one mature full-vocabulary projection, norm/JS and recorded readout. The FP32 head allocation/setup is a one-time cost recorded separately in config, not included in per-video steady-state totals. Shared baseline time includes state-capture overhead and is explicitly an upper bound; it must not be presented as a clean native benchmark or used to claim exact native-relative overhead. Peak memory includes the FP32 head copy. Full-corpus totals and GPU batch-projection drift remain to be measured in the declared environment.

## Independent CPU evidence

Test/result: `runs/20261003_m1_contraster/independent_review/check_contraster.{py,json}`. CPU only, CUDA disabled; torch 2.7.1+cu128 / transformers 4.57.6. Deployment uses a different reviewed environment, so actual CUDA parity/cost remain smoke requirements.

- An actual 36-layer small Qwen3-VL text model, in FP32 and BF16, reproduces all 17 captured last-query states against returned intermediate states. Native final hidden state, full KV tensors and weights are exact; candidate norm executes once, mature is not renormalized, and hook removal restores ordinary computation.
- Full-vocabulary JS agrees with an independent float64 reference. A deterministic asymmetric-distribution witness gives a different selection under reversed KL and under label-restricted JS, while the implementation matches standard full-vocabulary JS. Token-first contrast is separately distinguished from class-margin subtraction. Identical distributions use the first candidate.
- Logged JSON reconstructs the fixed L18 margin with the same full-vocabulary-normalizer cancellation. An actual small-model `read_video` run checks the shared forwards, ordinary reference parity, cache reuse/crop, original global, missing speech, branch-index schema and 4 fps output. For B=3, deployment is 6 calls and smoke is 9. Its measured mature projection drift is approximately `3.39e-8`, a synthetic CPU value only.
- The corrected report finishes using synthetic labels and stubbed metric records; no benchmark GT or actual metric file is accessed. Both launch scripts pass shell syntax checks.

The author's six-layer self-check and saved `selfcheck/invariance.json` were also inspected. Neither JS magnitude nor selected depth establishes an early-topic/late-hateful-evidence interpretation, and omission of plausibility truncation can amplify rare label variants. Those remain declared empirical diagnostics and claim boundaries, not code-review findings.
