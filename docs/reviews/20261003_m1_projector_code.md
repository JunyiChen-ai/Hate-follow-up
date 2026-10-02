# M1 Projector — independent code review

Date: 2026-10-03. Reviewer: `/root/m1_grounder_code_review`, independent of the implementer. Scope: rule 6, observation-affecting correctness of `experiments/20261003_m1_projector/{README.md,projector.py,measure.py,analyze.py,launch/}` and the shared model/evaluation interfaces. Proposal review was already PASS. No production source was changed, GPU execution or benchmark evaluation started, real GT/performance read, or hash used.

## Decision: PASS

No required correction was found. The implemented attention intervention changes the final visual readouts through the declared per-head computation, while native global/answer/speech remain unchanged. The matched eager comparison is present in both collection and the primary acceptance gate. This is a code-integrity decision; 8B GPU parity, memory, cost and effectiveness remain for the declared smoke/full run.

Independent evidence:

- `runs/20261003_m1_projector/independent_review/check_projector.py`
- `runs/20261003_m1_projector/independent_review/check_projector.json`
- `runs/20261003_m1_projector/independent_review/check_projector.out`

Command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /home/jehc223/miniconda3/envs/HateVideo/bin/python runs/20261003_m1_projector/independent_review/check_projector.py`.

## Actual model and attention checks

The independent CPU fixture uses torch 2.7.1, transformers 4.57.6, an actual Qwen3VLModel with 36 text layers, hidden size 64, four query heads/two KV heads, mRoPE and a three-layer vision encoder with DeepStack outputs [0, 1]. It processes 20 synthetic image tensors of two patch-grid shapes, yielding 100 image-token positions in 20 noncontiguous blocks. Both FP32 and BF16 were executed. Shared prefix-message construction and output-head projection are retained, with synthetic question tokenization/input records. The production `read_video` function runs directly, without a source adaptation. This is not an 8B or real-media experiment.

- An independent oracle selects each original KV head explicitly (`head // groups`), calculates last-row ordinary attention and a separate non-image-key subset reference, and applies the declared vector update. Both boolean and additive masks, grouped-query attention with three query heads per KV head, noncontiguous image support, no-image identity, zero U and the additive `1e-8` denominator pass. Actual rotated/normalized Q/K/V from all 36 layers are additionally checked against this oracle with the real fixture's two-head GQA groups.
- The implementation computes `unit = U / (norm(U) + 1e-8)`, removes `(delta·unit)*unit` from `delta = O-U`, and adds `1.4*correction` to O in FP32 before casting back. Gamma zero returns the ordinary eager last-row output. The projection is per head before `o_proj`, not after head concatenation or at final class logits.
- All 100 actual expanded image-token positions are passed into the reference mask in every affected layer. Prefix scaffolding, global QA and query suffix positions are not marked as images. The mask is padded with false entries to the complete current KV length, including the query tokens. The source receives Q/K after native normalization and mRoPE and preserves the original causal mask.
- Native SDPA computes all query rows first; only the final row is replaced. Hooks on every decoder output verify that all earlier query rows remain exactly equal across native/eager/project, while the last row changes in all 36 layers. Changing the final query token leaves prior rows exact, confirming causality in the actual model path. Vision/DeepStack prefilling executes normally while intervention is inactive.
- Every branch leaves the original prefix/global-QA K/V unchanged after crop. Reading a cache copy versus the reused cache yields exact logits. mRoPE positions and offsets remain exact. Fresh native reconstruction, native global/answer/speech, missing-speech availability, input tensors, model/head weights and the attention registry all restore exactly in both dtypes.

The toy BF16 eager/native Yes-minus-No difference is `.0057621598` (FP32 `-5.96e-8`), while project/eager answer-token differences are nonzero. This demonstrates both active intervention and a real numerical reason to retain the matched eager arm. These random-model values are not performance or mechanism evidence, and native/eager bitwise parity is not claimed.

## Reader, evaluation and accounting

With three 8-second windows and one available speech question, `V=3, B=4`. The actual outer-model hook counts 20 calls including smoke, exactly `3+B+2V + 3+B`. Paired collection is 13 calls; each independently deployed arm is 7 (`3+B`). The 96-value 4-fps curves exactly match their window maxima. All three arms retain the same global score, answer and available speech value. Eager/project margins reconstruct from their saved answer-token logits.

A complete synthetic 333-video manifest (215/118) exercises prepare/report without reading corpus files. Config/constants, full arm coverage, native historical parity, saved-token reconstruction, window bounds/durations, speech alignment and geometry shape checks pass. Wrong epsilon, a missing project video, incorrect duration/window start/speech, decoded NaN or altered global are rejected. Config/coverage and decoded finite/global checks reject before GT loading; token alignment uses synthetic saved arrays only. Report bootstrap samples paired video differences. The synthetic eligible counts of 215/118 test the reporting unit, not the real datasets' within-video eligibility.

Synthetic metrics deliberately separate the arms. A `.02` improvement against native and `.015` against eager passes the numeric gate. Holding native gain at `.02` but reducing the improvement against eager to `.005` correctly prevents `performance_pass`; `mechanism_supported` remains false. Raw and decoded field names use project/eager consistently, without a stale copied candidate key.

Intercepted evaluation commands correctly call `src.eval.evaluate_four_datasets` and unchanged `experiments/20260926_twolevel/twolevel_r2.py` with `--noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`. Inputs, metrics, decoded arms and reports point only to the Projector run directories. Per-video diagnostics call canonical `within_video_macro`; evaluation formulas are not duplicated. Both launch scripts pass shell syntax checks. Analysis first validates prepare, launches three separate output arms, checks all exit codes and only then reports.

Outer-call equality does not mean equal computation. The wrapper retains a native SDPA reduction and adds both ordinary and image-masked last-row reductions, per-head projection and diagnostics in every affected layer. The measured standalone and paired seconds include these operations; smoke restoration is separately accounted. Actual 8B GPU time and peak memory, not the source paper's efficiency or these CPU fixture numbers, must substantiate the cost claim.
