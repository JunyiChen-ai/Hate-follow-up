# M1 Highlighter — independent code review

Date: 2026-10-03. Reviewer: `/root/m1_grounder_code_review`, independent of the implementer. Scope: rule 6, observation-affecting correctness of `experiments/20261003_m1_highlighter/{README.md,highlighter.py,measure.py,analyze.py,selfcheck.py,launch/}` and its shared model/evaluation interfaces. Proposal review was already PASS. No production source was edited, GPU or benchmark evaluation started, real GT/performance read, or hash used.

## Decision: PASS

No required implementation correction was found. Full-vocabulary salience, timestamp-local support, cached-value guidance and head balancing enter the final visual scores through the declared layers. Native global/answer/speech and empty-support windows remain unchanged. This is an implementation-integrity decision, not evidence of effectiveness or semantic localization. The declared 8B GPU smoke must still establish runtime parity, memory and cost.

Independent executable evidence:

- `runs/20261003_m1_highlighter/independent_review/check_highlighter.py`
- `runs/20261003_m1_highlighter/independent_review/check_highlighter.json`
- `runs/20261003_m1_highlighter/independent_review/check_highlighter.out`

Command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /home/jehc223/miniconda3/envs/HateVideo/bin/python runs/20261003_m1_highlighter/independent_review/check_highlighter.py`.

## Actual model, salience and local support

The CPU fixture uses torch 2.7.1 and transformers 4.57.6, with an actual Qwen3VLModel containing 36 text layers, hidden size 256, 32 query heads/eight KV heads, mRoPE, and a three-layer vision encoder with DeepStack outputs [0, 1]. Twenty processed synthetic images produce 100 image-token positions in 20 noncontiguous blocks. Both FP32 and BF16 execute the production `read_video` directly. Shared prefix messages and output projection are retained; question tokenization and media/ASR records are synthetic. There is no source-width or head-schema adaptation and no claim to have executed the 8B model.

- An independent observer of the actual prefix finalNorm output exactly matches the captured visual hidden rows. The hook is removed after prefix construction; the capture neither re-normalizes hidden states nor includes global/question states. Prefix vision and DeepStack computations run natively.
- Salience is checked against a full-softmax probability oracle, using 257 rows to cross multiple 128-row batches. A uniform 100-word vocabulary yields `.2`, whereas renormalizing the top ten would yield `1`; the implementation returns the former. A single probability-one token with underflowed alternatives gives finite zero salience, following `0 log 0 = 0`. Actual model visual states also match a separately computed full-vocabulary probability oracle. Projection and the entropy arithmetic use FP32, with no top-ten re-normalization.
- Mapping follows actual expanded image positions, processor image counts and timestamped frame order. Separate checks exercise an exact 8-second boundary, the last window's inclusive `t=24` endpoint, all-zero local weights becoming uniform, an empty support and rejection of a count assignment that crosses noncontiguous blocks.
- The actual 20-image reader fixture has local support sizes `[50, 0, 50]` across three windows. Thus the middle query is genuinely unsupported, while the last window includes its final frame at duration. Guidance has zero weight on non-image positions and on other windows' frames; it is normalized within the local support.

## Attention and unchanged branches

An independent per-head oracle selects each original KV head directly, weights its cached local values, and compares the resulting D against the actual dispatch input. It then uses the native SDPA output and independently computes cosine similarity, cross-head normalization, ReLU balancing and the additive update. All 32 heads are exercised. The FP32 reduction agrees within roundoff; BF16 outputs agree within the declared cast precision. A separate sparse guidance vector checks that the active fraction is computed on local G, including a value below `1e-8`, and gives exactly the expected one-third fraction within FP32 roundoff. Zero coefficient, zero D and all-zero cross-head similarity are also exercised; the last case uses the declared gamma-one fallback.

The active layer trace is exactly 4–17 inclusive. Earlier query rows remain bitwise native at all 36 block outputs; layers 0–3 remain entirely unchanged. Changing the final query token has no effect on previous rows, confirming causality in this model path. The guidance uses only cached prefix V; native SDPA retains access to the original context and its original attention mask. Later layers may propagate the changed last-row representation, as intended.

Zero coefficient and empty support produce bitwise native logits and hidden output in both dtypes. Every branch restores the prefix/global-QA K/V after crop, with exact equality between cache-copy and cache-reuse reads. mRoPE positions/offsets, fresh native reconstruction, native global score, forced answer, available speech and missing-speech flags remain exact. Input tensors, model/head parameters and the attention registry also restore exactly. Nonzero guided answer-token changes demonstrate active computation only; they are random-model plumbing evidence, not performance gains.

## Alignment, evaluation and cost

The complete reader fixture has `V=3` visual queries and one available speech query, so `B=4`. Actual outer-model hooks count 17 calls including smoke: `3+B+V + 3+B`. Paired collection is 10 calls, and standalone deployment of either arm is 7 (`3+B`). The no-frame window still has a native query forward, consistent with that count. The 96 values on the 4-fps grid exactly equal their window maxima. Salience, local guidance, 14-layer geometry and all 32 head coefficients are written with the expected schema.

A separate complete 333-video synthetic manifest (215/118) exercises prepare and report using those actual 32-head arrays. It verifies configs, coverage, native baseline parity, token-logit reconstruction, image/frame mapping, timestamp-local guidance and empty-window identity. Wrong top-k, a missing arm video, incorrect duration/window start/speech, an out-of-window saved guidance distribution, decoded NaN and altered global are rejected. Config/coverage and decoded finite/global checks reject before GT loading; token alignment checks use only saved synthetic arrays. No real corpus GT or prediction file is opened. Paired bootstrap samples videos, and all per-video within values use canonical `within_video_macro`; the synthetic eligible counts 215/118 are not claims about real dataset eligibility.

Synthetic metric fixtures exercise both passing and below-threshold outcomes. Report fields consistently use `highlight`/`highlight_raw`; no copied candidate field remains. `mechanism_supported` stays false even when the synthetic performance gate passes. Intercepted subprocess commands correctly invoke `src.eval.evaluate_four_datasets` and unchanged `experiments/20260926_twolevel/twolevel_r2.py` with `--noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`. Outputs are isolated under Highlighter's run directories. Both launch scripts pass shell syntax checks; prepare precedes evaluation, all child exit codes are checked, and report follows only successful evaluations.

Cost accounting charges salience projection to highlight and includes guided reductions and diagnostics in its branch time. The baseline prefix time includes read-only hidden capture, as declared. The full-vocabulary FP32 output matrix is retained by the engine; 128-row batching limits temporary logits, not that persistent matrix. Same outer-forward counts therefore do not imply equal time or memory. Actual 8B timings and peak allocation from the pending smoke are required for the cost statement. Nothing in this review establishes that the contextualized local values contain only local semantics or that truncated entropy measures hateful-content certainty.
