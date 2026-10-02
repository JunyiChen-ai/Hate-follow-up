# Independent code review: M1 visual contrast

Date: 2026-10-03. Reviewer: `/root/m1_grounder_code_review`.
Scope: the single rule-6 review of `experiments/20261003_m1_visual_contrast/{visual_contrast,measure,analyze,selfcheck}.py` and `launch/`, against the README/proposal. No candidate code was edited, GPU or benchmark evaluation was launched, or actual GT/candidate performance was read.

## Decision: PASS for the declared GPU smoke

No observation-changing bug requiring a fix was found. Independent CPU checks, including the actual processor, small multimodal model and synthetic report, pass. Exact native parity and measured memory/time on the deployment environment still require the declared five-video GPU smoke. This decision establishes neither performance nor mechanism support.

## Noise and input layout

The implementation uses the declared 1,000-step sigmoid beta schedule and zero-based step 500, including factors indexed 0 through 500. The resulting FP32 cumulative alpha is `0.7447987198829651`; an independent NumPy float64 product differs by approximately `1.94e-7`. Step zero is correctly treated as nonidentity; the identity test explicitly sets `abar=1`.

I read the installed lab2 transformer 5.15.1 processor source at `HateVLM/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py`. It normalizes before patchification, then inserts/expands the temporal axis after the channel dimension before flattening: C,T,P,P. The candidate's inverse reshape matches it. An actual local Qwen image-processor invocation also matches an independent, explicit per-frame/per-spatial-patch normalized-pixel reference exactly.

Noise uses a fresh CPU generator with seed 0 for each video's corruption. Each RGB spatial pixel/patch and frame gets its own sample; only the repeated temporal copies of a still image share the sample. Original temporal equality is asserted. The routine preserves global RNG state, leaves original pixels untouched and neither clips nor renormalizes the result. Repeated calls reproduce the same corrupted tensor; `abar=1` preserves the FP32 input exactly. Inconsistent temporal copies are rejected.

## Reader and scores

Native prefix/global/answer/local reads finish before corruption. The noisy input replaces only `pixel_values`; tokens, image grids, timestamps, ASR and prompts remain unchanged. Every new prefix resets RoPE state and receives a fresh cache. The corrupted prefix recomputes its own global-question/answer states but forces the original native answer; its global margin is diagnostic only. Local queries crop back to that prefix's question/answer length. The final native restoration re-encodes the original tensors and requires exact original global and all local reads.

Only the visual class margin changes, to `2*clean_visual-corrupted_visual`. Original speech margins and their availability are copied exactly; the corrupted path does not recompute speech queries. Both final arms retain the original global margin and answer. This is binary **class** log-odds contrast after native label aggregation, not token-variant contrast. Independent class-probability algebra and coefficient-zero/identical-read identities pass.

Windows without a sampled frame still receive the declared full-context visual query; no artificial local image or zero score is introduced. Counts are logged and reported. Raw maximum, truncated final window and 4 fps expansion are unchanged. No labels enter corruption, query selection or scores. Unchanged transcript tokens do not imply unchanged transcript hidden states in the noisy prefix; the method should retain that declared interpretation boundary.

## Evaluation, schema and cost

`prepare` requires complete paired manifest/check coverage, finite 4 fps curves, aligned windows/speech availability, exact native global/window reads versus `base_gridA`, exact preserved speech and reconstructible `2a-b` visual scores. Diagnostic branch counts and full-run forward counts are checked. Reporting validates decoded coverage, rate, length, finiteness and original global before GT access. Raw/final differences use `contrast-base`; the prior candidate's template-field bug is absent. Bootstrap samples paired video differences, with 2,000 resamples and seed 0. Gates match the declaration and mechanism support stays false pending controls.

Raw evaluation calls the sole canonical evaluator. The r6 command retains `--noleak`, `nscore`, `calib`, BMA duration, length prior, minimum 2 windows, grid 6 and `m2`, separately fit into each arm's directory. The CPU launcher waits for both successful evaluations before reporting; launch scripts pass shell syntax checks. No metric formula was duplicated.

Costs match actual work: native `3+B`; deployed contrast and the shared paired measurement `6+B+V`; smoke restoration adds `3+B`. Native prefix/global time is included once, native answer/local reading is included in base time, and contrast additionally charges tensor corruption/checking plus the fresh noisy prefix, global question, forced answer and V visual queries. There are no backward passes or per-window prefix rebuilds. Caches are used sequentially. Peak memory is the combined run peak, not a separately measured native-only peak.

## Independent evidence

Test/result: `runs/20261003_m1_visual_contrast/independent_review/check_visual_contrast.{py,json}`. Local CPU only, CUDA disabled; torch 2.7.1+cu128 / transformers 4.57.6. Deployment source inspection used version 5.15.1; GPU smoke remains necessary for that environment.

The test exercises the actual image processor and an actual small Qwen3-VL with vision layers, DeepStack and cached language queries, in both FP32 and BF16. `read_video` preserves every original tensor, text/grid, native speech/global/answer and model weight; corrupted pixels change; fresh native restoration is exact. A three-window fixture includes an empty-speech branch and a window without a sampled frame. With B=4 and V=3, deployment is exactly 13 forwards and smoke is exactly 20; output has 96 positions at 4 fps.

The complete report was executed using explicitly synthetic labels and stubbed metrics, without opening any benchmark GT or metric file. It produces the expected paired raw differences and missing-frame counts. Canonical evaluator/r6 commands were captured through a stub and never executed. The author's `selfcheck/noise.json` was also inspected. Fixture values are not benchmark evidence.
