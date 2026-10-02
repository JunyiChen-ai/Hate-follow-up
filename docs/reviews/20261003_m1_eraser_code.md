# Independent code review: M1 Eraser

Date: 2026-10-03. Reviewer: independent agent `/root/m1_grounder_code_review`.
Scope: rule 6 review of `experiments/20261003_m1_eraser/{erasure,measure,analyze}.py`, the launch scripts, and their shared input/MLLM/evaluation interfaces. No method implementation was edited, GPU was invoked, GT was read, or new performance result was inspected by this reviewer.

## Decision: PASS for GPU smoke

No remaining observation-changing implementation bug was found in the reviewed paths. The independent CPU fixture passes. Real multimodal re-encoding, native-baseline equality and native restoration still require the declared GPU smoke; the CPU fixture is not evidence of model-level numerical equality. This review does not establish performance, mechanism support or promotion.

One integrity omission was reported and corrected by the author before this decision: decoded predictions previously lacked explicit finite-value and 4 fps assertions. `report` now checks both decoded arms, every paired video, rate, finite scores, length and original global margin before reading GT. This matters because the shared evaluator can discard nonfinite positions. The correction was inspected directly.

## Observed implementation

- **Input deletion:** visual erasure removes exactly the sampled frames in each half-open window, with the final endpoint included. Speech erasure uses the same rounded proportional original-segment word indices as shared `window_text`. It preserves original segment timestamps, retains untouched segment strings exactly, omits emptied segments, and constructs each intervention from the original segments. Repeated words are removed by index rather than string matching. At runtime the flattened removed words must equal `window_text(...).split()`.
- **Fresh state:** every changed input goes through `prefix_messages`, `encode_prefix` and `prefix_cache` anew, after clearing `rope_deltas`. The altered branch asks the same global question; it contains no original answer turn, old local query or old KV cache. All native local branches finish before any altered prefix is built, and each native branch crops its own query suffix. The smoke restoration read rebuilds the full original input and requires exact original margin and prefix length.
- **Scores:** the estimator is original Yes-minus-No margin minus the freshly read altered-input margin. Removing visual content retains all original speech and vice versa. A visual window with no sampled frame receives exactly zero without an unnecessary model call; absent speech remains absent. Maximum fusion therefore floors a negative speech effect at zero in a frameless window. This behavior is declared and separately counted by analysis; it must not be described as observed visual evidence.
- **Label-free path and coverage:** the reader imports no GT, metrics or label-based selector. The fixed manifest, repaired ASR, sampled frame input, 8-second windows and 4 fps expansion retain the native input/grid conventions. Runtime failures abort rather than silently remove a video. `prepare` requires the complete declared manifest, both raw arms and integrity records to have identical keys, equal original global scores, equal windows/speech presence, finite curves and exact native window reads versus `runs/20260926_glr/base_gridA/predictions.jsonl`.
- **Evaluation:** both arms call the sole shared evaluator and the unchanged r6 command (`nscore`, `calib`, duration `bma`, length prior, minimum 2 windows, grid 6, `m2`, `--noleak`). Each arm has a separate fit/output directory. The CPU launcher waits for both successful exits before reporting. Within-video comparisons use shared `within_video_macro`; paired bootstrap samples video differences, with 2,000 resamples and fixed seed 0. GT appears only in evaluation/reporting, not in inference. Continuation and performance gates match the README; `mechanism_supported` remains false until the declared controls are completed.
- **Cost:** native deployment counts `3 + B` model forwards (prefix, global question, answer, local branches). Eraser counts `2 + 2E` (original prefix/global plus a fresh prefix/global pair per nonempty intervention). Combined experiment work counts `3 + B + 2E`, plus 2 forwards for smoke restoration. Eraser standalone time includes the original prefix/global read and every changed-input read; the field named `prefix_seconds` explicitly includes the original global question. Diagnostic restoration time is recorded separately. Full multimodal prefix processing is charged on every intervention; encoder work is not represented as free cached preprocessing.

## Independent CPU evidence

Command:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /home/jehc223/miniconda3/envs/HateVideo/bin/python \
  runs/20261003_m1_eraser/independent_review/check_erasure.py
```

The test and readable result are under `runs/20261003_m1_eraser/independent_review/check_erasure.{py,json}`. All assertions passed:

1. Boundary/overlap cases, repeated words and unusual whitespace preserve original word-index removal, counts and untouched text. Original frame and speech inputs remain unchanged.
2. Exact boundary frames are assigned once, including the final endpoint.
3. An instrumented synthetic Judge enforces a fresh prefix and cleared RoPE state at every intervention, native local reads before intervention, branch-cache isolation and final original-input restoration.
4. A three-window fixture gives visual effects `[2, 2, 0]` and speech effects `[-3, -3, -3]`, explicitly exercising the zero-frame maximum floor and the corresponding 96-point 4 fps curve.
5. The same fixture verifies 7 fresh prefixes including restoration, 6 native local branches, native cost 9 forwards, Eraser standalone cost 12, and combined smoke cost 21.

These are synthetic score values, not benchmark performance. The fixture verifies deletion and control flow only; native model numerics and actual GPU cost remain the smoke check's responsibility. No review test derives seeds or provenance from hashes.

## Claim boundary

The code implements actual input occlusion rather than a hypothetical exclusion instruction. Removing input also changes token positions, formatting and the surviving transcript context. Original coarse timestamps remain attached to retained words. A response difference alone does not prove a cross-modal interaction or identify causal ground-truth evidence. Any eventual gain remains development-selected and needs the predeclared quantity, time-alignment, original-query and score-shift controls, including the missing-frame floor diagnostic when relevant.
