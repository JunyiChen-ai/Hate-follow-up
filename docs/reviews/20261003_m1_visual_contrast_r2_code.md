# Visual contrast R2: supplemental code review

Date: 2026-10-03. Reviewer: `/root/m1_grounder_code_review`.
Scope: only the new window-scoped pixel path in `visual_contrast.py` / `measure.py`, its analysis schema/cost changes and round-aware launch paths. This supplements the existing rule-6 review, without reopening proposal/style review. No candidate source was edited, real GT/prediction/performance file was read, or GPU/benchmark evaluation was run.

## Decision: PASS for the declared five-video GPU smoke

No observation-changing bug requiring a fix was found. The new path passes independent pixel-block, actual small-model and synthetic analysis checks. Deployment-version numerical parity and full-input memory/time remain GPU-smoke requirements.

`mix_window_pixels` uses each `image_grid_thw` product `t*h*w` to locate complete **unmerged processor-patch** blocks. It correctly does not divide by the spatial merge factor used for language image-token counts. Image-grid length is checked against the original frame list. Processor/file order stays original. Selected frame indices must be unique/in range, and their pixel counts must cover the complete pixel tensor when all images are considered.

The complete R1 noisy tensor is generated once per video. Every window intervention starts with a new clean-pixel clone and copies only that window's block ranges from the same noisy tensor. It does not draw a new noise field, reuse the noise of a differently indexed frame or accumulate previous-window corruptions. Original pixels, other input tensors, image grids, token positions, transcript and prompts remain unchanged. The still-image temporal copies retain the same already-reviewed shared noise.

Window membership is half-open, with the final endpoint included; each sampled frame must belong exactly once. Each nonempty window gets a fresh mixed-input prefix after a RoPE reset, its own global question and the forced original answer, followed by only that window's visual query. The cache is discarded between interventions. Corrupted global margins are diagnostics, never substituted for the original downstream global. Native speech scores and availability remain exact.

A frame-free window retains its native visual margin as the corrupted read (`b=a`), so `2a-b=a`, without any extra model call. It is neither dropped nor assigned an invented image/zero score. The existing native-restoration smoke still checks the whole original global/local computation after every intervention has completed.

Forward counts are consistent: native `3+B`; R2 deployed/shared paired work `3+B+4K`, where K is the number of windows containing sampled frames; smoke adds `3+B`. Each increment of four comprises prefix, global question, forced answer and one visual query. Timing includes all mixed-prefix construction and inference. The new diagnostic-global mean is explicitly averaged over corrupted-prefix evaluations, rather than per video or as a downstream key.

Analysis retains complete manifest/baseline checks, original global/speech, finite 4 fps curves and canonical evaluation/r6 calls. R2 preparation additionally checks K, the intervention/global-list lengths, exact nonempty-window indices, frame-free identity and `3+B+4K`. CLI `--run-name r2_main` routes source, decoded and analysis output into separate R2 directories. The launcher arguments for this path are `run_lab.sh smoke window r2` / `run_lab.sh main window r2`, and `run_analysis.sh r2`; the R1 default remains full-scope. Both launch scripts pass shell syntax checks.

## Independent evidence

Test/result: `runs/20261003_m1_visual_contrast/independent_review/check_r2.{py,json}`. CPU with CUDA disabled, torch 2.7.1+cu128 / transformers 4.57.6. These versions differ from deployment, so the test is not a substitute for CUDA smoke.

- Actual processor output for three differently sized images produces raw patch blocks of 16, 24 and 8 rows. Empty, full, single-frame and noncontiguous selections reproduce exactly the expected noisy/clean blocks and offsets.
- An actual small multimodal Qwen3-VL, with vision layers and DeepStack, passes in FP32 and BF16. Frame times 0, 8 and 32 seconds exercise the start, internal boundary and final endpoint; a four-window video leaves one window frame-free. Every mixed prefix contains its selected R1-noise block and clean other blocks, and only the appropriate target query is called. Original tensors, speech/global/answer, model weights and native restoration are exact. With B=5 and K=3, deployed work is 20 calls and smoke work is 28. Output has 128 positions at 4 fps.
- Synthetic-only `prepare` and `report` complete with the new schema and correctly average the diagnostic-global list. A stubbed analysis entry verifies separate `r2_main` / `r2_main_decoded` paths without executing the evaluator. Synthetic labels/metric fixtures are not benchmark evidence.

I separately read only the real frame filenames/timestamps and manifest duration/identity metadata to check the new coverage assertion: all 333 videos assign every sampled frame exactly once. Counts are K=2,738 for 215 HateMM videos and K=2,358 for 118 HateClipSeg videos; frame-free windows number 1,030 and 1,233 respectively. These match the declared intervention counts. Evidence: `runs/20261003_m1_visual_contrast/independent_review/r2_frame_coverage.json`. No labels or predictions enter that check.

The verified change localizes the input corruption, while the freshly encoded query still uses full remaining context. It does not by itself establish causal visual hate evidence or finer temporal information for a single-read window; those remain the declared empirical claim boundaries.
