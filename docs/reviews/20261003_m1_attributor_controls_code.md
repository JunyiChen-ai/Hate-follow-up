# Attributor controls: supplemental independent code review

Date: 2026-10-03. Reviewer: `/root/m1_grounder_code_review`.
Scope: `experiments/20261002_m1_attributor/controls.py` and `launch/run_controls.sh`, checked against the existing README declarations. This supplements the original rule-6 review; it does not reopen review of the reader or ongoing primary experiment.

## Decision: PASS

No observation-changing bug requiring a fix was found. No method code was modified, GPU or evaluator was launched, GT was read, or full/partial primary performance was inspected. This decision permits the declared controls only after the complete primary result passes its existing gain gate; it is not evidence of gain or mechanism support.

The five transforms implement their declarations:

- `endpoint` uses the saved gradient at gate 1, without path integration. The original gate displacement is one, so no additional input multiplier is missing.
- `absolute` takes the absolute value of token contributions before window aggregation. This removes token signs, rather than merely taking the absolute value of an already aggregated window.
- `rotated` shifts contributions by half the number of tokens separately inside the visual and speech token sets. It retains the original token masks/membership, each modality's multiset of contributions and speech availability. It is a token-position rotation, not a guaranteed half-duration rotation.
- `density` sets media-token contributions to one and uses the original shared-membership splitting and number-of-windows scaling.
- `shift` adds one common per-video offset to every available native modality read. The offset is the difference of attribution and native mean window maxima, so the native raw ordering and ties remain unchanged while its mean window maximum matches attribution. Refitting r6 can still change decoded ordering; the code claims raw-order preservation and checks it there.

All controls retain the original global margin, boundaries, video identity, 4 fps grid and speech availability. The transformations copy source records/arrays and do not alter the saved primary predictions or token contributions. Copied model-call/time fields are explicitly identified as source-extraction cost; incremental model calls are reported as zero, without claiming an independently measured cheaper deployment path.

Generation requires matched complete base/attribute records (333 videos) from `r1_main_fp32_mem`. All CLI stages first require `r1_main_fp32_mem_analysis/summary.json` to have `any_qualifying_gain`; a missing summary or false gate blocks execution. The declaration's gate is any final primary metric gain of at least .01, as computed by the existing primary analysis. No GT or metric value enters a control's score transform. The sequential shell launcher stops on any unsuccessful stage and keeps control outputs separate from the primary run.

Evaluation invokes the canonical evaluator and the same r6 arguments as the primary pair: `--noleak`, `nscore`, `calib`, BMA duration, length prior, minimum 2 windows, grid 6 and `m2`. Each arm refits into its own output directory. Before GT reporting, raw/decoded arms must have matching complete keys, finite 4 fps curves, original lengths/global margins and unchanged raw speech availability. Report differences use the correct attribute-minus-control and control-minus-base directions. Within-video bootstrap resamples paired video differences through the already reviewed shared analysis helper. The report leaves `mechanism_supported` false and explicitly lists the remaining FP32 native-window and cached-value intervention checks.

## Independent evidence

Test: `runs/20261002_m1_attributor/independent_review/check_controls.py`; result: the adjacent `check_controls.json`.

Run on CPU with CUDA disabled. An independent per-token accumulation reference checked all five transforms with shared memberships, unsupported media tokens, both contribution signs and absent speech. Four saved numerical-smoke videos from `r1_smoke_refined` (12, 5, 29 and 33 windows) reproduced the original attribution mapping; endpoint and density also matched their saved diagnostic window values. Every arm preserved source inputs, boundaries, 4 fps expansion, original global margin and speech availability. Shift rank/tie preservation and mean matching passed.

An isolated synthetic-summary fixture checked missing/false/true primary-gate behavior. Evaluator and decoder commands were captured with a stub and never executed. The author's `controls_selfcheck/checks.json` was also read. Neither the running primary experiment nor Eraser was touched; no benchmark GT or performance was used in this review.
