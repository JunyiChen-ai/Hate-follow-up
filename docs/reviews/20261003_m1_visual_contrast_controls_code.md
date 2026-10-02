# Visual contrast controls: supplemental code review

Date: 2026-10-03. Reviewer: `/root/m1_grounder_code_review`.
Scope: only the new `experiments/20261003_m1_visual_contrast/controls.py` and `launch/run_controls.sh`, plus the existing r6 interfaces needed to interpret these controls. This is a narrow supplement to the completed rule-6 reader review. No actual GT, primary/control performance file, GPU or benchmark evaluation was accessed or run; no candidate code was edited.

## Decision: PASS

No observation-changing computation bug or required implementation fix was found. The parent has already established the qualifying primary-result gate and authorized these predeclared controls. The four CPU diagnostic arms may run.

Construction matches the declarations:

- `scale`: visual `2a`, native speech.
- `video_shift`: visual `2a-mean(b)`, native speech. The mean is over that video's original windows, without duration weighting.
- `shuffle`: visual `2a-b[perm]`, native speech; a new `default_rng(0)` is created per video. Corrupted-read values and their video mean are preserved as a multiset. Changed-index and changed-value counts distinguish effective from ineffective permutations, including one-window videos and repeated values.
- `common_shift`: the difference between candidate and native **mean window maxima** is added to both available native branches. Missing speech stays missing. Consequently each raw maximum receives the same offset, preserving its ranking/ties and matching the candidate's window-mean maximum.

All arms deep-copy source records, preserve the original global margin, modality availability, video identity and boundaries, and expand onto the original-length 4 fps grid. No GT enters construction. Source extraction times are removed; `calls=None` plus explicit zero new-model-call metadata avoids presenting inherited extraction work as newly measured diagnostic cost. Preparation requires matched 333-video base/check coverage from the already validated primary run. Outputs are separate from primary records.

Evaluation delegates to the existing canonical wrapper and unchanged r6 command: `--noleak`, `nscore`, `calib`, BMA duration, length prior, minimum 2 windows, grid 6 and `m2`. Separate arm directories prevent fit/output mixing. The launcher waits for every successful evaluation before reporting. Decoded key coverage, 4 fps, finite scores, lengths and native globals are checked. Differences are consistently arm-minus-base and contrast-minus-arm; paired bootstrap operates on video differences.

The author added the initially missing report coverage during this review: visual/speech raw within-video AUC on available branch frames, largest gain/loss examples and native-verdict strata. These additions pass the synthetic report test. Speech diagnostics omit unavailable frames and require both frame classes in the remaining subset; their reported sample counts delimit that comparison. The Yes/No strata apply to mixed-frame videos entering within-video evaluation, not all videos' classification accuracy.

## Interpretation boundary

The existing r6 applies corpus-level ranks separately by modality. Multiplying all visual reads by the same positive constant therefore preserves its temporal inputs, while the raw visual/speech maximum and video key can change. The code's scale-arm decoded-within invariance check is consistent with this r6 path.

`common_shift` matches raw `K=z_video+mean(window maximum)` to the candidate, but a different offset per video can change cross-video modality ranks, independently fitted temporal emissions and decoded order. It preserves **raw** within-video order only. It is not a control that changes only the global term. The parent was asked to state this explicitly beside the README's common-shift description; this is a wording boundary, not an execution blocker.

Similarly, `video_shift` preserves each video's clean visual ordering but can change visual/speech dominance and corpus ranks. A control matching the candidate can refute a need for the particular matched-window subtraction; it does not by itself identify a semantic cause. Improved decoded within-video AUC without new raw ordering, especially with coarse/single-window evidence, is insufficient to claim newly localized temporal evidence.

## Independent synthetic evidence

Test/result: `runs/20261003_m1_visual_contrast/independent_review/check_controls.{py,json}`. The fixture covers all four formulas, reset seed, unchanged input/global, missing speech, repeated corrupted values, an ineffective single-window permutation and a truncated final window. Common-shift raw ranks and mean window maximum are exact. It separately demonstrates that per-video common shifts can alter corpus ranks.

The entire report executes with in-memory synthetic labels and stubbed metrics, verifying both difference directions, available-branch AUC, correct-Yes/wrong-No subsets and gain/loss fields. No real GT or metric file is opened. Canonical evaluator/r6 commands were captured through a stub and never executed. The launch script passes shell syntax checking. Synthetic fixture values are not experiment evidence or a mechanism result.
