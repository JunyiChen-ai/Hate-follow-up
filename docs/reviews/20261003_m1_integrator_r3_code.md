# Integrator R3 — independent code review

Date: 2026-10-03. Reviewer: `/root/m1_grounder_proposal_review`, independent of the implementation agent and using the same session model. Scope: the branch-specific R3 revision only, under repository rule 6. This did not reopen proposal/novelty review.

## Decision: PASS after targeted fixes

The final `experiments/20261003_m1_integrator/branch_revision.py` implements the declared cached composition: R2 future-text visual reads with native speech, native global margin and native selected answer; the matched arm uses explicit-causal visual reads with native speech. Base numerical reads and the base score curve remain unchanged. No outstanding material bug was found in this scoped review.

No existing performance files, GT arrays, or real prediction files were read. The reviewer read only source code and the R3 declaration, ran CPU tests with synthetic records, intercepted evaluator subprocesses, and wrote artifacts under `runs/20261003_m1_integrator/independent_review/r3/`. No GPU or real evaluation was run. The main agent made the implementation fixes; the reviewer did not edit method code.

## Findings fixed and confirmed

1. **R2 cache identity was initially checked only by directory name.** The revised `validate_source` now checks root configuration for `r2_main`, `future_keys=text`, model, frame/window/fps constants and the native global/answer semantics. Each arm configuration must equal the root configuration plus its arm. It also calls the existing `analyze.prepare`, which checks prediction/check/manifest key coverage, native-read parity against the original base cache, per-video prefix mode, token mapping, added-edge counts and upstream forward counts. Synthetic R1/all-mode, arm-configuration mismatch, per-video mode mismatch and edge-count mismatch were all rejected.
2. **Matching two arms' timestamps did not establish a valid common schedule.** `compose` now compares every window with shared `fixed_windows(duration, 8)` and requires `ceil(4*duration)` samples. Both arms sharing a wrong window boundary, or both sharing a truncated curve, are rejected. The original midpoint/index clipping rule is preserved, including a final 4-fps cell whose midpoint lies after a short truncated video endpoint.
3. **Manifest duration was not compared with cached duration.** A synthetic 333-record run initially accepted a manifest duration of 15 seconds against an 8.1-second cache for the same video ID. The main agent added the duration assertion before composition. The same negative case now fails as required.

These were confirmations within this one code review, not additional general review rounds.

## Executed checks

Harness: `runs/20261003_m1_integrator/independent_review/r3/review_synthetic.py`.

Results: `runs/20261003_m1_integrator/independent_review/r3/synthetic_results.json`.

Command:

```bash
/home/jehc223/miniconda3/envs/HateVideo/bin/python runs/20261003_m1_integrator/independent_review/r3/review_synthetic.py
```

The final execution exited 0; all recorded boolean checks are true. Coverage included:

- Changed visual readings combined with original speech; absent speech remains absent. A deliberately large modified-branch speech value was excluded from the composed result. Input records were not mutated.
- Native global margin/stance preserved, correct `3+B` base and `6+B` revised call counts, and removal of inherited timing fields.
- Exact base score curves for ten duration cases spanning very short videos, exact 8/16-second boundaries, epsilon-adjacent endpoints and truncated final cells.
- Rejection of mismatched video ID, duration, frame rate, global margin, stance, timestamps, speech presence and curve length.
- Complete synthetic coverage of 215 HateMM and 118 HateClipSeg records, with all three composed arms delivered to evaluation; rejection of a missing future-arm video.
- Real execution of `analyze.prepare` with synthetic prediction/manifest/check dictionaries and synthetic token arrays. `np.load` was intercepted and allowed only the synthetic token paths, never GT paths.
- Real execution of `analyze.evaluate` with `subprocess.run` intercepted: raw predictions target the sole canonical evaluator, `src.eval.evaluate_four_datasets`; the subsequent decoder invocation is the existing `twolevel_r2.py` with `--noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`. Both datasets and the intended raw/decoded directories are passed correctly.

Synthetic output files and zero-valued mock metric dictionaries are test fixtures, not experimental evidence. They are confined to the independent-review directory.

## Scoring and runtime boundaries

`compose` uses prediction records only. It replaces speech values before applying the existing raw max rule, and retains per-modality window readings for the unchanged r6 decoder. Configuration/manifest/token checks introduce no labels. GT is referenced only by the evaluation command, after all composed predictions are written; that command was captured rather than executed during review.

The test confirms the cached ablation and evaluator wiring, not a freshly scheduled GPU deployment. Revised `calls=6+B` is a prospective schedule declaration. The output explicitly says that standalone revised timing is unmeasured and removes old prefix/branch timings, so cached CPU composition is not presented as new-video inference runtime. The README correctly requires standalone scheduling verification and cost measurement before promotion.

This review does not certify the actual on-disk R2 caches by inspection: the repaired production entry point must execute its source-validation checks when the main agent runs R3. It also does not assert any gain, mechanism, promotion, or result from the synthetic tests.
