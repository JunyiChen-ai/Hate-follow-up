# Independent supplemental review: M1 branch diagnostics

Date: 2026-10-03. Reviewer: independent agent `/root/m1_grounder_code_review`, same model as the parent session. **PASS under rule 6; no observation-affecting correction required.**

Scope is only `scripts/analysis/m1_branch_diagnostics.py`, including the final `n == 1` bootstrap change. No method, reader, evaluator or primary metric was changed or re-reviewed. The reviewer read no real GT, VCD R2 performance or Integrator performance and ran no benchmark evaluation.

The helper calls the canonical `within_video_macro`; it does not implement a second AUC formula. Its shared extent is `min(GT length, prediction length)`, consistent with the canonical evaluator. Raw and decoded prediction lengths remain equal across arms, with 4 fps and matching window boundaries. Frame centres map to the original windows. Missing/nonfinite speech reads are omitted before the branch-specific canonical call; a video counts only if the remaining frames contain both classes. Branch availability must match the native arm, so paired branch comparisons use the same frame subset. Reported branch `n` is the number of eligible videos, not frames.

The final/raw means and branch means use the appropriate video-level units. Bootstrap samples paired video differences with 2,000 resamples and seed 0; a single eligible video retains its mean but has `ci95: null`. Single-window and multiple-window contributions divide each group's sum of final-AUC differences by the **full eligible-video count**, so the two contributions add to the overall paired mean difference. This distinguishes a subgroup's conditional mean from its contribution to the complete mean. The output correctly warns that a constant single-window raw read cannot establish added within-window timing merely because r6 changes its final curve.

Independent synthetic test and output:

- `runs/20261003_m1_branch_diagnostics/independent_review/check_diagnostics.py`
- `runs/20261003_m1_branch_diagnostics/independent_review/check_diagnostics.json`

The fixture includes both GT-longer and prediction-longer cases; a video becoming single-class after canonical truncation; entirely missing, single-class and noncontiguous speech availability; three paired arms; and a single-window case with unchanged constant raw scores but changed decoded ordering. Both synthetic corpora produced exact canonical `n` and mean agreement within `1e-12`. The single-window final gain of `.5` contributed `.125` to a four-video mean, and the two subgroup contributions recovered the complete paired mean. Availability and prediction-length mismatches were rejected. The fixture was rerun after the `n == 1` interval change.

This is a report-stage diagnostic for the complete, finite predictions already validated by the experiment analysis. Its branch-specific subsets and counts must remain visible when interpreting visual versus speech differences. The helper does not establish a semantic fusion mechanism, and its output does not replace the canonical primary metrics.
