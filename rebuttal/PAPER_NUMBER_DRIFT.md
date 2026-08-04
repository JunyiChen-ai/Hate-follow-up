# Paper number drift: HateMM headline (forensics + re-pin record)

Provenance record for the drafting phase. It documents a one-video drift in the
HateMM headline discovered while building the E1 rebuttal experiment, the
evidence, the re-pin decision, and the camera-ready checklist.

## Summary

The paper reports TRIAGE HateMM accuracy 86.5 (186/215). The current offline
files no longer reproduce that number: the live pipeline yields 86.0 (185/215).
The cause is a rerun of the 72B HateMM verifier verdict file on 2026-05-05, after
the frozen result package was written on 2026-04-27. The rerun changed one
in-band verdict, which flipped one video. The decision is to re-pin the HateMM
numbers to the reproducible current-file values; no restore attempt is made,
because the AWQ rerun is non-deterministic and would only re-roll the outcome.

## What drifted

Only the HateMM 72B verdict file drifted. The file
`results/boundary_rescue/HateMM/offline_test_qwen2.5-vl-72b-awq.jsonl` was
regenerated on 2026-05-05, whereas the other two default-panel verifiers
(`gemma-3-27b-it`, `qwen2.5-vl-32b-awq`) and every small-panel verifier
(`qwen3-vl-8b`, `internvl35-8b`, `gemma-3-12b-it`) keep their 2026-04-17 files.
Any frozen number computed with the default panel `gemma-3-27b-it >
qwen2.5-vl-32b-awq > qwen2.5-vl-72b-awq` on HateMM is therefore stale by one
video. Numbers that do not use the 72B verifier on HateMM (all small-panel
results, Stage-1, and every EN/ZH/IH cell) are unaffected.

## Evidence

1. The original build path reproduces the drift. Running
   `build_experiments.eval_sequential` on the current files gives HateMM 0.860465
   (185/215), not the frozen 0.865116 (186/215) in
   `results/paper_stage2_experiments/all_results.json`.

2. The E1 reimplementation matches the live pipeline exactly. The diagnostic
   `scripts/rebuttal_e1_diag.py` finds zero per-video prediction differences
   between `build_experiments` (frozen band file plus current verdicts) and the
   E1-natural path (re-fit band plus current verdicts); both give 185/215, with
   identical hbar/rho (0.315883 / 0.904143). So E1 is faithful and the divergence
   is not a code-path bug.

3. File mtimes place the rerun after the freeze.
   - `all_results.json` frozen 2026-04-27 05:23
   - HateMM `offline_test_gemma-3-27b-it.jsonl` 2026-04-17
   - HateMM `offline_test_qwen2.5-vl-32b-awq.jsonl` 2026-04-17
   - HateMM `offline_test_qwen2.5-vl-72b-awq.jsonl` 2026-05-05 21:00 (the rerun)

4. The original verdicts are not recoverable. The only backup,
   `offline_test_qwen2.5-vl-72b-awq.jsonl.before_rerun_20260505_201050`, is a
   degraded intermediate: 176 of its 215 rows carry `pred=-1`. It is not the
   2026-04-27 verdict set that produced the headline, so the single flipped video
   cannot be identified from preserved files.

## Re-pin decision and new canonical numbers

Adopt the reproducible current-file values as canonical. Default panel
`gemma-3-27b-it > qwen2.5-vl-32b-awq > qwen2.5-vl-72b-awq`, live from
`build_experiments`:

| Dataset | ACC | M-F1 | M-P | M-R |
|---|---|---|---|---|
| MHClip-EN | 79.50 | 73.22 | 77.32 | 71.49 |
| MHClip-ZH | 83.89 | 81.94 | 80.82 | 84.05 |
| HateMM | 86.05 | 85.28 | 85.83 | 84.88 |
| ImpliHateVid | 82.79 | 82.68 | 83.60 | 82.77 |
| Average | 83.06 | 80.78 | 81.89 | 80.80 |

Only the HateMM row changes from the submitted paper. The submitted values were
HateMM ACC 86.5, M-F1 85.8, M-P 86.2, M-R 85.5, and average ACC 83.18. The
reviewer-facing submission keeps 86.5 for any reference to "Table 1 as
submitted"; the re-pin applies to the camera-ready and to every new rebuttal
artifact.

Claims that survive the re-pin unchanged:
- The abstract range "improves accuracy over the best label-free/few-shot
  baseline by 2.5--8.7 points" is unaffected. The minimum gain is ImpliHateVid
  (82.79 - 80.3 = 2.5) and the maximum is MHClip-ZH (83.89 - 75.2 = 8.7); both
  are independent of HateMM. The HateMM gain moves 7.0 to 6.6, which stays inside
  the range.
- "Outperforms the best supervised baseline on three datasets" is unaffected.
  HateMM 86.05 still exceeds the best supervised HateMM baseline (MoRE 83.4), and
  the supervised comparators in E4 (HCC1 85.4) as well.

## Camera-ready checklist

- [ ] Table 1, TRIAGE HateMM cells: ACC 86.5 to 86.0, M-F1 85.8 to 85.3,
      M-P 86.2 to 85.8, M-R 85.5 to 84.9; recompute the average column
      (ACC 83.18 to 83.06).
- [ ] Table 2 and any ablation/Δ rows that use the default panel on HateMM:
      re-derive from current files (each shifts by at most one video).
- [ ] Regenerate the frozen result package from current files, since it predates
      the rerun: `results/paper_stage2_experiments/{all_results.json,
      order_sweep.csv, main_ablation.csv, rho_sensitivity.csv,
      stage1_backbone_transfer.csv, SUMMARY.md}` (re-run `build_experiments.py`).
      The full-experiments package under `results/paper_full_experiments/` that
      pulls from these should be refreshed too.
- [ ] E4 supervised-comparison text: update TRIAGE HateMM 86.5 to 86.0 (still
      beats HCC1 85.4).
- [ ] Abstract range 2.5--8.7 and "three datasets" claim: no change needed
      (verified above); leave as is.
- [ ] Zero-shot block, single-protocol regeneration (separate provenance issue,
      see the E3a finding): regenerate ALL Table-1 label-free zero-shot rows from
      one documented protocol (Fig. resolver_prompt = generic HATEMM_DEF), since
      the current block mixes definitions across cells. eval_one on the frozen
      `offline_test_ih_*.jsonl` reproduces the ImpliHateVid cells exactly
      (Gemma-3-27B 71.9/70.6, Qwen2.5-VL-32B 73.4/71.8) but the EN/ZH/HM cells no
      longer reproduce from any preserved file.
- [ ] Section 4.1 wording "mean performance over 10 random seeds": the headline
      pipeline is deterministic (GMM random_state=42), so this sentence does not
      match the main-table code path. Reword.

## Downstream consistency

The new rebuttal artifacts are already built on current files:
`scripts/rebuttal_e1_prevalence.py` hard-gates its natural row against the live
pipeline (not the stale headline) and warns on the drift;
`scripts/rebuttal_e2_extract.py` recomputes all rows live via `build_experiments`
(default-panel HateMM = 86.0); the E3 scripts consume current verdict files.
