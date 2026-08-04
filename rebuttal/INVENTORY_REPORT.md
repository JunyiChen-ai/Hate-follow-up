# Inventory report — frozen artifacts vs rebuttal experiments

Produced by inventory subagent, 2026-07-09. All paths under /data/jehc223/EMNLP2.

## Headline findings

1. **All 6 verifiers have FULL-TEST verify-all verdicts on all 4 datasets** at `results/boundary_rescue/<ds>/offline_test_<verifier>.jsonl` (ImpliHateVid: `offline_test_ih_<verifier>.jsonl`). Coverage gaps are tiny parse-failure holes, never band-only subsets:
   - gemma-3-27b-it: EN 161/161, ZH 148/149, HM 215/215, IH 398/401
   - qwen2.5-vl-32b-awq: EN full, ZH full, HM full, IH 383/401
   - qwen2.5-vl-72b-awq: EN full, ZH full, HM 214/215, IH 392/401
   - qwen3-vl-8b: EN 159/161, ZH full, HM full, IH 398/401
   - internvl35-8b: EN 159/161, ZH full, HM full, IH 392/401
   - gemma-3-12b-it: EN full, ZH 148/149, HM full, IH 398/401
   → Pipeline recomputation under resampled pools is fully offline/CPU.

2. **Stage-1 scores**: `results/holistic_<slug>/<ds>/{train,test}_binary.jsonl`. 2b primary: EN 550/161, ZH 579/157(prerepro override), HM —/215 (**no train scores; HateMM pool = test split**), IH 1283/401.

3. **Pipeline entry point**: `results/paper_stage2_experiments/build_experiments.py::eval_sequential(cache, MAIN_ORDER, rho_mode="entropy")` reproduces the paper headline exactly (EN 0.79503 / ZH 0.83893 / HM 0.86512 / IH 0.82793, avg 0.83175, calls 0.686). Calibration pools: EN TR-Otsu(train 550), ZH TR-GMM(train 540), IH TR-GMM(train 1283), HM TF-li_lee(test 215). Bayesian update `ell += (2·pred−1)·log(ρ/(1−ρ))`, early stop at `ent(sigmoid(ell)) ≤ hbar`; ρ_D label-free (EN .825, ZH .845, HM .904, IH .937). In-band counts: EN 91, ZH 88, HM 100, IH 153.

4. **⚠ Determinism discrepancy**: headline pipeline is fully deterministic (GMM random_state=42); the paper's "mean performance over 10 random seeds" sentence in §4.1 does NOT correspond to the main-table code path. Do not repeat that claim in the rebuttal; fix in revision (REVISION_PLAN item).

5. **Small-verifier panels already computed** (order_sweep.csv, label-free ρ_D). Best all-small panel `internvl35-8b > qwen3-vl-8b > gemma-3-12b-it`: **avg ACC 81.36 / MF1 0.7784, 0.683 calls/video**; per-dataset EN 0.77019/0.67228, ZH 0.81208/0.78244, HM 0.84651/0.83446, IH 0.82544/0.82452. All 6 small orders within 0.3 pp of each other. Max model = 12B.

6. **72B full-test coverage for a zero-shot row**: EN 161/161 ✓, ZH 149/149 ✓, HM 214/215 (−1), IH 392/401 (−9). Ten videos total missing across HM+IH.

7. **Prior pool studies**: `scripts/run_reviewer_followup_checks.py` sweeps pool SIZE {0.1..1.0}×30 seeds (`results/boundary_rescue/reviewer_followups/unlabeled_pool_sensitivity_summary.csv`; prop=1.0 row = headline). `scripts/plot_deployment_collaboration.py` = class-balanced subsample (hand-picked seeds — do not reuse for claims). **Nothing sweeps pool class PREVALENCE** → E1 is a genuine gap, CPU-only.

## Key code paths
- Calibration: `src/boundary_rescue/baseline_preds_v2.py` (PROTOCOL dict), thresholds `src/our_method/quick_eval_all.py` (otsu/gmm, random_state=42), `src/boundary_rescue/thresholds.py` (li_lee)
- Pipeline: `results/paper_stage2_experiments/build_experiments.py`, `src/boundary_rescue/grid_eval_all.py` (judge_path)
- ZH SKIP_VIDEOS (8): `src/our_method/data_utils.py`
- Stale files to ignore: `ImpliHateVid/offline_test_qwen2.5-vl-72b-awq.jsonl` (175 rows, non-IH prompt), `offline_test_band_*` (band-only prompt ablations)
