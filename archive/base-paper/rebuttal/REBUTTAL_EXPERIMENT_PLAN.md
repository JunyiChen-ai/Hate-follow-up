# Rebuttal experiment plan — Submission 9795 (TRIAGE)

All outputs land in `results/rebuttal/`. Every number quoted in the author response must trace to a file there or to an existing frozen artifact. Constraints: sbatch only, ≤2 GPUs total, conda env `SafetyContradiction`.

## E1 — Prevalence-shift robustness (ARuf-C1) — CPU sbatch

**Question**: does the label-free pipeline survive when hateful videos are a severe minority (<10%) of the sample set?

**Design** (transductive prevalence resampling, offline recompute):
- For each dataset and each target prevalence π ∈ {0.05, 0.10, 0.20, natural}: keep ALL normal videos in the eval set, subsample hateful videos to reach π (EN: 112 normal → 12 hateful at π=0.10; ZH: 104 → 11; HM: 129 → 14; IH: 201 → 22). Natural = full test set (sanity row, must equal headline).
- Calibration pool must match the scenario: EN/ZH/IH fit on their train-split scores subsampled to the same π (labels used only to *construct* the scenario, never inside the method); HateMM's pool is the (subsampled) test split, as in the paper protocol.
- Re-run the full pipeline per replicate: threshold protocol (TR-Otsu / TR-GMM / TF-li_lee per PROTOCOL dict) → logit 2-GMM posterior → entropy band (mean-entropy rule on the subsampled pool) → sequential resolver MAIN_ORDER (g27 > q32 > q72) with ρ_D re-derived from the subsampled band entropy, consuming frozen per-video verifier verdicts.
- Seeds: 30 per (dataset, π); report mean±std ACC and M-F1.
- Comparators on the IDENTICAL subsampled eval sets (from frozen full-test preds): (a) Stage-1 mapper only; (b) zero-shot Qwen2.5-VL-72B-AWQ; (c) zero-shot gemma-3-27b-it. This shows whether TRIAGE's *relative* advantage survives imbalance.
- Videos missing a verifier verdict (≤9/dataset): fall back to previous posterior (skip that verifier), count and report.

**Prediction**: absolute M-F1 drops for all methods at low π (fewer positives); TRIAGE's delta over mapper-only and over single zero-shot models persists; band still concentrates errors. **Disconfirmation**: bimodal calibration collapses at π≤0.10 (threshold inside the normal mode, band degenerates) → report the failure mode honestly + note the natural mitigation (prior-anchored calibration) as future work.

**Output**: `results/rebuttal/E1_prevalence/{summary.csv, raw.csv, README.md}`.

## E2 — Small-verifier-only panel (cpjh-C3) — DONE (extraction only)

Frozen in `results/paper_stage2_experiments/{order_sweep.csv, all_results.json}`. Best all-≤12B panel (internvl35-8b > qwen3-vl-8b > gemma-3-12b-it): avg ACC 81.36 / M-F1 0.7784 at 0.683 calls/video; per-dataset EN 77.0, ZH 81.2, HM 84.7, IH 82.5 — beats the best label-free/few-shot baseline on ALL four datasets (best baselines: EN 76.4 LoReHM, ZH 75.2 LLaVA-OV, HM 79.5 ALARM, IH 80.3 MARS) with max model size 12B.
**Task**: extract into `results/rebuttal/E2_small_panel/table.md` with a peak-VRAM/deployment note (12B bf16 ≈ 24–28 GB → single 48GB GPU; vs 72B-AWQ ≈ 40+ GB). Include all 6 small orders to show order-insensitivity.

## E3 — Strongest-backbone zero-shot control (cpjh-C1, 7rqV-C3) — tiny GPU fill + CPU eval

**Question**: is TRIAGE's gain just "a 72B backbone", and were baselines given the same scale?

1. Verify protocol match: confirm the offline verifier prompt = the single-pass binary protocol used for Table 1 zero-shot rows (check `results/paper_full_experiments/existing_baseline_outputs.csv` provenance). If protocols differ, label the new row as "verifier prompt, single-pass" and note it.
2. GPU fill job (one sbatch, 1 GPU): score the missing videos — HateMM 1 + ImpliHateVid 9 for qwen2.5-vl-72b-awq (and opportunistically the other verifiers' gaps: EN 2×2, ZH 1×2, IH ≤18 — total ≤50 videos) with the same offline verifier prompts.
3. CPU eval: Qwen2.5-VL-72B-AWQ zero-shot row on all 4 full test sets (ACC/M-F1/M-P/M-R, same eval_one protocol + SKIP_VIDEOS).

**Prediction**: 72B-everywhere lands well below TRIAGE (its verdicts already lose to TRIAGE inside the band by construction of the ablations) at ~26× the parameter-weighted cost. Combined with Table 1's 27B/32B rows and the w/ Self-Con ablation, this closes 7rqV-C3.

**Output**: `results/rebuttal/E3_72b_zeroshot/{preds fills, eval.json, table.md}`.

## E4 — SAGE (ACL 2026 long .817) + recent supervised baselines (7rqV-C2, cpjh-C2) — pending lit extraction

- SAGE: Synergistic Adaptive Gating of Experts for Hateful Video Detection — supervised, HateMM + MultiHateClip. Awaiting exact tables/split protocol/code from lit-check. Decision tree: same fixed splits + public code → attempt reproduction; different protocol → cite with protocol footnote; no code → cite reported numbers with split caveat.
- Verified comparable supervised row already available to quote: HCC1 (WWW Companion 2025, fixed 217-video HateMM split): ACC 85.4 / M-F1 84.8 — both below TRIAGE 86.5/85.8. MM-HSD (ACM MM 2025): 87.8/87.4 but 5-fold CV protocol (not directly comparable — must be footnoted).
- Post-training LLM family (7rqV): MuPHI (RL post-training, image-text), ExPO-HM (ICLR 2026, memes), RA-HMD (EMNLP 2025, memes) — all require labeled hate data and are meme-domain; cite + scope note.

## Determinism note (internal)
Paper §4.1 says "mean over 10 random seeds" but the headline pipeline is deterministic (GMM random_state=42). Do NOT repeat the seed claim in any response; add REVISION_PLAN item to fix §4.1 wording. E1's 30-seed resampling gives the honest variance story.
