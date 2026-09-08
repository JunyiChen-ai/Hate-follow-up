# Revision Plan — Submission 9795 (TRIAGE), ACL ARR 2026 May

Paper: TRIAGE: Label-Free Hateful Video Detection with Boundary Mapping and Resolution.
Venue: ACL ARR 2026 May → EMNLP 2026. Mode: per_reviewer_thread (no char limit; target ≤5000 chars/thread).
Links: ISSUE_BOARD.md, STRATEGY_PLAN.md, PAPER_NUMBER_DRIFT.md, LIT_SAGE_REPORT.md, results/rebuttal/.

## Overall Checklist

Experiment-backed additions (rebuttal + revision):
- [ ] (ARuf-C1) Add prevalence-robustness study (E1) as new appendix section + summary sentence in §4; source results/rebuttal/E1_prevalence/ — commitment: `already_done` (experiment complete; paper edit pending)
- [ ] (cpjh-C3) Add small-verifier-panel table (E2) to §4.6/appendix + deployment-footprint paragraph; source results/rebuttal/E2_small_panel/table.md — commitment: `already_done` (experiment complete; paper edit pending)
- [ ] (cpjh-C1, 7rqV-C3) Add 72B single-pass control row (E3) to Table 1 label-free group or ablation discussion; source results/rebuttal/E3_72b_zeroshot/table.md — commitment: `already_done` (experiment complete; paper edit pending)
- [ ] (cpjh-C2, 7rqV-C2) Add SAGE (ACL 2026) to supervised group: reproduced-on-our-splits numbers if E4b completes, else cited numbers + split-protocol footnote — commitment: `approved_for_rebuttal` (E4b running)
- [ ] (cpjh-C2) Cite HCC1 (WWW Comp. 2025, fixed 217-test split, HateMM 85.4/84.8) and MM-HSD (ACM MM 2025, 5-fold CV, 87.8/87.4 with protocol footnote) in supervised-baseline discussion — commitment: `already_done` (numbers verified; paper edit pending)
- [ ] (7rqV-C2) Cite post-training moderation family (MuPHI arXiv 2605.29951, ExPO-HM ICLR 2026, RA-HMD EMNLP 2025) in related work with scope note (meme-domain, label-dependent) — commitment: `already_done` (verified; paper edit pending)
- [ ] (ARuf-C2, 7rqV-C1) Expand §2 related work: multi-(M)LLM application systems + selective-inference positioning; sharpen the "what generic coarse-to-fine does not give you" paragraph — commitment: `approved_for_rebuttal`
- [ ] (7rqV-C3) Add analysis paragraph: why label-free MLLM arbitration can exceed small supervised fusion models (training-set size ceiling vs pretraining knowledge; per-dataset zero-shot vs TRIAGE decomposition) — commitment: `approved_for_rebuttal`

Integrity corrections (self-identified; MUST land in revision regardless of reviewer asks):
- [ ] (integrity-1) Re-pin HateMM headline: Table 1 TRIAGE HateMM 86.5/85.8/86.2/85.5 → 86.0/85.3/85.8/84.9 (185/215 on current files); update Δ rows, §4 prose "+7.0" → "+6.5", abstract range check (2.5–8.7 unaffected: min IH, max ZH); update Table 2 ablation HM column and E4-related text. Source: PAPER_NUMBER_DRIFT.md — commitment: `approved_for_rebuttal`
- [ ] (integrity-2) Regenerate results/paper_stage2_experiments/{all_results.json, order_sweep.csv} and any figure fed by them from current files — commitment: `approved_for_rebuttal`
- [ ] (integrity-3) Regenerate ALL Table-1 zero-shot label-free rows under one documented protocol (per-dataset-def offline protocol, which reproduces IH + q32-EN cells; currently mixed provenance) and update Fig. resolver_prompt caption to state per-dataset definitions — commitment: `approved_for_rebuttal` (GPU job, camera-ready window)
- [ ] (integrity-4) Fix §4.1 "We report mean performance over 10 random seeds" — the headline pipeline is deterministic (GMM random_state fixed). Replace with the accurate determinism statement; seed-based variance is reported only where resampling exists (E1, pool-sensitivity) — commitment: `approved_for_rebuttal`. NEVER repeat the 10-seed claim in any reviewer response.

## Grouped view
- Table 1 & §4.2: integrity-1, E3 row, SAGE row, HCC1/MM-HSD cites.
- §2 Related Work: multi-LLM systems, post-training family.
- §4 analyses/appendix: E1 prevalence study, E2 small-panel study, 7rqV-C3 analysis paragraph.
- Backend artifacts: integrity-2, integrity-3.
- §4.1 setup text: integrity-4.

## Commitment summary
- already_done (evidence exists, paper edit pending): 5
- approved_for_rebuttal: 7
- future_work_only: 0
- needs_user_input: 0 (re-pin decision made by director 2026-07-09, surfaced to user; overridable)

## Out-of-scope log
- (ARuf-C3) venue fit — answered in thread; no paper edit (explicitly delegated to ACs by the reviewer).
- Generic-def (Fig. resolver_prompt) 72B zero-shot GPU regeneration for the REBUTTAL — dropped; superseded by the per-dataset-def control row (E3) and folded into integrity-3 for camera-ready.
