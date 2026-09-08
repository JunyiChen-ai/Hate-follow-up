# Rebuttal State — Submission 9795 (TRIAGE)

- **Venue**: ACL ARR 2026 May (OpenReview) → preferred EMNLP 2026
- **VENUE_MODE**: per_reviewer_thread (ARR official comments per reviewer; no hard char limit, target ≤ ~5000 chars per reviewer response)
- **Round**: initial rebuttal
- **AUTO_EXPERIMENT**: true (user explicitly asked: "把该补实验补了" — run the needed supplementary experiments)
- **Current phase**: COMPLETE — all evidence + drafts + follow-ups ready; awaiting user review before posting to OpenReview
- **Date started**: 2026-07-09

## Phase log
- 2026-07-09: Phase 0-2 done — REVIEWS_RAW.md, ISSUE_BOARD.md created. 3 reviewers: ARuf (3, positive-lean), cpjh (3.5, pivotal, confidence 5), 7rqV (3, swing-negative).
- 2026-07-09: Phase 3 strategy — 4 experiments identified (E1 prevalence, E2 small-verifier panel, E3 72B zero-shot control, E4 SAGE/recent-supervised). Inventory + literature subagents dispatched.
- 2026-07-09: E4 done — SAGE (ACL 2026 long.817) verified supervised MoE, no code, HateMM 87.10 on random 7:1:2 split → cite + footnote, no reproduction (LIT_SAGE_REPORT.md). HCC1 fixed-split 85.4/84.8 = cleanest supervised comparator.
- 2026-07-09: ⚠ INTEGRITY: HateMM headline 86.5 unreproducible from current files (72B verdict file rerun 2026-05-05 post-freeze, one in-band flip) → RE-PINNED to 185/215 = 86.0, avg 83.06. Also: Table-1 zero-shot block mixed provenance; §4.1 "10 random seeds" false for deterministic pipeline. All → REVISION_PLAN; forensics → PAPER_NUMBER_DRIFT.md (engineer writing).
- 2026-07-09: Jobs — 16344 (GPU fills, running), 16345 (GPU 72B zero-shot EN/ZH/HM + q32-EN spot-check, queued), 16346 (CPU diag+E1+E2, queued).
- 2026-07-09 (later): E1 done (calibration survives 5-20% prevalence; TRIAGE > mapper at all π; 5% tradeoff honest). E2 done (small panel 81.4 avg, beats all label-free baselines). E3 resolved from frozen files — no GPU needed (72B uniform avg 76.0 vs TRIAGE 83.1; 32B > 72B uniformly on all 4). 16344 hung→scancelled; 16345 72B step crashed (awq illegal memory access) but spot-check landed: Table-1 zero-shot rows = per-dataset-def protocol. Generic-def regen → camera-ready (integrity-3).
- 2026-07-09: SAGE code found public (user-supplied footnote after two wrong "not found" reports) → E4b reproduction on fixed splits launched (sage-repro agent).
- 2026-07-09: Phase 4-6 done — three per-reviewer drafts written, self-linted (3 provenance fixes), Codex xhigh stress test round 1 applied (MCP_STRESS_TEST_round1.md). REVISION_PLAN.md written (8 reviewer items + 4 integrity items). Drafts: ARuf 4233 / cpjh 4571 / 7rqV 5392 chars. PENDING: SAGE repro numbers (E4b) to post as thread follow-up; E3 fills job 16348; user review of drafts before submission.

## Evidence ledger (provenance)
- Paper numbers: paper/sections/4_experiments.tex Table 1/2 (frozen artifacts in results/paper_stage2_experiments/, results/paper_full_experiments/)
- New rebuttal experiments: results/rebuttal/ (to be created; every number must trace to a file there)

## Compute constraints (cluster)
- All jobs via sbatch; ≤2 GPUs total (running+queued); conda env SafetyContradiction (vllm 0.11.0)
