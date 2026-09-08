# Experiment Audit Report: RAVEL

**Date:** 2026-08-28  
**Auditor:** fresh GPT-5.6-Sol ultra agent (same-family, provisional)  
**Overall verdict:** FAIL

## Checks

- **Ground-truth provenance — PASS.** The isolated evaluator loads real dataset
  `y4` annotations; inference manifests do not expose GT fields.
- **Score normalization — FAIL.** The inherited dense readout applies a
  per-video ECDF transformation and does not retain raw scores alongside the
  transformed scores. Interval F1 is unaffected.
- **Result existence — PASS.** Listed JSONL artifacts and all twelve saved
  summaries exist and independently recompute to numerical tolerance.
- **Dead metric code — PASS.** Pooled, within-video and interval functions are
  called. The per-dataset interval metric is event-micro F1 despite an
  inaccurate `macro-counted` docstring.
- **Scope/fairness — FAIL.** The selected fusion was chosen after inspecting
  fresh32 and was then evaluated on full288 containing that cohort. Only 278 of
  644 test videos enter the paired table; three RAVEL errors are skipped rather
  than fail-closed; one seed is recorded; 99/275 successful dense curves do not
  match the GT grid length.
- **Evaluation type — WARN.** Real-GT, custom evaluator, partial curve-eligible
  test cohort, adaptively reused for method selection; exploratory rather than
  held-out.

## Claim impact

- Saved per-dataset measurements: **supported descriptively**.
- Label-free inference at runtime: **supported**.
- Existing RAVEL boundary advantage: **exploratory only**.
- SOTA claim from paired278: **unsupported**. Dataset-macro F1@0.5 is 0.205,
  but event-pooled RAVEL F1 is 0.165 versus A10's 0.170; the apparent macro
  advantage is heavily influenced by four MHC events.
- End-to-end three-call claim: **incorrect**; the audited chain uses five model
  forwards including cached M1.

## Required actions

1. Freeze the simplified method before selecting a disjoint cohort outside all
   Stage-B, fresh32 and full288 records.
2. Count decode failures as empty predictions for interval metrics.
3. Report raw and transformed dense metrics and enforce the canonical grid.
4. Store exact code/input/model hashes and real call categories.
5. Treat current 288 results only as mechanism discovery.

`review_independence: same-family`  
`acceptance_status: provisional`
