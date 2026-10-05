# Program23 R4 source-bound code review

2026-10-05 — **PASS, same-family provisional**, separate reviewer instance from the author. Final R4 revision only; no proposal or performance re-review.

Independent evidence: `runs/20261005_m1_program/bound_code_review/{report.md,oracle.py,routes.py,summary.json,run.log}`. Actual CPU environment: torch 2.11.0+cu128 / transformers 5.15.1, random Qwen3VL with 36 language layers and 32 query heads, FP32/BF16 with 18/20 images. Reused only the author's input adapter, not oracle results.

Confirmed the minimal explicit-position correction for 4D attention masks: native positions and dense hidden/margin exact. All 36 physical masks match the intended causal source restriction. Perturbing all dropped image KV leaves hidden/margin exact; selected KV perturbation changes hidden. Original cache and subsequent native replay remain exact.

Production `read_video`/`validate_records` passed action, join, fresh native fallback and missing-speech fixtures: 14 reader forwards including three separately charged diagnostics. Current prefix binding, native visual question, R3 speech and immutable full-source cost accounting passed. R1–R3 complete production returns match the saved pre-R4 source snapshot `prior_handle_measure_20261005.py` exactly on the same actual-model fixture. Ownership/UNKNOWN/foreign-window tests, unequal image-token groups, canonical evaluator/r6 command capture, launcher syntax and revision4 smoke/main/analysis routing passed.

No remaining concrete blocker. No production edits, GPU, pretrained weights, real GT or prediction/metric files were used. CPU fixtures do not replace target 8B smoke or demonstrate improvement. Preserved text/global/stance KV may already contain image information: this is direct image-key restriction, not clean causal isolation. Author full333 source checks remain separately attributed; this reviewer did not rerun them.
