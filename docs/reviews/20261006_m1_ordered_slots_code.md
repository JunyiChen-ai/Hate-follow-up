# Candidate34 ordered slots independent code review

2026-10-06 — **PASS, same-family provisional.** Reused independent reviewer instance distinct from the author; the unique required Rule6 implementation review, without reopening the accepted proposal.

Evidence: `runs/20261006_m1_ordered_slots/independent_code_review/{report.md,oracle.py,reader.py,pooling.py,summary.json}` and logs. Actual synthetic-video acquire→validate passed: ten windows, two slot batches, thirty-two embeddings, exact PTS/token/current-source replay and read-only pixel checks. Ordered/NONE/tie selection, sequential prior captions and actual-only final evidence passed; corrupt source/offset/assignment fixtures were rejected. Source provider choices/vectors were scripted software fixtures.

Twenty independent actual 36-layer CPU reader tests cover FP32/BF16 × 18/20 frames × preceding/following/bilateral/no-remote/missing paths. Native G/allraw, independent V/S execution, all-layer KV/rope restoration, actual source images, clone/counts and full cost accounting pass. Separate real-tokenizer 36-layer embedding tests match independent user-token FP32 mean/normalization exactly and restore rope. Environment: torch 2.11.0+cu128 / transformers 5.15.1; only the author's reader input adapter was reused.

Canonical evaluator/fixed-r6 command capture and launcher syntax/resource checks pass. No concrete blocker remains. No production edits, CUDA, pretrained weights, real GT/prediction-score/metric access, hashes or Git-ID provenance. This does not establish actual 8B/GPU behavior or scientific performance.
