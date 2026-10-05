# OTT exact source-replay thread correction

2026-10-05 — **PASS, same-family provisional.** Reused independent reviewer instance, distinct from the author; narrow implementation-fix confirmation only.

Independent evidence: `runs/20261005_m1_ott/replay_thread_fix/independent/{report.md,replay.py,summary.json,run.log}`. On torch 2.11.0+cu128 / transformers 5.15.1, production `select` replayed all current actual source caches at four threads: **333/333 full plans exact**, comprising HateMM 215 and HateClipSeg 118. No tolerance was introduced.

The harness executed production `prepare`'s new first statement, `torch.set_num_threads(4)`. Actual non_hate_video_137 reproduces the one-thread differences only in masses, contributions, transport cost and marginal residual; four threads match all saved fields exactly. Production measurement already uses four threads.

The diff adds only that setting and two comments. Native/current-input guards, exact plan equality, source/spec, prediction data, evaluator ordering and scientific computation remain unchanged. No concrete blocker remains from this mismatch.

No production edits, GT, prediction-score/metric fields, pretrained weights, CUDA, content hashes or Git-ID provenance were used. Only source-plan values were decoded from mixed record files. This confirmation does not assert completion of the separate full prepare or canonical evaluation.
