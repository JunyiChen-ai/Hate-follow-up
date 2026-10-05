# Candidate33 MERIT independent code review

2026-10-06 — **PASS, same-family provisional.** Reused independent reviewer instance distinct from the author; required once-only Rule6 implementation review, without proposal reconsideration.

Evidence: `runs/20261006_m1_merit/independent_code_review/{report.md,oracle.py,reader.py,summary.json}` and logs. Complete production acquire/validate passed on an actual decoded synthetic video: eight windows, thirteen filters, thirty-four embeddings, exact token/offset/source replay, bounded two-round retrieval and four-ID union. Altered current ASR, embedding offsets and selected IDs were rejected; source audit did not mutate PNGs. Provider choices/vectors were scripted, not model accuracy evidence.

Six independent actual 36-layer CPU reader tests cover FP32/BF16 with no remote, remote, and no-frame fallback. Native G/allraw, independent V/S cache/rope restoration, fresh fallbacks, original current/remote media ownership, diagnostics and full source cost accounting pass. Environment: torch 2.11.0+cu128 / transformers 5.15.1; only the author's input adapter was reused.

Known-vector MaxSim/ties/exclusions, actual source-ID JSON, unavailable/stopping paths, canonical evaluator/fixed-r6 command capture and launch syntax/resource inspection pass. No concrete blocker remains.

No production edits, CUDA, pretrained weights, real GT/prediction-score/metric access, content hashes or Git-ID provenance. Author full333/model evidence is separately attributed. This confirms the declared functional adaptation's implementation, not scientific improvement; actual GPU smoke remains next.
