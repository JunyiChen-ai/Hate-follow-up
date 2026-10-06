# Ordered-slots B analysis log/PID isolation — narrow confirmation PASS

2026-10-06. Independent GPT-6-astra reviewer; same-family provisional. Scope is only the output-directory routing fix in `experiments/20261006_m1_ordered_slots/launch/run_analysis.sh`, continuing the existing interface/launcher checks. No scientific review was reopened and no production files were modified by the reviewer.

Evidence: `runs/20261006_m1_ordered_slots/analysis_B_log_fix/independent/{check.py,summary.json,run.log}` and isolated shell fixtures. Executed `.cache/envs/HateVLM/bin/python runs/20261006_m1_ordered_slots/analysis_B_log_fix/independent/check.py`; result `ORDERED_SLOTS_B_LOG_FIX_PASS`.

`bash -n` passes. Exact comparison with prior committed source confirms that all stage commands from the HateVLM prepare invocation onward and all exported environment settings are unchanged. The analyzer itself is unchanged, including evaluator/r6 arguments and guards.

Executed the actual suffix expression, directory creation and log/PID redirections with only their destination prefix redirected into owned fixtures and Python/activation replaced by stubs. Eighteen cases cover unset SOURCE_INTERFACE, explicit A and B, each under success or prepare/activation/base-evaluation/optimized-evaluation/report failure.

- Unset and A write both `run.log` and `run.pid` to `runs/20261006_m1_ordered_slots/r1_full_main_analysis`.
- B writes both files to `runs/20261006_m1_ordered_slots/r1_full_main_B_analysis`, matching the analyzer's existing B output stem. The other interface's fixture directory remains absent.
- Prepare runs in HateVLM before HateVideo activation. Each injected stage failure returns its failure status and prevents all subsequent stages. No evaluator can follow a failed prepare.

This confirms path isolation and unchanged execution gating only. It does not establish main-run completion or successful actual prepare. The submitting agent must wait for complete paired-reader DONE and set `SOURCE_INTERFACE=B` for the intended analysis; the launcher does not itself wait for that job. A failure/pilot artifacts remain preserved. No GPU, GT, prediction contents or metrics were used, and the original analysis directories were not written during testing.
