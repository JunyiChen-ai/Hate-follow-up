# Lab-server native runtime and launch confirmation — PASS

2026-10-06. Independent GPT-6-astra reviewer, same-family provisional. This is a narrow infrastructure confirmation; Candidate 36's completed scientific review is not reopened. No production change was needed.

Scope: `scripts/check_native_runtime.py` output/command metadata additions, `experiments/20261006_m1_videoevent/launch/lab_server_native.sbatch`, and `launch/lab_server.sbatch`.

Independent evidence: `runs/_setup_lab_server_hatevlm/code_confirmation/{check.py,summary.json,run.log}`. Executed with `.cache/envs/HateVLM/bin/python runs/_setup_lab_server_hatevlm/code_confirmation/check.py`; result `RUNTIME_LAUNCH_CONFIRMATION_PASS`.

- Parsed current and prior committed script source as AST. The entire computation from `allrows` through the final completion log is unchanged: same five-video selection, native 20-frame/full-ASR path, old raw comparisons, windows, V/S calls, maximum and exact equality checks. Only parser/output setup and command metadata changed. No old prediction file was read by this confirmation.
- Executed only the actual parser/output-setup AST against an isolated fixture directory. Default remains `runs/_setup_local_hatevlm/native_smoke`; requested lab-server relative output and an absolute in-root output pass. Relative traversal, absolute outside-root and symlink escape are rejected before directory creation. The real launch explicitly selects `runs/_setup_lab_server_hatevlm/native_smoke`.
- Config command now includes the actual script argv and its `--out` value. This is readable command provenance; native numerical inputs are unaffected.
- Both shell scripts pass `bash -n` and explicit resource checks: `local-sc448960`, one GPU, four CPUs and 32G. Cwd is `/home/junyi/Hate-follow-up`; environment activation is `/home/junyi/miniconda3/bin/activate HateVLM`; model, Triton and CUDA cache paths are in the project, with offline model loading.
- Executed unchanged shell bodies using shell-function stubs for cwd, activation, GPU inspection and Python. Default scientific execution calls extraction then reader, both with `--smoke`; `SCOPE=main` calls both without that flag. Simulated extraction exit 23 prevents reader execution. Simulated `nvidia-smi` failure prevents any Python call. Native launch passes the required isolated output argument. Both scripts use `set -euo pipefail`.

The native and scientific scripts are separate jobs, so their cross-job order is the submitting agent's responsibility: first obtain actual native-five exact equality on lab-server, then submit the scientific fixed-five smoke. This CPU confirmation does not substitute for that runtime result. Source acquisition and reader stay in the same scientific job/host; no cohort split or scientific path change is introduced.

Remote availability, package versions, media and model presence reported by the author were not independently rechecked here. This confirmation started no GPU/remote workload and read no GT, actual prediction values or metrics. PASS establishes the narrow output/launch behavior, not native equality on lab-server, source-generation success or method performance.
