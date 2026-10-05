# Interval-witness R2 lab-server launcher — narrow confirmation PASS

2026-10-06. Independent GPT-6-astra reviewer; same-family provisional. Scope is solely `experiments/20261005_m1_interval_witness/launch/lab_server_r2.sbatch`. The scientific review in `docs/reviews/20261006_m1_interval_native_context_visual_code.md` remains unchanged and was not reopened.

Executed `.cache/envs/HateVLM/bin/python runs/_setup_lab_server_hatevlm/interval_r2_launch_confirmation/check.py`. Evidence: `runs/_setup_lab_server_hatevlm/interval_r2_launch_confirmation/{check.py,summary.json,run.log}`. Result: `INTERVAL_R2_SERVER_LAUNCH_PASS`; no production modifications needed.

The script is byte-for-byte equal to the lab1 launcher after only the declared partition, home-path and environment-activation substitutions. `bash -n` passes. It requests `local-sc448960`, one GPU, four CPUs, 32G and the existing project-relative `runs/20261005_m1_interval_witness/slurm_%j.out`. Cwd is `/home/junyi/Hate-follow-up`; activation is `/home/junyi/miniconda3/bin/activate HateVLM`; HF/Triton/CUDA caches stay in that project, with offline model loading and unchanged numerical environment settings.

Executed the unchanged shell body with functions replacing cwd, activation, GPU inspection and Python. Default and explicit `SCOPE=smoke` each call only `measure_r2.py --smoke`; `SCOPE=main` calls only `measure_r2.py`. There is no extraction, source regeneration, cross-host reading or split submission in the launcher. Injected cwd, activation and GPU-inspection failures stop before Python; injected Python failure exits with status 23. `set -euo pipefail` remains active.

This confirms launch behavior only. The author reports actual lab-server native-five equality and plans current source/token/pixel CPU replay before scientific execution; neither result was reread or replaced by this shell test. Native equality and successful source replay remain prerequisites for submission. Smoke and any subsequent complete run each execute on lab-server as a whole. No GT, actual prediction values, model inference, GPU task or remote action was used for this confirmation.
