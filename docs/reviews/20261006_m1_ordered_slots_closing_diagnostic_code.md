# Ordered-slots B closing diagnostic — narrow code review PASS

2026-10-06. Independent reviewer GPT-6-astra; same-family provisional. Scope is only `experiments/20261006_m1_ordered_slots/closing_diagnostic.py` and `launch/lab1_closing_diagnostic.sbatch`. This is an observational diagnostic, not a method, interface, performance revision or reopened candidate review. No concrete defect affecting the intended diagnostic was identified; production code was not modified.

Independent execution:

```bash
SOURCE_INTERFACE=B CUDA_VISIBLE_DEVICES='' .cache/envs/HateVLM/bin/python runs/20261006_m1_ordered_slots/closing_diagnostic_cpu_checks/independent/check.py
bash -n experiments/20261006_m1_ordered_slots/launch/lab1_closing_diagnostic.sbatch
```

Evidence: `runs/20261006_m1_ordered_slots/closing_diagnostic_cpu_checks/independent/{check.py,summary.json,run.log}`. All assertions passed. The independent script uses the real tokenizer, synthetic fixed heads/forward functions and synthetic main-loop source fixtures; it never loads pretrained weights or real source-generation outcomes.

- Original Stream versus AuditStream: masked compound quote, bare-quote termination, generation-cap exit and softcapped-logit cases preserve exact logits, tokens, events, selected value, forward-token sequence and positions. The observer reads the returned logits without in-place mutation, and adds no model forward. It records raw top tokens, quote-leading terminal tokens, bare-quote rank and the actual appended choice.
- Fixed-five IDs come from the existing `selected_rows(True)` manifest rule (first two per corpus plus HateMM `hate_video_114`); no alternative cohort is selected. Production main reconstructs each video's first original caption with `UNKNOWN` previous context and the same B caption writer/system/generation cap. Synthetic main-loop checks confirm that call binding.
- Main-loop tests reject differences in each of the nine compared fields: tokens, events, selection, prompt, input tokens, image grid, positions, rope delta and truncation. Failures raise rather than create a successful new diagnostic result. Source fixture bytes and nanosecond mtime remain unchanged across success and all nine mismatch cases.
- The successful five-video fixture checks release of prior stream cache/hidden references before the next generation. Original module Stream is restored after success and after every tested replay failure. The source-validation path is replay-only and introduces no model forwards.
- Shell syntax passes. Launch uses `/home/jehc223/Hate-follow-up`, `local-sc474397`, one GPU, four CPUs, 32G, `.cache/envs/HateVLM`, offline model/cache paths and `SOURCE_INTERFACE=B`. It calls only the diagnostic. Output is under `runs/20261006_m1_ordered_slots/closing_diagnostic_B`; no source cache writer, scoring or evaluator is invoked.

Operational condition remains unchanged: submit this diagnostic only after the original B fixed-five guard fails. The script itself does not decide or resubmit that run; the submitting agent must observe that prerequisite. B's existing prompts, caps and original guard remain unchanged.

Limits: synthetic checks establish noninterference and replay-failure behavior, not that masked quote closures caused the real 8B generation cap. Actual model-level equality is enforced by the diagnostic's field comparisons when it runs. No GPU workload, GT, real prediction scores, metrics, hashes or Git identifiers were used in this review. CPU PASS alone is not a pretrained-generation success claim.
