# Candidate34 OrderedSlots R3: independent narrow delta confirmation

Date: 2026-10-06. Reviewer: separate GPT-6-astra instance, same-family provisional.

**PASS within the requested R3 implementation scope.** No concrete implementation defect was found. This confirms the declared delta and its CPU execution/input plumbing; it does not establish pretrained 8B behavior, performance, mechanism, or promotion.

## Scope and inspected behavior

Reviewed the R3 declaration in experiments/20261006_m1_ordered_slots/README.md, measure.py, analyze.py, launch/lab1_R3.sbatch, run_analysis.sh, and the available reader fixtures. R3 restores the exact R1 source-packet literal_ASR entries for LOCAL/current, prequel and sequel. It retains R2's fresh native S and its requirement for both actual LOCAL frames and nonempty REMOTE source_ids; absence uses fresh native V without a source-image branch. Native full-ASR prefix, visual question, native G/own stance, model path, source selection and extraction are unchanged.

Independent AST comparison against runs/20261006_m1_ordered_slots/r3_reader_cpu_checks/before_R3_measure.py removed only the six declared revision-dependent changes (accepted revisions, B assertion, packet ASR condition, method label, bundle and config revision markers) and obtained identical remaining AST. This uses the readable source snapshot, not a content identifier.

## Independently executed checks

Evidence directory: runs/20261006_m1_ordered_slots/r3_delta_review/.

- model_check.py / model_summary.json: 20 actual CPU random-weight Qwen3-VL 36-layer cases: FP32/BF16 × native 18/20 frames × no_remote/prequel/sequel/both/no_frames. Production read_video and validator executed. Native S is exact; source-absent V is native with no source branch; active R3 V and source evidence equal corresponding R1 results. Every layer's K/V and rope state remain exact around scoring. Middle no-speech windows, native clone/vision accounting, forward counts and full source cost are covered. Both current R1 and R2 produce whole-output equality against the pre-R3 snapshot with deterministic clocks. Missing/wrong revision, S/question/source-suffix mutations, zeroed source cost and invented source evidence are rejected.
- The model fixture uses small random weights and an explicit ASR-presence-sensitive synthetic token adapter. It proves branching/cache arithmetic and makes the tested packet delta numerically observable; it is not a real language-semantic or pretrained-model test.
- source_check.py / source_summary.json: all 333 actual B source metadata records, 7,359 windows, current ASR, stored caption/slot embedding assignments and 14,695 unique per-video PNGs checked read-only. All 7,359 R3 packet contents and ordered image paths equal R1; removing only literal_ASR fields produces R2 contents. There are 3,071 remote-assigned windows and 3,069 source-eligible windows; all 3,069 eligible packets were processed with the actual CPU tokenizer/image processor and production encode_branch.
- Native 20-frame/full-ASR prefixes were rendered for every video. For the first eligible packet of each applicable video, both hypothetical Yes and No contexts were checked against full rendering: 504 references with exact expanded token IDs, image grids and all three mRoPE axes. No actual prediction or stance score was loaded. PNG format/dimensions were parsed; metadata bytes/mtime and PNG size/mtime stayed unchanged. Saved generation/embedding accounting reconciles to 458,703 source forwards. This checks existing source inputs and processor bindings; it does not regenerate raw-video crops.
- wiring_check.py / wiring_summary.json: R1 still requires changed S > 0; R2/R3 require changed S == 0, with the remote and changed-V requirements retained. Reader/analyzer namespace expressions agree for R1 A/B, R2 B and R3 B smoke/main. Canonical evaluator and fixed r6 commands were intercepted and inspected, never evaluated.
- analysis_shell.py / analysis_shell_summary.json: eight shell-stub cases cover default R1 A, R1 B, R2 B and R3 B, successful preparation and preparation failure. Distinct log/PID directories are correct; failed strict preparation prevents subsequent evaluation/report commands.
- bash syntax and launcher stubs confirm local-sc474397, one GPU/four CPUs/32G, repository cwd, HateVLM activation, offline caches, SOURCE_INTERFACE=B and READER_REVISION=3. Default smoke and explicit main invoke only measure.py; invalid SCOPE and failed device check stop execution. These are stub checks, not Slurm or CUDA execution.

## Limits and disposition

R3 output is isolated under r3_full_smoke_B / r3_full_main_B and matching analysis paths; it does not merge earlier predictions. R1/R2 default behavior is retained in the tested production paths. The revision2/3 designation follows the recorded declaration; this review does not assess the prior error analysis or performance evidence.

No scientific/author/foreign files were edited. No GPU, Slurm submission, GT, real prediction values, metrics, content hashes or Git identifiers were used. All executed tests passed. Actual pretrained fixed-five and whole-333 execution remain outside this CPU confirmation.

The original task is preserved in request.txt, executed command/tool trace in command_trace.json, and reviewer response in response.md within the evidence directory. The four check scripts, logs and summaries above are the reproducible evidence.

