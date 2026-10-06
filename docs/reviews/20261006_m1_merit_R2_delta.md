# MERIT33 R2: independent narrow delta confirmation

2026-10-06. **PASS — same-family provisional.** This confirms only the
predeclared R2 speech-reader change and its execution isolation. It does not
reopen the candidate's initial code/proposal review. No subagents, GPU jobs,
GT loading, actual predictions or performance metrics were used by this review.
The reviewer changed no production scientific code.

The readable pre-R2 `measure.py` snapshot and current implementation differ
scientifically only in whether remote speech is appended to the new S question.
R1 remains the default. R2 uses the original speech question in a fresh forward,
asserts exact equality to fresh native S, and marks/isolate its records and runs
as revision 2. The V memory builder—including literal ASR, LOCAL-only behavior,
source IDs and actual images—is unchanged. Source collection, MaxSim/key/filter
logic, constants, native G/stance, max and fixed r6 are not changed by this delta.
R2 reads the original source metadata and runs a fresh reader; it does not splice
old R1 predictions or regenerate source metadata.

Independent evidence is in `runs/20261006_m1_merit/r2_delta_review/`:

- **Actual CPU reader:** `check_reader.py` / `reader_summary.json` pass 16 cases:
  FP32/BF16 × native18/20 × remote-both, LOCAL-only, no-frames and no-speech.
  Actual Qwen 36-layer/32Q/8KV × 128/three-DeepStack forwards use random weights
  and an explicit remote-ASR-presence-sensitive fixture. With deterministic
  timing, current default R1 bundles exactly equal the entire readable pre-R2
  implementation's bundles. R2 retains every V/source-branch result, native
  prefix/G and source cost, while new S exactly equals native S. The fixture
  actually changes R1 S when remote speech is supplied; equality in R2 is not
  merely an insensitive fixture. There are 516 exact full-KV/mRoPE restoration
  checks across native, source-image and clone branches. Wrong revision, S,
  speech question and source-cost records are rejected.
- **Original inputs and costs:** `check_inputs.py` / `input_summary.json` bind
  all 333 current source metadata records to current ASR, native image paths,
  window ownership and source PNG shapes. All source hosts are `sc448960`.
  Independently recomputed original counts are 874,507 source forwards, 7,352
  source vision forwards and 15,981.468620364089 source seconds
  (266.3578103394 minutes). All remain charged; no source was regenerated.
  The check covers 14,695 source PNGs, 6,658 native image paths and 7,359 windows.
  It additionally runs full existing source-proof replay for the first video
  of each corpus and 82 real native-processor visual-branch comparisons across
  both hard stances. R1 and R2 visual tokens, pixels, grids and positions are
  exactly equal, with literal local/remote ASR preserved. This is input/software
  evidence, not pretrained reader performance.
- **Isolation, launch and guards:** `check_launch.py` / `launch_summary.json`
  use relocated shell stubs, never actual GPU or evaluator commands. R2's
  sc448960 launcher requests the declared local partition, one GPU, four CPUs
  and 32G; it calls only revision-2 `measure.py`. Smoke/main dispatch and invalid
  SCOPE rejection pass. Analysis uses isolated R1/R2 paths and inherited revision
  values. A forced strict-prepare failure prevents environment switching and
  both evaluation stages. Captured commands retain the sole canonical evaluator
  and unchanged fixed-r6 arguments. `traces/guard_summary.json` independently
  exercises the actual analyzer expression: R1 still requires changed S, R2
  requires zero changed S, and both retain remote-source/new-V execution guards.

No observation-changing delta bug was identified and no production correction
was requested. One reviewer shell-fixture path substitution initially failed;
only that reviewer-owned fixture was corrected, and its failed log is retained.
The full task, response, commands, logs, readable delta and reviewed-source
snapshots are recorded in `runs/20261006_m1_merit/r2_delta_review/traces/TRACE.md`
and adjacent files, without content digests or Git identifiers.

This PASS does not establish pretrained R2 numerical behavior, effectiveness,
component attribution or promotion. The already planned same-cohort real8B
fixed5 and complete333 run remain unexecuted by this reviewer. No scientific
standard, cost obligation, revision budget or scheduling priority was changed.
