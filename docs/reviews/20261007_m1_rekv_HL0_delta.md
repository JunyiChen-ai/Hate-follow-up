# ReKV37 HL0: independent narrow delta confirmation

2026-10-07. **PASS — same-family provisional.** This is a narrow confirmation of
the originally planned H0+L0 companion, not a new proposal, broad code review,
scientific revision or success gate. No observation-changing delta bug was
identified, and no production correction was requested. The reviewer used no
subagents, ran no GPU jobs, read no GT or performance-metric files, and changed
no production code.

The implementation freshly rebuilds source memory with zero source-to-source
history, then exactly replays H0 against the completed same-host H0 records.
It subsequently calls the existing selection batch with `arms=('L0',)`.
`equal_r0` checks source metadata, source/query arrays, selections, margins and
native evidence; only smoke-clone flags are excluded from replay identity.
HL0 retains every true LOCAL block and native whole-video context/G/stance/S,
while its selected explicit remote lists and direct/inherited source ancestors
must be empty. It is not context-free vision.

The sole increment to the shared helper is its optional arm subset, validation
of that subset and iteration over it. Default `arms=ARMS` preserves the original
seven-arm behavior and selection arithmetic. Atomic complete-video persistence
precedes dense-memory release; incomplete attempts use the established
preserve-and-rebuild path. Failed-attempt time/calls and batch peak after the
callback are retained. Actual batch work includes the fresh native/source/H0
replay plus HL0 queries and diagnostics; standalone estimates are separate and
retain original source decoding and the stated native20/ASR acquisition debt.

Independent evidence is under
`runs/20261006_m1_rekv/history_local_delta_review/`:

- `check_cpu.py` / `cpu_summary.json`: four actual Qwen CPU cases,
  FP32/BF16 × native18/20, with 36 language layers, 32Q/8KV × 128 and three
  DeepStack stages. Every case includes nonempty S and an uncovered no-LOCAL
  window. Fresh H0 source/key/query/selection/margin identity passes. All seven
  default arms execute; the subset L0 exactly matches the default batch's L0
  query arrays and predictions. Native KV/mRoPE/method restoration, clones,
  native fallback and rejection of corrupted ancestry, remote IDs, call counts
  and H0 margins pass. These use an explicit fixture tokenizer/random weights,
  not pretrained semantics.
- `check_inputs.py` / `input_summary.json`: independently completed strict
  current-input replay for **all 333** videos. Actual raw PTS/source PNG pixels,
  current native pixels and runtime, H0 source encoding evidence, positions,
  ancestry, query vectors and per-layer selection proofs all pass through the
  existing validators with the actual CPU processor. Prior H0 records are from
  `sc474398`. Only prior unlabelled main/H0 bundles and input/proof arrays were
  used; no metric was computed or inspected. This does not substitute for fresh
  pretrained HL0 execution.
- `check_orchestration.py` / `orchestration_summary.json`: isolated stubs verify
  history=0/L0-only dispatch, atomic completion then release, completed resume
  with zero new forwards, early/read failure audits, post-callback peak capture,
  HL0 output naming, smoke/main and invalid SCOPE handling, and prepare-failure
  blocking of evaluation. Captured evaluation commands use the sole canonical
  evaluator and unchanged fixed-r6 CLI. No GPU, real prepare/evaluator or Slurm
  submission occurs in this check.

The lab3 launcher retains the explicit sc474398 partition, one GPU, four CPUs
and 32G, with one sequential whole-cohort reader. Analysis names are isolated
under `history_local_*`. Its L0−HL0, H0−HL0 and interaction contrasts are
descriptive; `mechanism_supported` stays false and no new threshold or promotion
gate is added. HL0 cannot retroactively rescue failed original novelty claims
or establish dense-only novelty. Scheduling priority and revision budgets are
unchanged.

Full prompt, responses, commands, logs, readable helper delta and reviewed source
snapshots are retained in `traces/TRACE.md` and adjacent files, without hashes,
checksums or Git result provenance. Real8B fixed5/full333 HL0, measured hardware
costs and the resulting conditional scientific interpretation remain outside
this code confirmation.
