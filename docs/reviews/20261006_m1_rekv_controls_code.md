# ReKV37 controls: independent scientific code review

2026-10-06. **PASS after the concrete corrections below; same-family
provisional.** One independent reviewer instance; no subagents. This review
covers the new controls and their actual reader/validator increments, not a
second broad review of main R1 or its proposal. No GPU job was run, no GT or
actual performance metrics were loaded by reviewer checks, and no scientific
code was changed by the reviewer.

## Findings and same-review fix confirmation

The initial cost findings are preserved verbatim:

> controls.main config guard and load_asr run before try/finally, so config/ASR failures still lose attempt audit (same resolved main issue reintroduced); initialize audit/start before gate and wrap ASR too.

> read_video captures checks['peak_GiB'] before after_read=selection_batch, so controls peak excludes all seven arms and smoke clones; capture final peak after callback or separately in controls after it.

Both are fixed. `controls.main` now creates the attempt audit before configuration
validation and ASR loading, then executes both within the same `try/finally`.
Independent injected configuration, ASR and reader failures propagate while
persisting `completed=false`, positive elapsed time and actual call counters.
The rejected configuration remains unchanged. A new `batch_peak_GiB` is read
after the complete selection callback, including its clones, and aggregated by
`controls_analyze`; original R0 component timing/peak semantics remain intact.
An independent orchestration check asserts the order of callback completion and
peak capture, complete-record atomic writing, and a completed-record resume with
zero new forwards. This is a stub check of control flow, not a measured GPU peak.

During review the author briefly restricted deletion/binding gates to the
metrics that had passed main promotion. The frozen ablation plan and Rule 14(g)
allow any common primary metric satisfying their specified thresholds. The
reviewer identified that mismatch and the author restored the frozen gate.
`main_gain_concordance_metrics` is now a separate descriptive field. An
independent synthetic-number test passes a ROC deletion/binding comparison when
the synthetic main gain is within-only, confirming that the original gate has
not been silently narrowed. No actual control result was used to set a gate.

## Mechanism and implementation checks

- **R0 and selection arms:** the fresh source/native/query path is compared to
  main source metadata, representative arrays, query arrays, per-layer IDs,
  margins and native G/S. L0 removes explicit remote reads; C0 uses actual
  distance to the half-open interval; L1 shares this arm's layer-0 IDs; C1 matches
  each R0 layer's complete-block token multiset using chronology; L2 applies the
  layer-0 score preference under those per-layer token budgets. D0 maximizes
  nonoriginal donors within exact token buckets, preserving unavoidable original
  donors and using the declared cyclic preference. LOCAL is retained. Validators
  replay the actual arm rather than demanding false cosine-top4 proofs.
- **T0:** real timestamp substitutions enter `source_message` and are re-encoded
  through actual source model forwards. Actual pixels, indices, PTS, ownership,
  R0 selected IDs, packing, token budgets, image rows and spatial positions stay
  fixed; presented timestamps are separately recorded. The permutation attains
  the maximum cross-window assignment in its exact token buckets. It measures
  timestamp-conditioned memory and descendants, not an isolated role coefficient.
- **H0:** source-to-source direct and inherited ancestry are empty while the
  native global context remains. It is not an isolated-frame or context-free arm.
- **State, calls and outputs:** each query uses one model forward and no vision
  forward. Native G/S remain exact. Native KV, mRoPE state and model attention
  methods are restored across controls and an injected mid-layer failure.
  Complete video records are atomic; incomplete source memory follows the
  existing preserve-and-rebuild path. Whole selection batches use eight query
  passes: one fresh original R1 query pass plus the seven declared arms,
  including the additional R0 identity replay. Counters and timings retain this
  cost rather than presenting it as the minimal seven-pass forecast.
- **Evaluation and launching:** the two controls launchers preserve one host,
  original fixed5/full333 selection, explicit lab3 Slurm partition and normal
  sequential modes. Shell syntax passes. No GPU job is launched by this review.
  Scoring/matching has no GT access. Only the unchanged canonical evaluator and
  fixed r6 CLI produce metrics; the post-whole diagnostic helper calls canonical
  metric helpers and is not imported into scoring. Full six-metric/raw-ordering
  interpretation remains downstream work, not a software PASS claim.

## Independent evidence

All evidence is under `runs/20261006_m1_rekv/controls_code_review/`; reviewer
commands, logs, decisions and readable reviewed-source snapshots are retained in
`traces/controls-code-review/TRACE.md` and the same directory.

1. `operator_checks.py` / `operator_summary.json`: 160 exhaustive feasible-set
   D0 comparisons, 56 exhaustive admissible-permutation T0 comparisons, and
   independent L0/C0/L1/C1/L2/D0 selection oracles, including empty remote pools,
   insufficient alternate donors and heterogeneous token buckets. PASS.
2. `production_checks.py` / `production_summary.json`: actual Qwen 36-layer,
   32Q/8KV × 128, three-DeepStack CPU forwards in FP32 and BF16 with random
   weights, 12 decoded source images and an uncovered tail window. All seven
   selection arms, fresh R0 identity, nonempty S, native fallback, H0 and strict
   arm-proof rejection pass. A reviewer-authored timestamp-aware fixture extension
   puts displayed time text into actual source token sequences: T0 changes tokens,
   representative KV and output margins while frozen selection/packing and native
   predictions remain valid. An injected layer-9 selection failure restores KV,
   mRoPE and attention methods. These are software causality checks, not semantic
   understanding or pretrained 8B evidence.
3. `actual_processor_check.py` / `actual_processor_summary.json`: independent
   rerun of the unchanged real fixed5 using the actual native tokenizer and
   processor. All 624 presented-time substitutions change input token IDs; 609
   cross an actual 8-s ownership boundary. Exact pixel arrays, grids, image rows,
   token counts and positions are preserved. No model weights or metrics are
   used in this check.
4. `orchestration_checks.py` / `orchestration_summary.json` and `gate_check.py` /
   `gate_summary.json`: independently verify the two cost corrections, atomic
   completion/resume, failure audits and restored frozen gate. PASS.

No remaining observation-changing implementation bug was identified within this
control scope. Real pretrained 8B fixed5 controls, actual GPU/host-memory costs,
complete 333-video outputs, all final/raw metrics and mechanism interpretation
are still required. This code PASS establishes none of those outcomes and does
not promote ReKV37 or alter the frozen scientific standards.
