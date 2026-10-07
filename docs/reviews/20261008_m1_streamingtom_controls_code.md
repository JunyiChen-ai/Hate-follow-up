# Independent code review (Rule 6): StreamingTOM R1 mechanism controls 1–3

Date: 2026-10-08. Reviewer: one independent agent (same model family). Review was read-only. No file was edited, no GPU or Slurm job was run, and no hash was computed.
Code reviewed: commit `76a089d` ("Add StreamingTOM R1 mechanism controls 1-3 with default-preserving hooks"). sc474398 (lab3) is at the same commit.
Declaration reviewed against: `experiments/20261007_m1_streamingtom/README.md`, section "R1 mechanism controls 1–3 (declared 2026-10-08, before any control run)".

## Verdict: PASS (one conditional finding)

I found no bug that would change the observation or the conclusion if the jobs run as written and the recorded R1 comparison flags come back true. Finding 1 is a guard that is recorded but never enforced. It only matters if the flags come back false. Without the fix, a false flag on the `uniform` job would not stop evaluation, and the result would not show it.

## Findings

1. **The rebuilt-vs-R1 exact-comparison flags are recorded but not enforced. The `uniform` job has no other guard against a host-level numeric difference.**
   - **Where:** `controls_analyze.py:52-53` appends failures to `comparison_failures` only. `controls_analyze.py:82-83` writes `PASS=True` whatever `comparison_all_exact` is. `controls_analyze.py:91` (evaluate) checks only `PASS`. `report()` (`controls_analyze.py:99-120`) never reads `comparison_all_exact`.
   - **Why it matters:** the `uniform` job runs on sc474397, but R1 was generated on sc474398.
     - The LOCAL frames in the `uniform` arm are read from that job's own fresh vision features (`memory.local_features` → `block['features']`).
     - So if `features_exact` or `deepstack_exact` were false on sc474397, the `uniform` arm would differ from R1 in two ways: the declared uniform token selection, and different LOCAL embeddings. Its source-memory inputs would differ too.
     - `prepare` would still pass, and `report` would still issue a `component_supported` verdict for `uniform` with no sign of the second change.
     - The `dualpath` job is protected by the hard `replay == R1` assertion (`controls_analyze.py:77`). `uniform` has no in-job replay.
   - **What is already established:** native LM reads matching r6 bitwise across hosts is shown. r6 came from sc448960 and R1 reproduced it on sc474398. GPU vision features matching bitwise across hosts has not been shown for this pipeline: R1's strict replay on sc474397 recomputed grouping on CPU from the persisted features.
   - **Minimal fix:** in `prepare`, set `PASS=comparison_all_exact`, or add `assert not s['comparison_failures']` before writing `alignment.json`. Alternatively, copy each job's `comparison_all_exact` into `controls_analysis/summary.json` and require it before reading that arm's gate.

No other findings.

## What I verified

**1. R1 is bit-for-bit unchanged with default arguments**
- `reader.factory` / `visual_margin(pick=None)`: the `pick is None` branch is the original `select(...)` line, unchanged. `scores` is not used in the trace.
- `source_encoding.reduce(grouping=group)`: the default is the same `ctr.group` object.
- `collect.collect(grouping=group, persist=True)`: proof saving is the same, just moved inside `if persist`.
- Every existing caller passes these arguments by position (`measure.py:62,71,79`, `validate.py:37`, `model_selfcheck.py:43,48,49`), so the new keyword defaults apply.
- `controls_selfcheck.py` runs the production `read_video` plus strict `validate_bundle` with the hooks in place, in both the FP32/native18 and BF16/native20 fixtures. Result: `runs/20261007_m1_streamingtom/controls_cpu_checks/run.log`, `CONTROLS_CPU_PASS`.

**2. Each arm makes exactly the one declared change**
- **`no_remote`:** `pick` returns `[]` at every layer.
  - `translate([], stance_cache_logical_start)` returns `end = stance_cache_logical_start`. So the suffix positions are `relative + stance_cache_logical_start`, which is what R1 computes when it has zero remote blocks. These are also the `position_ids` the model receives.
  - The KV is `[native prefix, suffix]`, and the mask's past length equals the prefix length.
  - LOCAL embeddings, DeepStack injection, `reader_role_text`, question text and question rows all come from the same `encode_local`.
- **`nearest`:**
  - The candidates are all frames not in LOCAL. Since LOCAL is the half-open `[start, end)`, these are exactly the frames with `t < start` or `t >= end`.
  - The distance is `start − t` or `t − end`, as declared. Ties are broken by `(distance, frame index)`. I checked all 333 R1 records: frame `index` and `time` are strictly increasing together, so the lower index is the earlier frame.
  - The count is `min(4, number of non-LOCAL frames)`, the same as `select`. 11 covered windows have fewer than 4 candidates, and R1 and `nearest` behave the same way on them.
  - The same tuple is used at every layer. Packing order, positions and the KV path match R1. Only the `select` call is replaced.
- **`uniform`:**
  - `uniform_group` gives `g = min(BUDGET=50, n)` roots at `(2k+1)n // (2g) = floor((k+.5)n/g)`. Each root is a single member with weight 1.0, so `aggregate` returns the exact original rows for the projector and every DeepStack level.
  - There is no static/dynamic split, merging or saliency ranking. Saliency is still computed by `vision.extract` and is not used.
  - Everything else is unchanged: `pack`, the source history of the 4 previous blocks (`collect`), uint4/text KV (`Memory.append`), the representatives (mean pre-RoPE key over the retained visual rows), per-layer question `select` (`pick=None`), and LOCAL taken from full uncompressed features.
- **`replay`:** `pick=None` with the `ctr.group` memory. This is the same code path as R1 `read_video`, minus `save_native` (CPU-only processor work that does not touch the model or the cache) and minus the proof writes.
- **Arm independence:** the reader never writes to the cache (`cache.get_seq_length()` is asserted unchanged), restores `rope_deltas`, and reads memory layers from disk on each call. So running three arms in sequence within a window does not interact.

**3. The mechanism reaches the final score**
- Each arm's `values[a]` becomes `measure.prediction` (R1's 4 fps curve builder and `max(V, S)`), then `<job>/<arm>/predictions.jsonl`.
- In the evaluate stage, the sole evaluator runs on that file (raw), then `twolevel_r2.py` runs with flags identical to R1's `analyze.evaluate`. The decoder reads `z_visual`, `z_speech` and `z_video` per window and calls the same evaluator.
- The evaluator groups rows by `method`, and each file contains only one method.
- Windows without LOCAL re-read the native V (`margin`, asserted `== v`) and store it for every arm, as in R1. `prepare` re-checks this per window and per arm.

**4. Labels and GT**
- `controls.py`, `reader.py`, `collect.py`, `source_encoding.py`, `inputs.py` and `memory.py` never open `data/gt_4fps`.
- From each manifest row only `dataset`, `video_id` and `duration` are used.
- `data/gt_4fps` appears only as `--gt-dir` for the sole evaluator in `controls_analyze.evaluate`, and inside `twolevel_r2.py`, which passes it to the same evaluator. This happens after `alignment.json` PASS.

**5. Alignment and bookkeeping**
- The R1 reference record is loaded by the same `(dataset, video_id)`, and `frames == reference['frames']` is asserted.
- Windows come from `windows_for(row, segments)`, the same as R1. I checked all 333 R1 records: `local_ids` reproduces every stored LOCAL list, the trace count equals the window count, there are 7352 covered and 7 uncovered windows, and no R1 remote list contains a LOCAL frame.
- Question vectors are saved for `replay` in `dualpath` (and compared with R1 `question_vectors.npy`) and for `uniform` in `uniform`. `prepare` recomputes the `uniform` retrieval from those saved vectors.
- The forward-count assertion matches the arm structure.
- `report` takes R1 from `r1_full_main_decoded/optimized/metrics.json` and r6 from `runs/20260926_twolevel/r6_bma/metrics.json`. It requires the decoded `replay` metrics to equal R1 exactly, and checks the 84/99 cohort.
- The gate is `R1 − control ≥ .01` on the same metric in both corpora, across the three primary metrics. This matches the declaration.
- The decoder, evaluator and `src/temporal.py` have not changed since 2026-10-02, which is before the R1 decode of 2026-10-07.

**6. Launch scripts**
- `controls_lab3.sbatch` uses partition `local-sc474398`, runs `--job dualpath`, and uses the same conda `HateVLM` activation as R1's `lab3.sbatch`.
- `controls_lab1.sbatch` uses `local-sc474397`, runs `--job uniform`, and uses the repo venv `.cache/envs/HateVLM`. Its torch 2.11.0+cu128, transformers 5.15.1 and av 18.1.0 equal R1's recorded `config.json`.
- Both scripts use `SCOPE=smoke|main` (default smoke), 4 CPU / 32G, and write output, HF, Triton and CUDA caches only inside the repo. The analysis script uses the same envs as R1's `run_analysis.sh`.
- sc474398 has all 215/118 R1 records, the R1 proofs and the source cache that `compare()` and `frames_for` need.

## Notes (not findings)

- The `nearest` tie rule applies often. On the actual 0.5 fps grids, 3738 of 7352 covered windows have the 4th and 5th smallest distances exactly equal, and the rule gives that slot to the earlier frame. This is as declared.
- Across all R1 layer-reads, R1 retrieval shares on average 0.17 of 4 frames with `nearest`, and is identical in 0.46% of them. So `nearest` really is a different selection.
- Not checked: no GPU, real-8B or performance execution. Cost figures in the README remain forecasts.
