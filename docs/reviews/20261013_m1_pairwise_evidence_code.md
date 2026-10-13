# Independent code review (rule 6): candidate 42, evidence-bound pairwise comparison (2026-10-13)

Commit requested: 6abcb5c. Working tree at review time: fccd901 (three commits later: `int(...)` cast of the
swap counter in `compare_analyze.py:97`, lab2/lab3 sbatch files source the conda `HateVLM` env instead of the
`.cache/envs/HateVLM` symlink, README pilot section). The code differences between 6abcb5c and fccd901 were
read; none changes the numbers, so the findings below apply to both.

Scope (rule 6 only): does the comparison enter the final score, is any label or GT read in the scoring path,
feature / timestamp / 4 fps grid / video_id alignment, cache correctness against the current inputs, use of the
sole evaluator and the fixed r6 decoder. Style, robustness, refactoring and theory were not reviewed.

Files: `experiments/20261013_m1_pairwise_evidence/{README.md,compare.py,compare_analyze.py}`,
`launch/lab{1,2,3}_compare.sbatch`, `launch/run_compare_analysis.sh`. References read: `src/stance_cache.py`,
`src/mllm_judge.py`, `experiments/20261007_m1_streamingtom/{placement_controls,inputs,measure}.py`,
`experiments/20260912_pwc/pilot0.py`, `archive/experiments/20261010_m1_grounded_adjacent/probe_native_analyze.py`,
`experiments/20260926_twolevel/{twolevel_r2,twolevel}.py`, `src/eval/evaluate_four_datasets.py`, one placement
record, one r6 prediction line, the finished pilot (Slurm 406, 5 videos / 158 windows / 316 edges) and its
`prepare --smoke` output. No GPU code was run. CPU checks run by the reviewer are listed at the end.

## Verdict

No blocker. The 333 run can proceed with the code as committed. One launch precondition outside the code
(finding 1) must be met before the HateClipSeg job is submitted on lab3.

## Findings

1. FIX (launch precondition, no code change) `compare.py:143-146` `placement_record()` opens
   `runs/20261007_m1_streamingtom/placement_controls_main/records/<ds>/<vid>.json`. lab3 has 0 / 0 of these
   records (lab2 has 215 / 118, lab1 all), so `sbatch lab3_compare.sbatch` would run the native reads of the first
   video and then stop with `FileNotFoundError`; no wrong number is produced, but the planned dispatch (README
   "HateClipSeg on sc474398") does not run. Before submitting on lab3: `git pull` (lab3 is at 03e41ef), rsync the
   placement records, and `mkdir -p runs/20261013_m1_pairwise_evidence` on lab2 / lab3 (absent on both; the
   `#SBATCH --output` path must exist). lab2 / lab3 `HateVLM` env versions equal lab1's job env
   (torch 2.11.0+cu128, transformers 5.15.1, numpy 2.5.2), so the strict per-window equality of the native reads
   with the placement records (`compare.py:166`) is expected to hold there as it did for the earlier jobs.

2. NOTE `compare.py:47-54` smoke selection. `--smoke --datasets HateMM` would write 3 records, while
   `compare_analyze --stage prepare --smoke` (`load_bundles`, `selected_rows(smoke)` with both datasets) expects the
   5 fixed videos and would stop on the missing HateClipSeg records. The sbatch default passes both datasets, and the
   pilot was run that way, so nothing is affected; recorded only so that nobody runs a one-corpus smoke.

3. NOTE Pilot gate reading (README "Pilot on the fixed five"): swap agreement is .599 on the 192 HateMM pilot
   edges and .766 on the 124 HateClipSeg edges; the declared rule is on the pilot edges together (210 / 316 = .66
   ≥ .6) and was applied that way. Not a code issue; stated here because HateMM alone sits under the threshold.

4. NOTE `compare_analyze.py:221-223` reads the rule-8 gate as "the same metric gains ≥ .01 in both corpora"
   (`common`). This matches `RESEARCH_ITERATION_RULES.md` rule 8 ("至少一项指标在两语料上都提升 ≥ .01") and the
   README sentence; recorded so the stricter reading is not mistaken for a bug later.

## Checks against the brief (all passed)

1. Comparison enters the score. `read()` stores `l_ij = (d_ij − d_ji)/2` per edge (`compare.py:172-175`);
   `per_video()` reads them back and `solve()` gives `u` (`compare_analyze.py:118-133`); `compositions()` forms
   `full_l = l`, `cmp_order = mean(l) + u`, `cmp_sum = l + u` and the twelve permutation arms
   (`compare_analyze.py:136-140`); `compose()` writes the value into `extra.windows[*].z_visual`, recomputes the
   fused `z = max(z_visual, z_speech)` (or `z_visual` when the window has no speech, as `prediction()` and
   `measure.prediction`), and rebuilds `score_curve` with the same index rule
   `clip(((arange(ceil(4·duration)) + .5)/4 // 8), 0, n−1)` as `compare.prediction()`, `measure.prediction()` and
   the previous candidate's analysis (`compare_analyze.py:110-115`). The r6 decoder reads `extra.windows[*]`
   `z_visual` / `z_speech` (`twolevel.prep`) and the fused `z` for the video key (`twolevel.intercept`), both of
   which `compose()` updates. Reviewer's CPU run on the 5 pilot records: `cmp_order` and `cmp_sum` curves differ
   from `full_l` on 5 / 5, `u` is zero-mean, `mean(z_visual)` per video is identical across the three arms,
   `corr(u_i − u_j, l_ij)` = .94–.98 (least-squares fit, sign preserved), `perm_order_0 ≠ cmp_order`.
2. No label in the scoring path. `compare.py` imports nothing under `data/gt_4fps`; it reads the manifest
   (durations), k20 frames, ASR, the placement records (asserting `checks.GT_read is False`) and writes
   `GT_read=False`. `compare_analyze.build()` reads the job records and the placement records only; GT is loaded in
   `diagnostics()` (called from `report()` only) and by `src.eval.evaluate_four_datasets` /
   `twolevel_r2.py` in subprocesses. `prepare()` reads r6's predictions (no labels).
3. Cache. `compare()` asserts the cache length equals `stance_cache_tokens` at entry, sets
   `rope_deltas = ctx['rope']`, and in the image path mirrors `suffix_margin`: render on
   `msgs + history + [user turn]`, `startswith(ctx['head'])`, `positions()` on the suffix shifted by
   `stance_cache_logical_start`, `attention_mask` popped, `past_key_values=cache`, `use_cache=True`; the text-only
   path (both windows without LOCAL frames; 7 of 7359 windows lack frames, no video lacks them in all windows, so
   this path occurs at most on a few edges) uses `cached_choices(..., in_place=True)` exactly as `margin()` uses
   `cached_margin`. The `try` opens before either path and `finally` does `cache.crop(n)` and restores
   `rope_deltas`, so both paths and any exception leave the cache at `stance_cache_tokens`; the entry assertion of
   the next call would catch a leak. `choice_logprobs` receives a 1-D row in both paths
   (`last_hidden_state[0,-1]`), and it does `hidden_row[None]` itself. Count assertions: per call forward +1 and
   vision +1 (images) or +0 (text); per video `forwards == 3 + n_windows + n_speech + 2·edges` and
   `vision == 1 + comparisons_with_images`; `prepare()` re-asserts both from the stored checks. Held on the
   pilot (632 comparisons, all with images). The A/B/C first-token sets are asserted disjoint and are
   {32, 64, 264, 362} / {33, 65, 293, 425} / {34, 66, 272, 356} in the pilot audit.
4. Graph and indices. `edges_for(n, vi)` uses `default_rng([0, vi]).permutation(n)`, distances 1 and 2,
   `(min, max)` pairs, sorted; complete graph for n ≤ 4; no labels. Self-check (`--stage selfcheck`, PASS on
   disk) verifies i < j, edge count, connectivity and exact recovery by `solve()` for n = 1..60, 90, 120. The same
   function is called by the job and by `prepare()`, which asserts the stored edge list equals it. `video_index` is
   the position in the full 333 list before the `--datasets` filter (`selected_rows`, enumerate then filter);
   reviewer check: HateClipSeg-only run gives first index 215, the pilot records carry indices 0, 1, 55, 215, 216,
   and `prepare()` asserts `checks.video_index == vi`. Sign chain: `d = log((pA + pC/2)/(pB + pC/2))` with
   `lp` aligned to `CHOICES = (A, B, C)`; `d_ij` from `compare(first=i, second=j)` (i shown as A), so positive
   `d_ij` means i more violating; `l_ij = (d_ij − d_ji)/2`; `solve()` builds `L u = b` with `b[i] += l`,
   `b[j] −= l`, which is the normal equation of `Σ (u_i − u_j − l_ij)²`, so `u_i − u_j ≈ l_ij` and positive `u`
   means more violating. Same convention as `pilot0.py` ("d('ij') scores window i as A").
5. Compositions. `l` is `traces[*].adjacent_local.z` or `native_visual` when `adjacent_local` is null
   (exactly the windows without LOCAL frames, verified on all 333 placement records); traces, `b['windows']`,
   `base.extra.windows` and `windows_for()` are in window order and `prepare()` asserts `i`, bounds, native V, S
   and LOCAL times agree across the four. Edge indices are window indices (`segs[i]`), `u` is indexed the same
   way, `compose()` zips `extra.windows` with the values in order. Permutation controls permute `l_ij` among the
   video's edges with `default_rng([100 + k, vi])`, k = 0..5, before the same `solve()`.
6. Alignment paths exist and are the declared ones: `runs/20260926_glr/base_gridA/predictions.jsonl` (333 rows),
   `runs/20260926_twolevel/r6_bma/metrics.json`, `placement_controls_main_decoded/{adjacent_local,adjacent_native}/`
   and `local_controls_main_decoded/local_clean/metrics.json`, placement records 215 / 118. `prepare()` asserts
   `z_video`, `extra.windows` and `score_curve` equal r6 (held for the 5 pilot videos; the reviewer confirmed the
   placement base equals r6 on all 333 as well). Reviewer check: `full_l` composed from the stored reads
   reproduces `placement_controls_main` `predictions.adjacent_local` (`score_curve` and `extra.windows`) exactly
   on 5 / 5 pilot videos, so the `report()` binding check (`full_l` decoded == `adjacent_local` decoded, 1e-9)
   should hold given the deterministic decoder with identical flags (the stored
   `placement_controls_main_decoded/adjacent_local/config.json` shows noleak, nscore, calib, bma, length,
   min_windows 2, bma_grid 6, arm m2, fusion or).
7. Launch files. Partitions lab1 `local-sc474397`, lab2 `local-sc474399`, lab3 `local-sc474398` match the
   machine table; `--gres=gpu:1`, 4 CPU, 32G (under the 56000 MB lab limit); `--output` and every job output
   under `runs/20261013_m1_pairwise_evidence/`; `HF_HOME`, offline flags set; `SCOPE` defaults to `smoke` and
   rejects other values; `${DATASETS:-HateMM HateClipSeg}` unquoted so the two names split into two arguments
   (intended; fine under `set -u`). `run_compare_analysis.sh` uses the HateVLM python for selfcheck / prepare /
   build and the HateVideo conda env for evaluate / report; `compare_analyze` imports cleanly under HateVideo
   (numpy 1.26.4, scipy 1.17.1, torch 2.7.1). The permutation seeds are drawn in `build` (HateVLM, numpy 2.5.2),
   the same env as the job; `report()` re-draws them under HateVideo but uses only `u`, `l`, edges and the
   decoded metrics, so the numpy version difference cannot touch a reported number.
8. Evaluator and decoder calls. `evaluate_one()` calls `python -m src.eval.evaluate_four_datasets --predictions
   … --gt-dir data/gt_4fps --datasets HateMM HateClipSeg --out …` and
   `experiments/20260926_twolevel/twolevel_r2.py --run <folder> --datasets HateMM HateClipSeg --noleak --transform
   nscore --key calib --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2 --out-root
   <selection_decoded> --tag <name>`, identical flags to
   `archive/experiments/20261010_m1_grounded_adjacent/probe_native_analyze.py:103-105` and to the placement
   analyses. `report()` reads the decoded `metrics.json` (`per_dataset` rows keyed by `dataset`), the same format
   as `r6_bma/metrics.json`.

## CPU checks run by the reviewer (read-only on the repository; nothing written outside this file)

- `edges_for` for n = 3, 5, 6 and the self-check output on disk (`runs/20261013_m1_pairwise_evidence/analysis/selfcheck.json`, PASS).
- On the 5 pilot records: `load_bundles(True)` → `per_video` → `compositions` → `compose`; comparison of
  `full_l` with the stored `adjacent_local` prediction; `u` mean, `corr(u_i − u_j, l_ij)`, fused `z` recomputation,
  curve length = `ceil(4·duration)`.
- All 333 placement records: `traces` length = window count, `adjacent_local` null exactly where LOCAL is empty
  (7 windows, 0 whole videos), `base` equals r6 (`extra.windows`, `score_curve`, `z_video`).
- `selected_rows` indices with and without the dataset filter; pilot audit `choice_ids`, host, completion.
- Import of `compare_analyze` under HateVideo; torch / transformers / numpy versions of the job env on lab1, lab2,
  lab3 over ssh; presence of placement records, frame caches, r6 predictions and the output directory on lab2 / lab3.
