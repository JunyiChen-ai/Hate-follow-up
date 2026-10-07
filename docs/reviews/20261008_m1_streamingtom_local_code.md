# Code review: LOCAL decomposition of StreamingTOM control #1 against r6 (commit db57cf2)

Reviewer: independent code review, 2026-10-08, sc474397. This was a read-only review. No GPU was used and no job was submitted. Scope: the `reader.py` `role` argument, `local_controls.py`, `local_analyze.py`, `local_selfcheck.py`, `launch/local_lab1.sbatch` and `launch/run_local_analysis.sh`. These were checked against the declaration in `experiments/20261007_m1_streamingtom/README.md`, section "Decomposing control #1 against r6".

## Verdict: PASS

I found no bug that would change the observation or the conclusion. Below are the checks behind each of the seven review items, followed by three notes. None of the notes blocks the run.

## Checks performed

1. **B (`custom_native`) is r6's native read through the R1 attention code.** Confirmed.
   - Question ids: `local_controls.py:57` calls `j.branch_ids(ctx['msgs'], question, ctx['history'], head_text=ctx['head'])`. This is the same call as `src/stance_cache.py:34`, with the same `yesno_question(..., 'visual')` text (`local_controls.py:90`). No image tokens are added.
   - Positions in the native read. `_step` calls `Qwen3VLModel.forward` with no image grid and with `rope_deltas = ctx['rope']`. In transformers 5.15.1, `compute_3d_position_ids` then gives `arange(n, n+L) + rope_deltas`, where n is the cache length.
   - Positions in B. B passes `relative + stance_cache_logical_start`, and `relative` is asserted to equal `arange(L)` (`local_controls.py:58-59,64`). By construction in `build`, `stance_cache_logical_start = p.max()+1 = len(ids) + delta`, with `delta` asserted equal to `rope_deltas` and `len(ids)` asserted equal to the cache length.
   - I checked this on all 333 R1 records: `stance_cache_logical_start == stance_cache_tokens + native_rope` in 333/333.
   - Inside the R1 code, `end = translate([], start)[1] = start`, so `current = relative + start` at every layer. The per-layer trace assert at `local_controls.py:66` confirms `remote_ids == []`, `source_tokens == 0` and `suffix_logical_start == start`.
   - Readout: `j.margins_fp32(hidden[None])[0]` on `last_hidden_state[0,-1]` after the final norm. This is the same FP32 Yes/No margin as `cached_margin`.
   - **Other ways B differs from the native read, besides the attention function.** None of them changes a number:
     - B calls `language_model` directly instead of going through the `Qwen3VLModel.forward` wrapper. The embedding lookup is the same, and the positions are passed explicitly instead of being derived from `rope_deltas`; they are equal, as shown above.
     - The R1 attention ignores the `attention_mask` argument and builds its own boolean causal mask `columns <= past + row`. In the native read, past > 0 and L > 1, so `_ignore_causal_mask_sdpa` returns False and transformers materialises the same boolean pattern.
     - With a mask present, `use_gqa_in_sdpa` is False. The native SDPA call therefore also uses `repeat_kv` and `is_causal=False`, so both paths give SDPA the same kind of inputs.
     - The native read writes K/V into the cache and then crops it. B concatenates `prefix.keys` with the new K without writing. The key tensors are the same.
     - RoPE cos/sin are recomputed once per layer in B, from the same positions and the same dtype. The native read computes them once per model.
     - `mean_query` runs and copies to the CPU; this is a diagnostic only. `rope_deltas` is saved and restored.
   - Independent CPU check (scratchpad only, no repository writes): random-weight 6-layer Qwen3-VL fixture, with rope delta −10 so the offset is not trivial. The positions passed to the rotary embedding were identical at every layer, and |B − native| = 0.0 in both FP32 and BF16. This agrees with the author's `local_cpu_checks/summary.json` (0.0 / 0.0).
2. **E (`local_clean`) equals control #1 without the role text.** Confirmed.
   - Same LOCAL ids: asserted equal to control #1's trace at `local_controls.py:91`.
   - Same question object and same time-label format: `reader.py:16`.
   - Same pick: `nothing` returns `[]` at every layer; control #1's `picker('no_remote')` does the same.
   - Same position scheme: the suffix starts at `stance_cache_logical_start`.
   - Same features: see item 3.
   - The only change is `content=[]` instead of `[role text]` (`reader.py:14`).
   - Independent real-processor CPU check on 29 covered windows of the 5 smoke videos:
     - The suffix text of E equals the R1/control #1 suffix with the role string removed.
     - Grids and image counts are identical, and the question-row token ids are identical.
     - The question positions are shifted by exactly the 48 removed tokens on all three axes.
     - The default-role evidence reproduces the recorded R1 branch input exactly.
3. **`no_remote_replay` reproduces control #1 with the same LOCAL tensors.** Confirmed.
   - `ProofFrames.local_features(i)` reads `records[i]['full_projector']` and `full_deepstack` (`local_controls.py:49-50`).
   - `collect.py:55-56` saves these from the same `captured` tensors that `Memory.append` stored as `features`/`deepstack`. Control #1 read the latter through `Memory.local_features`.
   - `records[i]` and frame `i` are aligned (`block_id == i` in 333/333).
   - `controls_dualpath_main_analysis/alignment.json` has PASS, with no `comparison_failures` in either corpus (features and DeepStack exact for every frame).
   - The replay is checked exactly per window (`local_controls.py:105`), per video in prepare (`local_analyze.py:39,41`), and on the six decoded metrics in report (`local_analyze.py:76`).
4. **Defaults are unchanged.** Confirmed.
   - `role=SPEC['reader_role_text']` reproduces the old `content` list exactly.
   - The R1 (`measure.py:71,79`), `controls.py:105`, `validate.py:73`, `preflight.py:42` and `model_selfcheck.py:48-49` callers all use the default.
   - The real-processor check above reproduced the recorded R1 inputs exactly with the modified code.
5. **Scoring path.** Confirmed.
   - Each arm writes its own `local_controls_main/<arm>/predictions.jsonl` (`local_controls.py:164-166`).
   - Each then goes to `src.eval.evaluate_four_datasets`, followed by `twolevel_r2.py`. The flags are identical to `r6_bma/config.json` (noleak, nscore, calib, bma, length prior, min-windows 2, grid 6, m2).
   - Fusion is max(V, S) through `measure.prediction`. S and G are native and asserted unchanged.
   - No GT is read in the job. The decoder reads GT only through the evaluator subprocess.
6. **Analysis.** Confirmed.
   - Steps (`local_analyze.py:78`): `code_path = B − r6`, `frames = E − B`, `role_text = control#1 − E`, `total = control#1 − r6`. These add up exactly.
   - Floors: .005 / .005 / .01 (`local_analyze.py:15`).
   - "No effect" means all six |d| < floor.
   - The frames gate is: the same metric is ≥ .01 in both corpora, and every difference is > −floor (`local_analyze.py:80-81,87`). This matches the declaration.
   - The r6 reference is `runs/20260926_twolevel/r6_bma/metrics.json`, and native reads are asserted equal to the `base_gridA` reads it was decoded from.
7. **Launch scripts.** Confirmed.
   - `--partition=local-sc474397`, `--gres=gpu:1`, 4 CPUs, 32G.
   - HateVLM environment, the same as the control runs.
   - HF, Triton and CUDA caches are inside the repository.
   - `SCOPE` is smoke or main, and any other value fails.
   - Outputs go to `runs/20261007_m1_streamingtom/local_controls_{smoke,main}[_analysis|_decoded]` and `local_controls_analysis/`.
   - Evaluate and report use HateVideo, as in the controls analysis.

## Notes (none blocks the run; no fix required)

1. **`reader.py:14` / `local_controls.py:94-96`: removing the role text also changes how the first time label is tokenized.**
   - In control #1, the chat template joins the role text and the first label with no separator ("…interpretation.Actual LOCAL source…"). The tokenizer splits this into `'.Act', 'ual'`. In E, the label starts with `'Actual'`.
   - So control #1 − E removes 49 ids and adds 1: the role tokens, plus a re-split of the first word of the first label. All later ids are identical (508 trailing ids in the checked window).
   - Removing the role text cannot avoid this, and the effect stays inside the "role text" step. If the README describes E exactly, it can mention this.
2. **`launch/local_lab1.sbatch:3`: the replay crosses hosts.**
   - Control #1 ran on sc474398 (`controls_dualpath_main/config.json`); this job runs on sc474397.
   - The exact replay therefore needs the R1 attention path to give identical bits on both hosts. Native reads and vision features are already bit-identical across these two hosts, so this is likely.
   - If it fails, prepare stops (`local_analyze.py:39,41`) and no wrong number is produced. The cost is a repeated run on sc474398.
3. **`local_controls.py:101-103`: E uses native V on the 7 windows without LOCAL, while B uses its own read there.**
   - If B differed from native on those windows, that small code-path difference would be counted in the "frames" step instead of "code path". The three steps would still add up exactly.
   - This affects 7 of 7359 windows, and B is expected to equal native bit-for-bit. `alignment.json` reports `custom_native_max_abs` per corpus, so any such difference will be visible.

## Evidence locations

- The independent checks ran from the session scratchpad and wrote nothing to the repository. Scripts: B vs native on the fixture; E vs the recorded R1 input with the real processor, on the 5 smoke videos.
- Records and files read:
  - `runs/20261007_m1_streamingtom/r1_full_main/records/*` (333)
  - `controls_dualpath_main/records/*` (333)
  - `controls_dualpath_main_analysis/alignment.json`
  - `local_cpu_checks/summary.json`
  - `runs/20260926_twolevel/r6_bma/config.json`
- GT was not read.
