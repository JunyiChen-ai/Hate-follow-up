# Code review: candidate 46 (topic-direction removal at read time), rule 6

Date: 2026-10-17. Reviewer: independent agent (rule 6 of `RESEARCH_ITERATION_RULES.md`). Scope: only bugs that would change what the experiment observes or concludes.

Files reviewed (working-tree state at 08:00 NZDT, i.e. commit `0b8ca9c` plus the two uncommitted edits to `directions.py` made during the review: zero-norm guard in `direction()` / `.5` for a constant or non-finite projection in `aucs_of()`, and the orthogonalised label-shuffled controls; README §2 updated for the second):
`experiments/20261017_m1_topic_removal/{topic_bank.py, common.py, directions.py, measure.py, analyze.py, launch/lab1_directions.sbatch, launch/lab1_measure.sbatch}`, `src/window_level.py`, README §2.
Infrastructure read, not edited: `src/mllm_judge.py`, `src/stance_cache.py`, `src/video_inputs.py`, `experiments/20260926_twolevel/{twolevel_r2.py, twolevel.py}`, `src/eval/evaluate_four_datasets.py`, `experiments/20260922_til/til_measure.py` (the code that produced r6 `runs/20260926_glr/base_gridA`), `archive/experiments/20261017_m1_generation_side/{analyze.py, common.py}` with `docs/reviews/20261017_m1_generation_side_code.md`, `archive/experiments/20261014_m1_speaker_bound/precheck.py`, the installed `transformers 5.15.1` sources (`models/qwen3_vl/modeling_qwen3_vl.py`, `cache_utils.py`), and the outputs of Slurm jobs 489 (failed), 490 (`directions_490_unorthogonalised/`) and 491 (`runs/20261017_m1_topic_removal/directions/`, `data/topic_direction/qwen3vl8b/`). No measure smoke has run; `measure.py` and `analyze.py` were checked by reading, plus a CPU tokenizer run for `directions.frame()`.

## VERDICT: PASS

No blocking bugs found.

### Blocking

None.

### Non-blocking notes

1. `common.py:23` `IMPLEMENTATION` still says "5 label-shuffled controls"; the controls are now orthogonalised against the topic direction and re-normalised (`directions.py:161-162`, README §2, `PROVENANCE.md`). The measure records and `config.json` carry this string, so a later reader of `runs/.../main/` will not see the orthogonalisation unless they open `PROVENANCE.md`. One-word edit before the measure run keeps the record self-describing; no number changes.
2. `measure.py:122-124` resumes from any existing `records/<ds>/<vid>.json`. `runs/20261017_m1_topic_removal/main/` does not exist yet. The direction files were rewritten at 07:52-07:53 by job 491; any measure run must start after that (it does, nothing has run), and no smoke directory should be resumed as `main`.
3. `measure.py:52-53` `proj_topic[l]` is the mean projection over all branch tokens (window header, slice, question, assistant header), not over the window text alone, so `spearman_group_hits_vs_proj` in `analyze.py:90` mixes the fixed question tokens into the per-window projection. It is report-only (README §2 lists it under "报告, 不设门"); no gate uses it.
4. `analyze.py:151` `calls=3 + 2 * len(ww)` counts two reads for silent windows too (candidate 45 counted speech windows). Cosmetic field; nothing reads it.
5. Outside rule-6 scope, recorded only because it bears on what the direction is: in `topic_bank.py`, 10 of the 16 protected terms are capitalised mid-sentence (Black, Jews, Muslims, ...) while all 16 non-protected terms are lower-case there. The direction and its D1 AUC (best .998 at the embedding output) therefore also carry a capitalisation / proper-noun component. The lists are declared in README §2 and the novelty review changed them, so this is a declared design choice, not a code error.
6. The `last` direction at the embedding output is the zero vector (same final token for every sentence; `directions.py:111-114` now returns zero instead of NaN, `validity.json last.auc_test_per_layer[0] = .5`). Only `directions_body.pt` is used by `measure.py`, so this has no effect on the reads.

## What was checked and found correct

### 1. `directions.py` (stage 1)

- Body positions (`directions.py:59-69`). CPU run of `frame()` with the real Qwen3-VL processor on all 1280 bank sentences: every sentence occurs exactly twice in the rendered text (transcript line and question), `rindex` picks the question copy, the selected token indices are contiguous and all lie after the last `<|im_start|>user`, and decoding them gives `" " + sentence + "\n\n"` (the first token carries the leading space, the last token is the final `.` merged with the two newlines that precede "Is THIS window ..."). Offsets are character offsets into `full` for this fast tokenizer; the overlap rule `b > start and a < end` selects exactly the sentence tokens, 0 failures / 1280. The body token counts in the run log (7-12) match.
- Recorded points (`directions.py:35-57, 96`). Forward hooks on `language_model.embed_tokens` and the 36 entries of `language_model.layers`; `Recorder` appends in firing order, which is embedding first (called from `Qwen3VLModel.forward` before `language_model`) then layers 0..35 (`modeling_qwen3_vl.py:829-837`); `assert len(rec.body) == L` (37) held for all 1280 forwards in jobs 489-491, so each point fires exactly once. The decoder layer returns a plain tensor (`modeling_qwen3_vl.py:537-566`), the hook handles both tensor and tuple. Layer 35's output is recorded before the final `norm`, the same place `measure.py` ablates.
- Fit / test (`directions.py:104-109`): `fit = even & ~attack` (even carriers minus 12, 14, 16, 18 = 16 carriers x 32 = 512 sentences), `test = ~even` (20 odd carriers x 32 = 640, including the 4 odd attack carriers). `validity.json D1` reports 512 / 640.
- Direction (`directions.py:111-114`): `mean(X[fit & prot]) - mean(X[fit & ~prot])`, normalised per point. Saved `directions_body.pt` is 37 x 4096 float32, all row norms 1.0000 (+-3e-7), finite.
- Shuffled controls (`directions.py:150-168`): for each of the 16 fit carriers the 32 protected labels are permuted with `torch.Generator().manual_seed(k)` (so each carrier keeps 16/16), the same mean-difference formula is applied on the fit mask, then the component along the topic direction is removed and the row re-normalised. Saved `directions_shuf0..4.pt`: 37 x 4096, all row norms 1.0, |cos| with the topic direction <= 2.2e-7 at every point, max |shuf - topic| element .23-.27 (they are different vectors, not copies). Test AUCs of the orthogonalised controls .45-.56 at every layer (raw, before orthogonalisation: up to .916 for seed 0, cos up to .68, see `directions_490_unorthogonalised/validity.json`), so the change does what README §2 now declares.
- AUCs (`directions.py:120-127, 132-135`): `roc_auc_score(prot[test], proj[test, l])` per point on the odd carriers; also fit-set, attack-only test, non-attack test. Term-held-out (`directions.py:138-139`): fit on `fit & half` (first 8 terms of each group, `topic_bank.py:75`), test on `test & ~half` (other 8 + 8 terms on odd carriers). All-carrier direction and per-layer cosine as declared. D1 = body best-layer test AUC >= .95: `.9978` at point 0, per-layer minimum `.938`, PASS.
- Files written: `directions_body.pt`, `directions_last.pt`, `directions_body_all_carriers.pt`, `directions_shuf0..4.pt`, `bank.json`, `PROVENANCE.md` under `data/topic_direction/qwen3vl8b/`; `run.log` (first line host), `run.pid`, `config.json`, `validity.json` under `runs/20261017_m1_topic_removal/directions/`. Matches README §2 and §3.
- Inputs: only `topic_bank.py`; no manifest, ASR, frame or GT file is opened.
- Text-only forward with no cache (`directions.py:96`): `can_compute_mrope` is false and `past_key_values_length == 0`, so `Qwen3VLModel` neither reads nor writes `rope_deltas` (`modeling_qwen3_vl.py:1118-1147`) and the text model uses plain `arange` positions.

### 2. `measure.py` Ablator and read path

- Hook (`measure.py:45-56`): inactive while `dirs is None` (returns `None`, output untouched); otherwise `h2 = h - (h.u) u` over all positions of the forward, returned as a tensor (or re-wrapped tuple), which replaces the layer output; `Qwen3VLTextModel` assigns `hidden_states = layer_outputs` (`modeling_qwen3_vl.py:837`), so the ablated tensor feeds the next layer and, for the last layer, the final norm and the Yes/No read.
- Index (`measure.py:41-43`): `points = [embed_tokens] + layers`, hook `i` uses `dirs[i]`; identical ordering to `Recorder.points`, so embedding = row 0, layer i = row i+1.
- Activation window (`measure.py:60-73, 128-146`): `abl.dirs` is set only inside `read()` after `branch_ids` (tokenisation only) and cleared in `finally`; `assert abl.dirs is None` before `build()`; the replay `margin()` runs with `dirs None`. Prefix, verdict, stance turn, replay and the stored visual reads are therefore un-ablated.
- Cache and rope (`measure.py:62-72` vs `src/stance_cache.py:32-36`): same sequence (`n = get_seq_length()`, set `rope_deltas` from `ctx["rope"]`, `branch_ids` with the stance history and head, `cached_margin(in_place=True)`, `crop(n)`, restore `rope_deltas`).
- Random directions (`measure.py:76-79`): Gaussian rows normalised per row, seed 0.
- Order: `shuf = [load(SHUF(k)) for k in range(5)]`, `z_shuf = [read(... S) for S in shuf]` (`measure.py:111-115, 143`), so `z_shuf[k]` is seed k; `analyze.arm_score` reads `t["z_shuf"][k]` for arm `shufk`.
- Question tokens: `yesno_question(i, n, t1, t2, window_text(segments, t1, t2), "speech")` with `load_asr(ds)` defaults and `frame_paths(ds, vid, 20)` (`measure.py:103, 127, 136-140`); r6 (`til_measure.py:55-59, 97-98`) used `window_text` on the same `load_asr` output, 20 `k20` frames and the same `yesno_question`. The per-window assertion `measure.py:137` ties speech presence to r6, and the replay on the first two videos per corpus (`replay_max_abs`) is the content check.

### 3. Branch K/V written under ablation

During the ablated forward each layer's `DynamicLayer.update` concatenates the branch tokens' K/V (computed from ablated hidden states) onto the prefix K/V. `cache.crop(n)` with the positive legacy argument crops every layer back to the absolute length n (`cache_utils.py:165-188`, `Cache.crop` loops over all layers `:1602-1610`, `DynamicCache.crop` forwards to it `:2051-2056`), so the ablated entries are removed before the next read; prefix entries are never recomputed. The branch forward only reads `rope_deltas` (`modeling_qwen3_vl.py:1131-1145`, `past_key_values_length > 0`); the restore at `measure.py:72` is a safeguard identical to `stance_cache.margin`. No other model state is touched by the hooks (`proj` is a Python list on the Ablator).

### 4. `analyze.py`

- D2 (`analyze.py:76, 84-86, 92`): pooled window AUC over all GT videos' speech windows with a clean label; `abl >= z_speech - .05`. D3 (`analyze.py:77-83, 93-94`): rows from within-eligible videos only; macro AUC per video over videos with both window labels; `abl >= z_speech + .02` and `mean(shuf0..4) <= abl - .02`; `rand_mean` reported only. One row set per corpus carries every score (`window_rows` writes `z_speech`, `abl`, `shuf0..4`, `rand0` into each row), so all scores are on the same windows.
- D1 is read from `validity.json["D1"]["PASS"]` (`analyze.py:73, 109`); decision = D1 and D2 and D3 on both corpora.
- Predictions (`analyze.py:136-153`): per window `z = w6["z"]` from r6 (key term unchanged), `z_visual = t["z_visual_r6"]` asserted equal to r6's, `start/end` asserted equal to r6's grid, `z_speech` = the arm's read exactly where r6 had speech, `extra.z_video` = r6's. Window keys starting with `z_` are still only `z_visual`, `z_speech` (`twolevel_r2.set_modalities`), `intercept = z_video + mean(z)` (`twolevel.py:360-361`) is therefore r6's key. `score_curve` = per-window `max(z_visual, z_speech_arm)` (speech) or `z_visual` (silent) painted at frame centres `(f + .5) / 4 // 8`, length `ceil(4 * duration)`, the same expansion as r6 (`til_measure.py:112-114`) and candidate 45. The only design change against the reviewed candidate-45 code is the removal of the affine map, as stated in the brief.
- Decode flags (`analyze.py:27-28`) equal r6_bma's `config.json` (`noleak`, `nscore`, `key calib`, `bma`, `bma_prior length`, `min_windows 2`, `bma_grid 6`, `arm m2`); evaluator called as `src.eval.evaluate_four_datasets --predictions ... --gt-dir data/gt_4fps --datasets ... --out ...` with the unified evaluator; `report()` reads `per_dataset[*].{frame_ROC_AUC, frame_PR_AUC, within_video_macro_ROC_AUC}`, which is the layout of `r6_bma/metrics.json` (.8971/.6942/.7508, .7168/.6711/.6373).
- GT: `gold()` is called only in `precheck()`; `predictions()` reads records and r6 only; `decode()` reaches GT only through the evaluator and twolevel's own metric call, as for r6_bma.

### 5. `src/window_level.py`

`frame_span` = `int(round(4.0 t))` clipped to the GT length, identical to candidate 44 `frames_of` and candidate 45 `frame_span` (`archive/.../common.py:113-116`); `window_label` = 1 if share >= .5, 0 if share == 0, else dropped, identical to candidate 45's `window_table` filter; `within_eligible` = `0 < sum < len`, identical; `gold` reads the test split of `data/gt_4fps/<ds>.npz` as before.

### 6. Launchers

Both: `--partition=local-sc474397`, `--gres=gpu:1`, 4 CPU / 32G, `--output=runs/20261017_m1_topic_removal/slurm_*%j.out`, offline HF cache under the repo `.cache/`, Triton / CUDA caches under the repo. Outputs of the three stages go to `runs/20261017_m1_topic_removal/` and `data/topic_direction/qwen3vl8b/` only. `${SMOKE:+--smoke --name main_smoke}` is safe under `set -u`.
