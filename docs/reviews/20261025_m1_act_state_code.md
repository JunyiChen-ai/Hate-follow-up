# Code review: candidate 50 (act-state reading, mask form A1), rule 6

Date: 2026-10-25. Reviewer: independent agent (rule 6 of `RESEARCH_ITERATION_RULES.md`). Scope: only bugs that would change what the experiment observes or concludes.

Files reviewed (working tree on uoa-lab1, the experiment directory is untracked at commit `92a3b7f`):
`experiments/20261025_m1_act_state/{README.md, act_state.py, measure.py, stage0.py, stage1.py, smoke_cpu.py, launch/lab1_stage0.sbatch, launch/lab1_stage1.sbatch}`.
Infrastructure read, not edited: `src/mllm_judge.py`, `src/pre_rotary_memory.py`, `src/stance_cache.py`, `src/window_token_regions.py`, `src/video_inputs.py`, `src/eval/evaluate.py`, `src/eval/evaluate_four_datasets.py`, `experiments/20260922_til/til_measure.py` (the code that produced `runs/20260926_glr/base_gridA` and `runs/20260927_dehate_external/reads_gridA`), `experiments/20260926_twolevel/{twolevel_r2.py, twolevel.py}` (load, modalities, metrics call), `archive/experiments/20261002_m1_grounder/{grounder.py, measure.py}`, `runs/20260926_twolevel/final_rawkey/config.json`, and the installed `transformers 5.15.1` sources in `.cache/envs/HateVLM` (`models/qwen3_vl/modeling_qwen3_vl.py`, `cache_utils.py`, `integrations/sdpa_attention.py`).
Protocol: README §3 compared with `review_m1.md` lines 309–367 by `diff`. They are identical.

## VERDICT: PASS

No blocking bugs found.

### Blocking

None.

### Non-blocking notes

1. **Provenance.** The experiment directory is not in git yet (`git status`: `?? experiments/20261025_m1_act_state/`). The run `config.json` (`measure.py:342`) records the code paths, the date and the implementation tag `c50-a1-v1`, but no commit. Rule 7 and CLAUDE.md ask for a commit in the code-version note, and for sbatch files to be in git. Commit before the stage-0 submission and keep that commit in the run record. This changes no number.
2. **Re-rotation tolerance.** `KEY_ROT_REL_TOL = 1e-2` (`measure.py:61`, checked at `:189-192` on every video) allows a key error of up to 1 % of max |k|, while the expected value is 0. The same `rotate` is applied to the donor keys of A1_other (`:200`). If any video shows `own_span_key_rel_diff > 0` in `checks.jsonl`, then A1_other differs from A1 by that rotation error as well as by the donor content. The self-swap check (1e-3 logit, first video per corpus, `:234-240`) bounds the effect on the reads. When gate (d) is reported, also report the largest observed `own_span_key_rel_diff`. No change is needed if it is 0.
3. **README §7 wording.** The table lists "explicit-mask base vs the ordinary SDPA branch" on the first video per corpus. `measure.py:229` runs this check only when the `base` arm is in the run, i.e. stage 1. Stage 0 and stage 2 have no `base` arm. There, the masked path is covered by:
   - the `all_local` replica (HateMM, HCS);
   - crop/copy;
   - self-swap.

   DeHate has no external replica of the masked path. This is wording only: the plumbing code is the same for all corpora.
4. **Stage-0 choices.** S1–S3 use the visual change only, and S3 uses Pearson (`stage0.py:127-131`). The protocol text in §3 does not name a modality. The choice is declared in README §5 before any run, so it is a design choice, not a code error.

## What was checked and found correct

### 1. The A1 span [P, P+32) (brief item 1)

**What the span holds.** It is the tokens that `verdict_state` (`measure.py:125-150`) appends after the prefix:
- `b0` = `Judge.branch_ids(msgs, VIDEO_QUESTION)`;
- `a0` = `Judge.answer_ids(msgs, VIDEO_QUESTION, stance)`.

`measure.py:145` asserts `cache.get_seq_length() == P + len(b0) + len(a0)`, and `:185` asserts `ncache == P + len(span_ids)`.

**Independent CPU tokenizer check** (Qwen3-VL processor, offline). For both stances:
- `b0` = 29 tokens, `<|im_start|>user\nDoes this video contain content that violates any of the above rules?\n\nAnswer "Yes" or "No".<|im_end|>\n<|im_start|>assistant\n`;
- `a0` = 3 tokens, `Yes|<|im_end|>|\n` or `No|<|im_end|>|\n`. `Yes` and `No` are single tokens, so the length is 32 for both stances.

**Why the span does not depend on the video.**
- The prefix string always ends in `<|im_end|>\n` (`render(msgs, add_generation_prompt=False)`), so the span text is the same for every video.
- `answer_ids` took the strict seam path, not the loose-seam fallback.

**No window or video content.** The span holds only the verdict question turn, the assistant header and the one-word stance turn. The stance is chosen by `z_video > 0`, as in r6.

**Mask columns.** `window_keep` (`act_state.py:43-50`) marks `[P, cached_len)` from the actual cache length, so the visible extra span is exactly `b0 + a0` for every video by construction.

**Re-run of smoke part 2** (4 videos per corpus, all three corpora):
- P equals the r6 `prefix_tokens`;
- the span is 32 tokens at plain text positions;
- the scaffold is 290 tokens;
- the span tokens are identical across videos of the same stance.

### 2. Mask schedule (brief item 2)

**Where the mask is applied.**
- `QueryAccess` (`act_state.py:53-112`) registers forward pre-hooks on the 36 `language_model.layers[i].self_attn` modules.
- Each hook is a no-op while `self.active is None`.
- `active` is set only inside `restrict()`, which only `margin()` uses, i.e. only for window branches.
- `prefix_cache`, the verdict forward, the stance forward, the donor passes and the `plain` check (`measure.py:230`) all run with `active = None`.

**Every layer, once.** `restrict()` asserts `visited == [0..35]` after each masked forward, so each layer received the mask exactly once.

**The hook reaches the attention call.**
- The decoder layer passes `attention_mask` as a keyword (`modeling_qwen3_vl.py:550-558`), so the keyword path of the hook replaces it.
- SDPA with an explicit mask sets `is_causal = False` (`sdpa_attention.py`).

**Mask tensors.** They are Grounder's: bf16 additive mask, `finfo.min`, shape (1, 1, q, n + q), causal within the query.

**Allowed set.** `keep[:P] = scaffolding | local[i]` for `all_local`, A1 and A1_other alike. `keep[P:]` is False for `all_local` and True for A1 / A1_other. So A1 = `all_local` + the span, exactly. Re-run of smoke part 2: the two keep vectors differ exactly on `[P, P+32)`. Part 3: the injected mask equals the keep vector at all 36 layers, by keyword and by position.

### 3. A1_other (brief item 3)

**Donor rule.**
- `donor_map` (`act_state.py:185-194`) gives the cyclic next id in `sorted()` order within the same sign group.
- The sign is `z_video > 0` from the r6 reads, over the whole test split of that corpus. Corpora are separated by `load_manifest(..., [ds])`, and the r6 reads are filtered by dataset.
- A group of one video stops the job (`measure.py:101`).
- At run time:
  - span tokens and stance are re-checked equal (`:196-199`);
  - the target's z_video must equal r6 exactly (`:176-178`);
  - so must the donor's (`:156-157` for donor passes; main-pass donors went through `:176`).

**Capture.**
- `pre_rotary_memory.observe` hooks `k_norm` (pre-rotary key) and `v_proj` (value), one forward each.
- `index not in keys` asserts that a single forward is captured.
- This matches `Qwen3VLTextAttention.forward` (`modeling_qwen3_vl.py:481-488`): `k_norm(k_proj(h))` → `apply_rotary_pos_emb` → `cache.update`; the value is `v_proj(h).view().transpose()`.
- `join_forwards` concatenates the verdict forward (29) and the stance forward (3). `:146` asserts 32.

**Positions.**
- `span_positions` (`act_state.py:137-153`) takes the target's `get_rope_index` positions (`stance_cache.positions`).
- It asserts that they equal `arange(P, P+32) + delta` on all three axes, and that `delta` equals the live `rope_deltas`.
- This is the rule the model uses for cached forwards (`compute_3d_position_ids`, `modeling_qwen3_vl.py:1131-1141`: cache index + `rope_deltas`, no `attention_mask` passed by `_step`).

**No stale rope_deltas.**
- `get_rope_index` is pure: it does not write `self.rope_deltas`.
- A donor pass sets the donor's `rope_deltas`, but the target's own prefill recomputes it (`past_key_values_length == 0`, line 1118) before any target branch. Stale deltas cannot reach the target branches.

**Rotation.**
- `rotate` uses the model's own `rotary_emb` and the same bf16 formula as `apply_rotary_pos_emb`. There is no `kernels` package in the env, so neither RoPE nor RMSNorm is replaced by hub kernels.
- The donor keys are captured before rotation, so the donor's own positions never enter.
- The own-state re-rotation is checked on every video (`measure.py:186-192`: keys rel ≤ 1e-2, values exact).
- A1 on a self-swapped cache is checked on the first video (1e-3).

**Swap.**
- `swapped_cache` deep-copies the target cache right after `verdict_state`, before any branch, and overwrites `[P:]`, i.e. all 32 span positions, at all 36 layers. Keys are re-rotated, values copied.
- Everything before P is the target's own. The branch questions, `history`, `head` and positions are the target's own too.
- None of the target's own span state remains in the A1_other cache. The source cache is unchanged: the deep copy has its own storage (smoke part 3).

**Store logic.**
- Descending order makes each donor's main pass come before its user's read. Donor passes happen at the cyclic wrap and at the `--first` boundary.
- `plan()` simulates the same store as `main()`. `assert not store` at the end of each corpus.
- Re-run of smoke part 1, donor passes:
  - stage 0: 1 / 2 / 2;
  - stage 1: 2 / 2;
  - stage 2: none.
- The README states the same counts.

### 4. Replica checks (brief item 4)

**Base.** The base replica is compared on every video with `BASE_TOL = 0.0`. The `z_video` check is exact too.

**`all_local`.** Compared on every HateMM / HCS video with tolerance 1e-3. Grounder `all_local` covers 333 / 333, so the check is never silently skipped.

**Branch sets.** Branch-set equality with the reference is asserted.

**Failure path.** Each check raises `CheckFailed` (a `SystemExit` subclass, exit status 1) before that video's records are written. In the sbatch, `set -e` and the ERR trap then print `C50_* FAILED`, and `grep -q RUN_DONE` guards the end.

**Exact equality is attainable on the compute host.**
- Grounder `r1_full/base` vs `base_gridA`: 333 / 333 videos, max |dz| = 0, z_video identical. That run used the same explicit-mask, in-place + crop plumbing, on sc474399; the r6 reads came from sc448960.
- Candidate 46's lab1 records (`runs/20261017_m1_topic_removal/main/records`, host sc474397, same `.cache/envs/HateVLM`): z_video equal to r6 on 333 / 333, branch replays 0.0.

**JSON floats.** Floats round-trip through JSON exactly, so an exact compare is valid.

### 5. Output format and decoding (brief item 5)

**Record format.** Records follow the r6 schema:
- `extra.windows[i]` has `start`, `end`, `z = max(z_visual, z_speech)`, `z_visual`, `z_speech`, plus token-count fields, none of which start with `z_`. So `twolevel_r2.set_modalities` sees exactly (`z_visual`, `z_speech`).
- `extra.z_video` is the target's own verdict.

**Grid.**
- `score_curve` is painted at frame centres `(f + .5) / 4 // 8`, with length `ceil(4 · duration)`, as in `til_measure.py:110-114`.
- The windows are asserted equal to r6's on every video (`measure.py:244`).

**Decoder flags.** `DECODER_FLAGS` reproduce `final_rawkey/config.json`: `noleak`, `nscore`, `raw`, `m2`, `bma`, `length`, `min_windows 2.0`, `bma_grid 6`. They are checked key by key against that file (`stage1.py:227-229`) and against every decoded arm's `config.json` (`:272-276`).

**Metrics.**
- The decoder scores its own output with `src/eval/evaluate_four_datasets.py`.
- Raw metrics use the same evaluator.
- Per-video within values come from `src/eval/evaluate.within_video_macro`, and their mean is asserted to equal the corpus number (`:287-289`).

### 6. GT and gates (brief item 6)

**No GT on the scoring path.**
- `measure.py`, `act_state.py` and `stage0.py` open no GT, label or npz file. Their inputs are the manifest, ASR, k20 frames and model reads.
- `stage1.py` reads `data/gt_4fps` only in `gold()`, for per-video within (evaluation), and through the evaluator.
- `reconstruct` and the decoder inputs are GT-free.

**Stage-0 statistics (`stage0.py:60-72`, `128-131`).**
- S1: median population sd of the visual change, ≥ .5.
- S2: median n·mean² / Σd², ≤ .8; a video with zero change everywhere counts as 1.
- S3: median Pearson correlation of d and d_o, ≤ .9.
- Eligibility: videos with ≥ 2 windows for S1 / S2, and ≥ 3 windows with non-zero sd of both changes for S3.
- Stop if any statistic fails on any corpus.

**Stage-1 gates (`stage1.py:300-309`).**
- (a) uses `.7458 / .6323`, (b) uses `.8905 / .6833` and `.7087 / .6616`, and (c), (d), (e) use a gain of `.01`. All three come from the decoded `metrics.json`.
- (d) also needs the bootstrap CI lower bound > 0.
- (f) uses the raw A1 within against `.670 / .600`.

**Bootstrap.** Paired video bootstrap over within-eligible videos, 10,000 resamples, seed 0, 95 % percentile interval of the mean difference (`:171-184`).

**Verdicts.**
- Kill = (a) or (d) fails on either corpus (`:332`).
- PASS = all gates hold on both corpora.
- Stage 2: floors `.6331`, `.6944 / .1518`.

**A1_recon.** The README rule: per video and modality, `(x − mean_al) / sd_al · sd_A1 + mean_A1`, with population sd. Offset only when there are < 2 windows or `sd_al = 0`. Then z = max over modalities, and the curve is repainted.

### 7. Launchers and environment (brief item 7)

**Slurm headers.** Both sbatch files set:
- `--partition=local-sc474397`;
- `--gres=gpu:1`;
- `--cpus-per-task=4`, `--mem=32G`;
- `--output=runs/20261025_m1_act_state/slurm_%j.out` (the README's `mkdir -p` comes first).

**Outputs.** All outputs are under `runs/20261025_m1_act_state/<run>/`. `run.log` starts with the hostname (`measure.py:317-318`, `stage1.py:211`).

**Markers.** `C50_STAGE0_DONE` / `C50_READS_DONE` on success, and `C50_* FAILED` from the ERR trap.

**Environment.**
- `PATH` points to `.cache/envs/HateVLM`: torch 2.11.0+cu128 and transformers 5.15.1, the versions in the `base_gridA` and `reads_gridA` configs.
- scipy 1.18.1 is present for `stage0.py` inside the job.
- The model weights are in `.cache/hf`, used offline.

**No stale outputs.** `runs/20261025_m1_act_state/` does not exist yet, so no stale resume is possible.

### 8. Hash rule (brief item 8)

`grep -i 'hash|sha|md5|digest|checksum|crc'` over the candidate files and the three shared modules it imports finds only docstrings that say none are used. Provenance is recorded by:
- path;
- date;
- implementation tag;
- host;
- torch / transformers versions.

See note 1 for the missing commit.

### CPU checks run for this review

- Tokenizer check of the span: the script `span_tok.py` is in the review scratchpad.
- `smoke_cpu.py --parts 1 2 3 --videos 4`, with HateVLM python and offline HF. Exit 0, `SMOKE_CPU_DONE`.
- Grounder `base` vs `base_gridA` comparison.
- lab1 candidate-46 record comparison.
- Environment and package checks.

No GPU job and no Slurm submission were made. No file other than this record was written in the repository.
