# Independent code review (rule 6): candidate 43, audio-bound speech evidence (2026-10-14)

Commit reviewed: b461de6 (branch follow-up, working tree clean at review time). Reviewed on sc474397 (uoa-lab1), CPU
only; the Qwen3-Omni weights were not loaded and no GPU code or Slurm job was run. Small CPU checks were run against
the caches with `~/miniconda3/envs/HateVideo/bin/python` and, for the processor and the transformers 5.15.1 source,
with `.cache/envs/HateVLM/bin/python` (the job environment's versions: transformers 5.15.1, torch 2.11.0+cu128).

Scope (rule 6 only): does the declared mechanism reach the final scores, is any label or GT array read in the scoring
path, feature / timestamp / 8 s window / 4 fps grid / video_id alignment, do the caches correspond to the current
inputs, is evaluation through the sole evaluator and the fixed r6_bma time level. Style, robustness, refactoring and
theory were not reviewed.

Files read: `experiments/20261014_m1_audio_bound/{README.md,omni_judge.py,measure_audio.py,analyze.py,extract_audio.py,
align_words.py}`, `launch/{campus_run,lab1_align}.sbatch`; shared code `src/mllm_judge.py`, `src/video_inputs.py`,
`src/eval/evaluate_four_datasets.py` (CLI), `src/eval/evaluate.py` (curve handling);
`experiments/20260926_twolevel/{twolevel_r2,twolevel}.py` (what the decoder reads from a predictions file);
`experiments/20261007_m1_streamingtom/{placement_controls,inputs}.py` (the `adjacent_local` rule the README cites);
`experiments/20260922_til/til_measure.py` (the r6 loader); the r6 reference `runs/20260926_glr/base_gridA/
{predictions.jsonl,config.json}` and `runs/20260926_twolevel/r6_bma/{config.json,metrics.json}`; the caches
`data/audio16k/` (+ `PROVENANCE.md`, `manifest_test.jsonl`), `data/asr_whisper_aligned/<ds>/words.jsonl` (+
`PROVENANCE.md`), `data/asr_whisper_large_v3/`, `data/frames_k20/`, `data/temporal_retrieved_kv/`; the transformers
5.15.1 sources `models/qwen3_omni_moe/{modeling,processing}_qwen3_omni_moe.py`, `cache_utils.py`, `masking_utils.py`.

## Verdict

One blocker: the forward-count assertion in `measure_audio.read` is wrong by one read per window, so the measurement
stops with an `AssertionError` on the first video of any run (smoke or main) before a single record is written. It is
a crash, not a silent wrong number; the fix is one term in one line. Everything else in the scoring path (window
geometry, clip bounds, aligned-word slices, derangement, per-arm `z`, the r6 schema the decoder reads, the KV-cache
branching, the absence of any label read, the evaluator and decoder calls, the gate logic) was found correct by
reading and by the CPU checks listed at the end.

## BLOCKERS

1. BLOCKER `measure_audio.py:190-191`. The expected number of thinker forwards per video is computed as
   `1 + 2 + n * 3 + (#visual_local) + (#speech_prop) + (#speech_aligned)`, but `read()` runs four unconditional
   forwards per window (`visual_native` text read, line 168, plus the three audio reads, lines 181-184), not three.
   The `visual_native` read is missing from the formula, so `actual_forwards == expected` fails by exactly `n` on every
   video with at least one window. Evidence: a CPU mock of the judge that counts calls exactly as the module hooks do
   (`OmniJudge.__init__` lines 55-58: one pre-hook call on `model.model` per forward; the transformers source confirms
   `Qwen3OmniMoeThinkerForConditionalGeneration.forward` calls `self.model(...)` once, modeling line 2159, and that
   `self.model` is the text model, line 1864-1868) runs the unchanged `read()` on hate_video_1 (12 windows) and stops
   with `AssertionError: (73, 61)`. Since the assertion sits before any record is written (`read()` is called from
   `main()` line 279 and the `.partial` file is written only on return), the smoke run and the main runs would both
   stop at the first video. Fix: count the `visual_native` read (`n * 4`, or `+ n`). The two companion assertions on
   line 192 (`actual_audio == 3 * n`, `actual_vision == 1 + #visual_local`) are correct. With only that term patched,
   the rest of `read()` ran to completion in the mock on three videos (see CPU checks) and produced records and
   predictions that satisfy the decoder's schema.

## ADVISORIES (non-blocking)

1. README §2 says `speech_prop` / `speech_aligned` use the r6 wording "with `(no speech)` when the slice is empty". The
   code does not read an empty slice: `measure_audio.py:175-180` records `None` and the arm's window carries no
   `z_speech`, exactly as r6 (the r6 reference has `z_speech` in a window if and only if `window_text` of the same
   loader is non-empty: 7359 / 7359 windows agree). The code is the behaviour the brief, README §6 check 2 and the
   decoder's missing-modality handling require; the README sentence should be corrected so the record and the code
   agree. Also README §2 "12.5 tokens / s: at most 100 tokens per clip" is stale: the processor gives 104 audio tokens
   for an 8 s clip (13 / s), as §10 already states (CPU check with the cached processor).
2. Audio track shorter than the manifest duration. `data/audio16k/manifest_test.jsonl` shows 10 videos whose wav is
   more than 0.1 s shorter than the manifest / frame-cache duration (HateMM 4, HateClipSeg 6); the largest are
   bit_7EOOUGa9y9h4 (180.65 s of audio for 187.9 s: window 23 at 184-187.9 s has no audio and gets the 400-sample
   zero pad = one audio token of silence; window 22 has 4.65 s), non_hate_video_624 (1.13 s), hate_video_215 (0.76 s),
   bit_awahfpJCG4c9 (0.34 s). The code handles it as README §2 says ("windows whose track is silent still get their
   clip") and only the last window of those videos is affected, so no change to the scoring path is needed; record the
   list in the run README. Consequence for README §6 check 3 ("sample count = (end - start) x 16000 +- 1"): it fails
   literally on the smoke set itself, because bit_0EHvMSiEHVoc's wav is 911 samples (57 ms) shorter than
   `round(duration x 16000)` (last clip 44715 vs 45626 samples). `analyze.check` does not implement check 3 (the trace
   stores `audio_samples`); state the rule as "window clipped to the track end" and check it from the traces.
3. `omni_judge.py:223,233` restore the cache with `cache.crop(n)` where `n` is the absolute prefix length. In
   transformers 5.15.1 a positive argument is the deprecated "absolute size" path (`cache_utils.py:165-182`,
   `warning_once` says it is removed in 5.18); under the pinned version it restores the cache to exactly `n` tokens, so
   the branch isolation is correct. A newer transformers would silently change the meaning (remove `n` tokens) and
   every later branch would see leaked state. The job environment is pinned (README §10; `config.json` records the
   version), so this is advisory; a one-line `assert cache.get_seq_length() == n` after each crop would make check 4
   self-verifying in the run.
4. README §6 check 1 promises "the token ids agree (the seam check of r6)". `measure_audio.py:207-214` only checks that
   the standalone ids of the `speech_audio` probe start with the prefix ids; it does not check `prefix + stance +
   branch ids == standalone ids` for any branch. The numeric check (|standalone - cached| <= 1e-3, recorded per branch
   kind) is implemented and is the decisive one; the full concatenation check is cheap to add (`branch_ids` /
   `encode` already give the suffix ids) and would make the record match the README.
5. `analyze.py:124-125` `no_loss` uses a strict `> -floor` where README §3 says "not below r6 by more than the floor"
   (`>= -floor`). Only an exact tie at the floor would differ; state the reading used.
6. Launch precondition: `#SBATCH --output=runs/20261014_m1_audio_bound/slurm_%j.out` requires the directory to exist
   on the campus machine before `sbatch` (the `mkdir -p` inside the script runs after Slurm opens the output file).
7. Diagnostic for reading the `text_aligned` control: the aligned slice differs from the proportional slice in 92 %
   (HateMM) / 94 % (HateClipSeg) of windows by word content, and also by presence: 216 / 325 windows have proportional
   text but no aligned word, and 209 / 322 the reverse. The `text_aligned` arm therefore changes which windows carry a
   speech read, not only the words. `analyze.check` reports `aligned_differs_share` (content) but not the presence
   change; both are worth reporting. Related input property: a Whisper caption hallucination on a non-speech track is
   spread by the forced aligner over the whole track (hate_video_1: six words, last end 94.976 s = track end), so such
   words appear in every window of `text_aligned` but only in the first window of `text_prop`. Not a code defect; the
   aligner output is what the control tests.

## Checks against the brief

1. Window geometry. Windows come from `src.video_inputs.fixed_windows(duration, 8.0)` in `read()` (line 156) and
   `prediction()` (line 131); for all 333 test videos they equal the r6 windows (`extra.windows` start / end / i) and the
   duration equals the manifest and the r6 record (CPU check: 0 differences over 7359 windows). The audio clip is
   `wav[round(t1 x 16000):round(t2 x 16000)]` (`clip`, lines 105-110), i.e. [8i, 8i+8) s: 128000 samples for every full
   window, last window `round(t2 x SR) - round(t1 x SR)` (hate_video_1: 111616, matches); the wav is read with
   `scipy.io.wavfile`, asserted 16 kHz mono int16 (lines 99-102), length `round(duration x SR)` for hate_video_1
   (1519616 = 1519616). The aligned slice is the words with `t1 <= (s + e) / 2 < t2` on the same window edges (line
   82); every word of a video falls in exactly one window (checked). The proportional slice is `window_text(segments,
   t1, t2)` with `load_asr(ds)` at its default `fill_untimed=True` (lines 162, 255), the same functions and cache the r6
   run used (`til_measure.py:32,55,176`): the slice is non-empty exactly where r6 has `z_speech` in 7359 / 7359 windows,
   and 316 windows would disagree under the pre-2026-09-26 loader, so the check discriminates. The strings are produced
   by the same function on the same file, hence byte-identical. The local frames are the `selected_indices` of
   `data/temporal_retrieved_kv/<ds>/<vid>/metadata.json` with the half-open rule `t1 <= t < t2` (lines 61-64, 164), the
   same source and rule as `placement_controls.local_ids` (`inputs.py:37-38`); the cache has `source_fps 0.5`, duration
   equal to the manifest and all frame files present for 333 / 333 videos, every frame in exactly one window, 7 windows
   without a local frame; those fall back to the native read (line 174) exactly as `placement_controls.py:152`. The
   label format `[t=..s]\n` equals the placement `NATIVE_LABEL`. The score curve index rule (frame centre // 8) is the
   one the r6 file obeys (r6 `score_curve` recomputed from its windows: 0 differences in 333 videos) and gives
   `ceil(4 x duration)` frames (r6 length equal for all 333).
2. Derangement. `derangement(n, video_index)` draws `numpy.random.default_rng([1, video_index]).permutation(n)` until
   no fixed point (lines 85-92); `video_index` is the row's position in the 333-row manifest over both corpora (lines
   253-254, 279; hate_video_114 = 55). Checked n = 1..59 for five indices: a permutation, no fixed point for n >= 2,
   reproducible, different across indices; n = 1 keeps its own clip. `speech_swap` pairs window i with the clip
   `wins[swap[i]]` of the same video's wav (lines 181-183) and records `source_window`; `speech_nbr` uses i+1 and n-2
   for the last window (`neighbour`, lines 95-96, checked n = 1..59). The per-video `derangement_ok` flag is recomputed
   from `swap` (line 186) and `analyze.check` requires it for every video.
3. Arms and the z the decoder reads. `ARMS` (lines 38-40) is the README table plus the declared `audio_nbr`;
   `r6_inputs` = (`visual_native`, `speech_prop`). `prediction()` writes per window `i, start, end, z_visual, z` and
   `z_speech` only when the speech branch was read, with `z = max(v, s)` else `v` (lines 131-135); the r6 file obeys the
   same rule in every window (checked). The top level carries `duration`, `score_curve`, `error=None`, `extra.z_video`
   (the verdict margin from `build`) and `extra.windows`, which is all `twolevel.load_run` (skips `error`), `prep`
   (`duration`, `extra.windows[*].start/end/z/z_visual/z_speech`, `extra.z_video`, `len(score_curve)`) and
   `set_modalities` (keys starting with `z_`, here exactly {z_visual, z_speech}) use. A CPU mock run of `read()` (with
   only the blocker's term patched) on hate_video_1, bit_0EHvMSiEHVoc and non_hate_video_229 (an alignment-failed video)
   produced records for all six arms that pass `twolevel.prep` (cells = ceil(duration / 4), L = r6 length, zv = the
   verdict margin), have `z_speech` exactly where `speech_prop` was read for the text arms and in every window for the
   audio arms, and whose `visual_local` falls back to `visual_native` where no frame lies in the window. The decoder
   call in `analyze.evaluate` (lines 105-107) carries `--noleak --transform nscore --key calib --duration bma
   --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`; every other flag is at its default, and the defaults equal
   `runs/20260926_twolevel/r6_bma/config.json` (k from min-windows = 4, d_gap / d_hate 80, fusion or, vlevel joint,
   vtemper none, kappa 1, kind two, evidence em, center false).
4. KV-cache branching. `branch_text_margin` / `branch_media_margin` take `n = cache.get_seq_length()` before the step
   and `cache.crop(n)` in `finally` (`omni_judge.py:216-233`); `Cache.crop` dispatches to every `DynamicLayer.crop`,
   whose positive-argument path restores the absolute length `n` (`cache_utils.py:165-182, 1602-1610`). The verdict
   question and the stance answer are run in place on the cache before any branch (`build`, lines 268-276, with the
   length asserted on line 276), `ctx.history` holds both turns and `ctx.head` the rendered string through the stance;
   every branch is rendered on `msgs + history + [branch turn]` and the result is asserted to extend `head` (lines
   244-249, 292-294). With the cached Omni processor the prefix, verdict, stance and branch strings extend each other as
   required, two consecutive user turns render, and the `{"type": "audio"}` / `{"type": "image"}` items render as
   `<|audio_start|><|audio_pad|><|audio_end|>` / `<|vision_start|><|image_pad|><|vision_end|>`; the prefix string is
   byte-identical to the Qwen3-VL template's rendering of the same messages (2535 characters), which is README §6
   check 2's second half. Rope positions: the prefix forward passes the processor's attention mask and no
   `position_ids`, so the thinker computes `get_rope_index` itself and stores `rope_deltas` (modeling 2136-2150);
   `prefix_cache` recomputes it standalone with the same positional argument order (`input_ids, image_grid_thw,
   video_grid_thw, attention_mask, use_audio_in_video, audio_seqlens, second_per_grids`, modeling 225-233; the
   `audio_seqlens` argument is `feature_attention_mask.sum(-1)`, exactly what the forward derives on 2131-2134) and
   asserts equality; the logical start is `max(position) + 1` (lines 199-202), and advances by the token count of each
   in-place text step (lines 205-214, 270-274). For a branch, `text_step` uses `arange(n) + logical_start` as a (3, 1, n)
   tensor and `branch_media_margin` uses `get_rope_index` on the suffix alone plus `logical_start` (lines 225-231);
   with explicit `position_ids` the thinker skips its own computation (modeling 2136) and the text model takes a 3-row
   tensor as the mrope positions (1682-1693), building the causal mask from the cache length (`create_causal_mask`
   with no text positions; the sdpa path drops the mask only when q_len == 1 or kv_len == q_len, `masking_utils.py:268`,
   so a multi-token suffix on a non-empty cache gets an explicit causal mask). `get_rope_index` is translation
   invariant per segment (text sequential, an image block starting at the running text index, an audio-only suffix
   taking the sequential branch at modeling 463 which coincides with the audio block rule `arange(audio_len) + st_idx`
   at 363), so suffix-relative positions plus the logical start equal the full-sequence positions. Read-out: the
   `lm_head` pre-hook receives the full hidden state (modeling 2171-2172, no `logits_to_keep` slice) and `[0, -1]` is
   the last suffix token, the end of the assistant header, in the cached path (`forward_hidden`, lines 175-186) and in
   the standalone path (`plain_margin`, lines 236-241, with `rope_deltas = None` and the model's own rope index). The
   only state that outlives a branch is the cache (restored) and `rope_deltas` (unused once positions are explicit).
   Numerical agreement cached vs standalone (|delta| <= 1e-3) is the declared GPU smoke check 1 and remains to be run;
   reading finds no defect in the path it will test.
5. GT. A grep over the experiment directory finds `gt_4fps` only as the evaluator's `--gt-dir` in `analyze.evaluate`
   (line 104); `measure_audio` imports `fixed_windows, frame_paths, load_asr, load_manifest, window_text` from
   `src.video_inputs` and not `within_defined_ids` (the one GT reader in that module); the manifest rows carry
   `dataset, duration, transcript, video_id, video_path` and no label field; `local_frames` asserts the frame cache's
   `GT_read is False`; the audio cache is decoded from the raw videos (`extract_audio.py:11-28`) and the aligned words
   from the Whisper chunks and the audio (`align_words.py:68-73, 116-132`). No label or GT array is opened anywhere
   between the inputs and the predictions; the `GT_read` flags are consistent with the code.
6. analyze.py. `FLOOR` = .005 / .005 / .01 (line 25); the performance gate is within >= +.01 on both corpora and no
   metric below the floor on either (lines 121-131); M1 = `audio - audio_swap`, M2 = `audio - text_aligned`,
   `aligned_vs_prop`, the reader swap reported with the < -.02 flag (lines 127-134); the reading chain (lines 135-142)
   follows README §3 in the declared order (M1 and M2; M1 and aligned >= prop; M1 alone; neither). The comparator is
   `runs/20260926_twolevel/r6_bma/metrics.json` (line 26), whose `per_dataset` rows carry the three metrics and
   `n_videos_defined` (HateMM .8971 / .6942 / .7508, 84 videos; HateClipSeg .7168 / .6711 / .6373, 99 videos). Raw window
   curves and decoded curves both go through `src.eval.evaluate_four_datasets` (lines 103-104 directly, and inside
   `twolevel_r2.py` line 840). `check()` guards the alignment facts the gates rest on: `GT_read`, the implementation
   tag, the derangement per video, window count equal to r6, `speech_prop` present exactly where r6 had `z_speech`
   (required to be 1.0 for PASS), and the arm predictions equal to the per-video records; `evaluate()` refuses to run
   without a passing `checks.json`.

Caches against the current inputs: `data/audio16k/` has 333 wavs (0 errors in `manifest_test.jsonl`) from the
manifest videos (HateClipSeg via the `input_video` of the frame cache, as the PROVENANCE says); `data/asr_whisper_aligned/`
has 215 + 118 records with the statuses the README reports (HateMM 208 aligned / 5 no_text / 2 failed, both CTC
length errors; HateClipSeg 118 aligned), word strings identical to the Whisper chunk words in order, starts
monotone, no `e < s`, no word outside the track; `data/frames_k20/` gives 20 frames for all 333; the r6 reference
has 333 records, no errors, 7359 windows.

## CPU checks run by the reviewer (read-only on the repository; nothing written outside this file)

- r6 reference vs `fixed_windows`: windows, durations, curve length, curve index rule, `z = max` rule, speech presence
  under `load_asr` default and under `fill_untimed=False`.
- `derangement` / `neighbour` for n = 1..59 and five video indices; `selected_rows(smoke)` equal to the streamingtom
  five; `clip` lengths on a synthetic track and on two real wavs.
- Cached `Qwen3OmniMoeProcessor` (offline): Yes / No id sets, two-user-turn rendering, audio and image placeholders,
  audio tokens for 8 / 3 / 0.025 / 6.976 s clips (104 / 39 / 1 / 91), image tokens and pixel cap for a real frame
  (91 tokens, 93184 pixels), prefix text equality with the Qwen3-VL template, head / answer / branch string extension.
- `data/audio16k/manifest_test.jsonl` wav length vs manifest duration; `words.jsonl` integrity and the aligned vs
  proportional slices over all 7359 windows; `temporal_retrieved_kv` partition and file existence for 333 videos.
- Mock-judge run of `measure_audio.read()` unchanged (assertion `(73, 61)` on hate_video_1) and with the expected
  count patched (three videos; records through `twolevel.prep`).
- transformers 5.15.1 sources for `get_rope_index`, the thinker forward's position handling, the text model's
  position / mask handling, `Cache.crop` / `DynamicLayer.crop`, and the sdpa mask rule.

VERDICT: FAIL (1 blocker)
