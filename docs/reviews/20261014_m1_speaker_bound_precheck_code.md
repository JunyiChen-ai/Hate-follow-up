# Independent code review (rule 6): candidate 44 pre-check, speaker-bound reading (2026-10-14)

Commit reviewed: 36dd40a (branch follow-up, working tree clean at review time). Reviewed on sc474397 (uoa-lab1), CPU
only, with `~/miniconda3/envs/HateVideo/bin/python`; the diarization venv `.cache/envs/diar/bin/python` was used only
to inspect pyannote / lightning signatures and source. No GPU code was run and no Slurm job was submitted. Nothing was
written under `runs/` or `data/`; the pre-check was re-run into the session scratchpad with the dry-run fake turns.

Scope (rule 6 only): window / frame / video_id alignment against the evaluator and the time model, whether the
pre-check's statistics implement README section 2, whether any GT enters beyond what README section 2 declares (the
pre-check reads test GT on purpose under rule 10; the method itself is unchanged), gate logic, and the diarization
input script. Style, robustness and performance were not reviewed.

Files read: `experiments/20261014_m1_speaker_bound/{README.md,precheck.py,diarize.py}`,
`launch/lab1_diarize.sbatch`; `src/eval/evaluate.py` (`within_video_macro`, `pooled`, curve truncation);
`experiments/20260926_twolevel/twolevel.py` (`to_frames`, `CELL`) and `gaussian_check.py` (the earlier window
labelling); `RESEARCH_ITERATION_RULES.md` rules 5, 6, 10. Data inspected: `runs/20260926_glr/base_gridA/
predictions.jsonl`, `runs/20260926_twolevel/r6_bma/{predictions.jsonl,metrics.json}`, `data/gt_4fps/{HateMM,
HateClipSeg}.npz`, `data/audio16k/<ds>/*.wav` (names, channels, rate), the dry-run fake turns in the scratchpad, the
real `data/speaker_turns/HateMM/turns.jsonl` (215 records, complete at review time; job 413 still writing
HateClipSeg), `runs/20261014_m1_speaker_bound/diarize/run.log` and `slurm_412.out`.

## Verdict

**PASS.** No bug was found that changes what the pre-check observes or the go / no-go decision.

An independent re-implementation of P1 and P2 written from the README text (not from `precheck.py`; kept in the
session scratchpad, not in the repository) reproduces the pre-check's dry-run numbers exactly on the fake turns:
P1 .607 / .576, P2 oracle .6551 / .6482, r6_bma .7715 / .6450, permuted oracle .6419 / .6212, all-eligible oracle
.6424 / .6171. The pre-check's all-eligible `within_r6_bma` (.75078 / .63735) equals the evaluator's
`within_video_macro_ROC_AUC` in `runs/20260926_twolevel/r6_bma/metrics.json` (84 / 99 videos) to all printed digits.

## Blockers

None.

## Checks performed, by item

1. **Window-to-frame alignment.** `frames_of` maps window `[t1, t2)` to frames `[round(4 t1), round(4 t2))` clipped
   to the GT length. Windows in `base_gridA` are `[8k, 8k+8)` with the last window ending at `duration`
   (`duration - last_end = 0` for all 333 videos; `window_offset = 0`). For integer starts the mapping is exact; the
   time model (`twolevel.py:to_frames`, frame centre `(i + .5)/4` into `CELL = 4` s cells, two cells per window) puts
   frame `i` in window `floor(i/32)`, the same frames. The earlier window labelling in `gaussian_check.py:50` uses the
   identical `round` rule. The evaluator never sees windows; it compares `y4` with `score_curve` after
   `n = min(len(y), len(s))` and after dropping non-finite frames. In the data `len(score_curve) - len(y4)` is 0 (4
   videos per corpus) or +1 (all others), so `curve[:n]` in `precheck.py:90` is exactly the evaluator's truncation,
   which is why .7508 / .6373 reproduce. It would **not** hold if a curve were shorter than the GT: the evaluator
   truncates `y` to the curve (`evaluate.py:76-77`) while the pre-check keeps all GT frames and edge-pads the curve
   (`precheck.py:90`), and the evaluator drops non-finite frames while `roc_auc_score` would raise. Neither case occurs
   in the two prediction files (checked all 333 records, no NaN in curves, no None `z`), so this is advisory A1.
   11 tail windows are shorter than one frame (min 0.018 s) and get zero frames; they cannot own a speaker (overlap
   < 1 s), contribute no frames to any curve, and only inflate `n_neg` by one in those videos. No effect on P1–P4.

2. **Speaker assignment.** `precheck.py:97-107`: overlap of every turn with the window summed per speaker, argmax,
   owner only if the summed overlap is ≥ 1.0 s; `spk_time` / `speech_total` ratio ≥ .10 and `dom_count ≥ 2` define a
   valid speaker; `multi = len(valid) >= 2`. This is README section 2 verbatim. Same-speaker overlapping turns would
   be double counted in both `ov` and `spk_time`, but the pyannote 3.1 output has none (0 overlapping same-speaker
   segment pairs in the 215 real HateMM records; per-speaker segments come from a binarized per-speaker track).
   `speech_total` is the sum of speaker time, so simultaneous speech of two speakers counts twice in the 10 % share
   denominator; this is one reading of "语音总时长" and makes the share test slightly stricter for everyone. Advisory A3.

3. **P2 oracle and permutation control.** `h_k` is the hateful-frame fraction of the windows speaker `k` owns
   (`precheck.py:130-135`), no-speaker windows get the video's overall hate fraction (`:136`), the curve is expanded on
   the same `fr` frames; both GT uses are the ones README section 2 declares and nothing else is read. The permutation
   (`:139-152`) shuffles the labels only among `has` (windows with an owner), leaves the grid, the frames and the
   no-speaker windows unchanged, uses `default_rng(0)` fresh per video, 10 draws, and recomputes `h_k` from the permuted
   grouping. `mean()` drops `None`, but on the eligible set `auc()` is never `None` for the oracle or r6_bma (both GT
   classes present, finite curves), and a multi-speaker video has ≥ 4 owned windows so the permutation list is never
   empty; oracle, r6_bma and the control are therefore averaged over the same videos. The gate `diff ≥ .05` is computed
   from those means (`:185, :208`). Note (not a bug, declared in the README): `spks` for the oracle includes speakers
   that are not "valid" (e.g. owning one window), whose `h_k` is then that window's own GT fraction; the permutation
   control is the README's declared measure of this grouping leak. On the real HateMM turns the oracle changes from
   .7354 to .7306 when restricted to valid speakers, so it does not decide the gate there (advisory A2).

4. **P3.** Positive window `hf ≥ .5`, negative `hf == 0`, false high = negative with window mean of the decoded curve
   strictly above the median over positive windows (`:158-161`); main speaker = argmax of hateful frames over owned
   windows, `None` if no owned window has a hateful frame (`:160`). Numerators and denominators are pooled window
   counts over videos that have a main speaker, numerator ⊂ denominator in both ratios (`:188-196`, `:199-204`). No
   off-by-one found; `r6w` is NaN only for zero-frame windows, which the `isfinite` test excludes and which never own a
   speaker.

5. **P4.** `pooled` starts as a copy of `z`; only windows whose owner is `k` are overwritten with the mean `z` of that
   owner's windows (`:154-157`); no-speaker windows keep their own `z`. Matches the README.

6. **Gates, decision, ids.** `gates` and the decision string implement "P1 and P2 pass on both corpora" (`:208,
   :246-248`). `records()` keys by `video_id`, `gold()` by `str(video_id)`, turns by the wav stem; the audio file names
   equal the reads ids exactly for both corpora (215 / 118). HateClipSeg GT id `yt_NzvfkIYS5Yg` has no read and no
   decoded curve and is skipped by `precheck.py:84`; eligible counts 84 / 99 equal rule 2's within counts. Videos whose
   diarization record is missing or failed count as non-multi in the P1 denominator (conservative, consistent with
   "within 可计视频").

7. **diarize.py.** pyannote.audio 3.3.2 / pyannote.core 6.0.1 / lightning 2.6.6 / torch 2.7.1 in the venv.
   `Pipeline.from_pretrained` has no `token` argument in 3.3.2, so the first call raises `TypeError` before loading
   anything and the `use_auth_token` branch runs. The checkpoint path goes through `pl_load` →
   `torch.load(f, map_location=..., weights_only=...)` by module attribute, so overriding `kwargs["weights_only"]`
   takes effect (job 413 log: 215 / 215 HateMM and ≥ 100 / 118 HateClipSeg ok, 0 failed; job 412 had failed earlier on
   a missing `matplotlib`, fixed in 36dd40a). `torchaudio.load` returns `(channels, time)` float32, mono 16 kHz files
   (checked), stereo would be averaged; `SpeakerDiarization.apply` returns an `Annotation` in 3.x, so
   `getattr(res, "speaker_diarization", res)` is the annotation itself; `itertracks(yield_label=True)` yields
   `(segment, track, label)` and the `for seg, _, spk` unpacking is correct; `labels()` counts speakers with at least
   one segment. Per-file exceptions are recorded as `status: failed` with `turns: []` and the loop continues
   (`diarize.py:101-105`), and the pre-check treats such records as no-speaker videos. `speech_seconds` is the sum of
   turn durations, so overlapping speech of different speakers is counted twice (10 of 215 HateMM records exceed
   `audio_seconds`; max 1.63 × the union of speech). The pre-check uses it only for the reported
   `speech_share_median`, not for any gate (advisory A4).

## Advisories (non-blocking)

- **A1 (length handling diverges from the evaluator in a case that does not occur).** `precheck.py:89-90` keeps all
  GT frames and edge-pads a shorter curve; `src/eval/evaluate.py:76-79` truncates both to the shorter length and drops
  non-finite frames. Current curves are 0 or +1 frames longer than the GT and finite, so the numbers are identical.
  Minimal fix if the pre-check is reused with other decoders: import `within_video_macro` from `src.eval.evaluate`
  for the r6_bma within (and feed the expanded oracle curves through the same function), which also answers rule 6's
  "统一评测器" item by construction.
- **A2 (oracle includes non-valid speakers).** `spks` at `precheck.py:128` is every owner, not only valid speakers; a
  speaker owning a single window hands that window its own GT fraction. This is declared ("每个说话人") and guarded by
  the permutation control; on the real HateMM turns it moves the multi-subset oracle by .005 (.7354 → .7306). If the
  user prefers the stricter reading, set `h_k` only for `k in valid` and give the other windows the video fraction.
- **A3 (share denominator).** `speech_total` at `precheck.py:95` is total speaker time, not the union of speech;
  simultaneous speech is counted once per speaker. Effect is on the 10 % threshold only.
- **A4 (`speech_seconds` in the diarization record).** Sum of turn durations, double counts overlapping speech;
  reporting only. If the field is meant as a speech-activity share, compute the union of the turns.
- **A5 (reviewer's own preliminary numbers, not authoritative).** With the complete real HateMM turns
  (`data/speaker_turns/HateMM/turns.jsonl`, 215 ok) the independent re-implementation gives P1 = 24 / 84 = .286 and
  P2 oracle .7354 vs r6_bma .6989 (diff +.036, permuted oracle .6695) on the multi-speaker subset. Both are below the
  declared gates (.30, .05). The authoritative numbers are those of `precheck.py` run after job 413 finishes; this
  note only says that the review did not find a bug that would move those gates.
