# Reader ablation under the adopted reader (2026-11-01)

## 1. What and why

The paper's ablation table (`paper/overleaf/sections/experiments.tex`, Table `tab:ablation`, "Reader" block) still
holds the r6-era reader ablations: the main-corpus rows come from `experiments/20260926_twolevel/README.md` §14.4
(reading components under `r3`), `experiments/20260910_spvl/README.md` §8–§9 (ASR windows, isolation), and the
DeHate rows from `experiments/20260928_infer/README.md` §11–§12 (`experiments/20260922_til/til_measure.py` flags).
Since 2026-10-31 the method uses the window-evidence reader (`experiments/20261031_window_evidence/`,
`src/window_evidence.py`). This experiment re-runs the six reader rows on that reader, Qwen3-VL-8B, all three
corpora, decoded with the current time level. It is an ablation of the adopted method, not a new candidate: nothing
is promoted or decided here.

## 2. The adopted reader (base) and the six arms (declared 2026-11-01, before any GPU run)

**Base = the adopted reader, unchanged.** Per video, one shared KV cache:
1. Prefix: system message, intro, the 20 uniform frames of `data/frames_k20` each preceded by `[t=..s]`, the whole
   timestamped Whisper transcript, the 9 rules and the reader note (`Judge.prefix_messages(frames, segments)`).
2. Whole-video question → `z_video`; the model's answer ("Yes" if `z_video > 0`, else "No") written back as its
   assistant turn (the stance turn; `src/stance_cache.build`).
3. Fixed 8 s grid. Per window two isolated branches on the shared cache (each read in place, then the cache is
   cropped back, so no branch sees another):
   - visual: the window's 0.5 fps frames (`data/temporal_retrieved_kv/<ds>`, frames with `t1 <= t < t2`) labelled
     `[t=..s]`, then the visual question `yesno_question(i, n, t1, t2, ·, "visual")`, as one user turn
     (`src/window_evidence.frames_margin`); a window without a 0.5 fps frame (7 of 16514 DeHate windows) asks the
     visual question alone (`src/stance_cache.margin`);
   - speech: `yesno_question(i, n, t1, t2, words, "speech")` with the words whose forced-aligned midpoint lies in
     `[t1, t2)` (`data/asr_whisper_aligned/<ds>`; videos without an alignment use the proportional slice
     `window_text`); read only when the slice is non-empty.
4. Window read `z = max(z_visual, z_speech)`; record schema `src/window_evidence.window_prediction`.

Each arm changes exactly one thing; everything not named stays as in the base.

| arm | old paper row | change against the base |
|---|---|---|
| A1 `no_transcript` | transcript context in the prefix | Prefix without the whole transcript: `prefix_messages(frames, segments, with_context=False)` (intro loses "The full transcript with timestamps follows the frames.", no `Transcript:` block; 20 frames, rules, note unchanged). Verdict and stance turn on that prefix. Window branches unchanged: the speech branch still gets its aligned words, which are then the only speech the model sees. |
| A2 `no_prefix_frames` | the 20 frames | Prefix without the 20 frames: `prefix_messages([], segments, with_frames=False)` (intro "Observe this video. The full transcript with timestamps follows."; no image in the prefix). Verdict and stance turn on that prefix. Window branches unchanged: the visual branch keeps its own 0.5 fps frames (the only frames the model sees, each branch only its own window's); the rare window without a 0.5 fps frame asks the visual question with no image at all. |
| A3 `no_stance` | stance turn | The verdict answer is not written back: `z_video` is read on the prefix, the cache is cropped back to the prefix, and every window branch follows the prefix directly (no verdict turn in its context). `z_video` is still used by the decoder. |
| A4 `asr_windows` | fixed 8-s grid | Windows = the Whisper segments (`load_asr`, as loaded for the prefix; start/end as given, in order) instead of the 8 s grid. Visual branch: the 0.5 fps frames with `s <= t < e`; speech branch: aligned words with midpoint in `[s, e)` (videos without an alignment: the segment's own text). Same question wording with the segment's index, count and times. Gaps between segments are unscored (4 fps curve −12 there, as the old arm); a video without transcript has no windows. Decoding: see §3. |
| A5 `no_isolation` | branch isolation | Branches see each other in order: one growing conversation per video after the verdict turn, window-major order (window 1 visual, window 1 speech, window 2 visual, …). Each branch is appended in place as a user turn (visual: frames + question; speech: question with words), its Yes/No log-odds is read, and the model's answer ("Yes" if the log-odds > 0, else "No") is written back as its assistant turn; nothing is cropped, so every later branch sees the frames, words, questions and answers of all earlier branches. |
| A6 `joint` | two branches | One branch per window instead of two: the window's 0.5 fps frames (labelled, as the visual branch) then `yesno_question(i, n, t1, t2, words, "joint")` with the aligned words of the window ("(no speech)" when empty), read for every window. Read key `z_joint`, `z = z_joint`. |

**Differences from the old arms (mapping decisions, declared before running).**
- A2: the old arm (`--frames 0`) removed all frames, so it had no visual branch (speech branches only, a joint branch
  for windows of videos without speech). Under the adopted reader the visual branch has its own frames, so A2 as
  defined by the request removes only the 20 prefix frames and keeps the visual branch. It measures the prefix
  frames, not "all frames".
- A5: the old arm (`--isolation sequential` in `til_measure.py`, and the SPVL causal-mask arm) kept each branch's
  question tokens and assistant header in the cache but generated no answers. The request defines A5 as keeping each
  answer, and the paper text describes the row as "each window's answer conditions the next", so A5 writes back the
  model's own Yes/No after every branch (the same rule as the stance turn). This is closer to the stated row than
  the old arm was; it is not the same intervention as the old arm.
- A4: as in the old arm, ASR windows are often longer than two 4 s cells and cannot enter the time level, so A4 is
  compared under the SPVL composition (§3).
- A6: the old joint arm (`--branches joint`) showed no frames in the branch (it relied on the prefix frames); under
  the adopted reader the single branch carries the window's frames and its aligned words, so it is the joint read of
  the same window evidence the two branches get.

## 3. Decoding and comparison

- A1, A2, A3, A5, A6: the current time level, exactly the adopted decode flags
  (`python experiments/20260926_twolevel/twolevel_r2.py --noleak --transform nscore --key raw --duration bma
  --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`), HateMM + HateClipSeg in one run, DeHate in its own
  run (EM is fitted per corpus on all its videos). A6 has one modality key (`z_joint`), so the time level has one
  chain (the decoder takes the keys present, as the old `m1_joint`). Base = the adopted decodes
  `runs/20261031_window_evidence/decoded_main/adopted_rawkey` (HateMM / HateClipSeg) and
  `runs/20261031_window_evidence/decoded/adopted` (DeHate).
- A4: SPVL composition (`experiments/20260910_spvl/compose.py --intercept zv_plus_mean --residual rank`) on both the
  A4 reads and the base reads, difference A4 − base under that composition (the old `spvl_asrwin` vs `spvl_fixed`).
- Differences: `experiments/20260927_dvd/analyze_dvd.py` (paired bootstrap over videos, 4000 draws, seed 0), arm −
  base, pooled ROC / pooled PR / within. Single evaluator `src/eval/evaluate_four_datasets.py` (called by both
  decoders). No label enters any read or decode.

## 4. Checks (in-job, no tensors saved)

- Smoke (`--smoke`): the first two videos of HateMM, HateClipSeg and DeHate (manifest order), every arm. The base
  must equal the stored adopted reads (`runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned`,
  `runs/20261031_window_evidence/DeHate/adopted`) in `z_video`, every `z_visual`, every `z_speech` and speech presence,
  max |Δ| ≤ 1e-3 (the window_evidence tolerance; that run got 0.0). Plus A5 on the longest HateMM video
  (999 s, the memory worst case of the growing cache).
- Full runs: arms whose prefix and verdict equal the base (A3, A4, A5, A6) must reproduce the stored `z_video`
  (≤ 1e-3) on every video; A5's first branch (window 1 visual) has nothing before it and must reproduce the stored
  window-1 `z_visual`. Violations end the job with `FAILED`.

## 5. How to run

Code: `src/reader_arms.py` (new shared module: `build_context` with the prefix-transcript / prefix-frames / stance
switches, `asr_windows`, `SequentialReader`, `record`), `experiments/20261101_reader_ablation/read_arms.py` (arms,
checks, records), `launch/lab1.sbatch`, `launch/lab_server.sbatch`, `decode_eval.sh`. Existing `src/` files unchanged.

1. Smoke (uoa-lab1): `SCOPE=smoke sbatch experiments/20261101_reader_ablation/launch/lab1.sbatch`
   → `runs/20261101_reader_ablation/smoke/`.
2. Reads, one corpus per job, the listed arms in arm-major order (all six when `ARMS` is empty):
   - uoa-lab1: `SCOPE=HateMM sbatch experiments/20261101_reader_ablation/launch/lab1.sbatch`, then `SCOPE=HateClipSeg …`,
     then `SCOPE=DeHate ARMS="no_isolation joint" NAME=DeHate_b sbatch experiments/20261101_reader_ablation/launch/lab1.sbatch`;
   - lab-server (after `git pull`; 0.5 fps frames, aligned words and the stored adopted reads rsynced from uoa-lab1 first):
     `SCOPE=DeHate ARMS="no_transcript no_prefix_frames no_stance asr_windows" NAME=DeHate_a sbatch experiments/20261101_reader_ablation/launch/lab_server.sbatch`.
   Each DeHate arm runs whole on one machine; the two DeHate run directories hold disjoint arms and are linked as
   `runs/20261101_reader_ablation/DeHate/<arm>` after the rsync back.
3. Decode, compose and compare (CPU, uoa-lab1): `bash experiments/20261101_reader_ablation/decode_eval.sh main` and
   `… DeHate` → `runs/20261101_reader_ablation/decoded/<arm>_<main|DeHate>/metrics.json`,
   `compose/<fixed|asr>_<group>/metrics_izv_plus_mean_rrank.json`, `analysis_<group>/table.txt`,
   `analysis_asr_<group>/table.txt`.

### 5.1 Timing run of the adopted reader (main-session request, declared 2026-11-01 before running)

The paper's cost paragraph needs a measured per-video time of the adopted reader on HateMM / HateClipSeg (until now
only component sums; the DeHate timing covered four read sets in one pass). One uninterrupted Slurm job per corpus,
the base arm only (no switch changed), each job alone on its GPU, on lab-server after its DeHate arm job:
`SCOPE=HateMM ARMS=base NAME=timing_HateMM sbatch experiments/20261101_reader_ablation/launch/lab_server.sbatch`, then
`SCOPE=HateClipSeg ARMS=base NAME=timing_HateClipSeg …`. Each record keeps `seconds` (the whole video after model
load: frame listing, prefix forward, verdict, stance turn, all window branches), `prefix_seconds` (prefix forward +
verdict + stance turn), `window_seconds`, `prefix_tokens`, `peak_GiB`; the base check against the stored adopted reads
(`z_video`, every window read, speech presence, score curve) runs on every video. The `prefix_seconds` field was
added to `read_arms.py` after the HateMM arm job (Slurm 1361) had started; the reads are unchanged.

## 6. Results

(pending)

## 7. Test-read log (rule 10)

(pending)
