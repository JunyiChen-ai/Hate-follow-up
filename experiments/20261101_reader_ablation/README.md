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

## 6. Results (2026-11-02; development-selected; Qwen3-VL-8B)

**Runs.** All on RTX 5090 (32 GB), Slurm, 0 errors, every video of every arm read:

| job | host | scope | elapsed | per-arm GPU minutes (sum of per-video seconds) |
|---|---|---|---|---|
| 1357 | uoa-lab1 sc474397 | smoke, 6 videos × 7 arms + A5 on the 999 s HateMM video | 4 min | — |
| 1361 | uoa-lab1 sc474397 | HateMM, A1–A6 | 75 min | A1 12.3, A2 11.8, A3 11.4, A4 9.4, A5 22.1, A6 8.2 |
| 1362 | uoa-lab1 sc474397 | HateClipSeg, A1–A6 | 59 min | A1 8.9, A2 7.2, A3 8.8, A4 7.4, A5 19.5, A6 7.3 |
| 1360 | lab-server sc448960 | DeHate, A1–A4 (`DeHate_a`) | 179 min | A1 48.0, A2 38.1, A3 49.4, A4 43.0 |
| 1363 | uoa-lab1 sc474397 | DeHate, A5–A6 (`DeHate_b`) | 128 min | A5 88.0, A6 40.2 |

Code: commits f93c8b1 (definitions), 721f429 (code), b07583b (launch `NAME`), 005f0f2 (timing fields; jobs 1362,
1363 ran on it, 1361 and 1360 on b07583b; the reads do not depend on it). lab-server reads rsynced back to
`runs/20261101_reader_ablation/DeHate_a/`; lab-server reproduced lab1 reads exactly in earlier runs
(`experiments/20261101_video_plan` §5) and here on every shared value (`z_video` checks 0.0).

**Checks.** Smoke: the base equals the stored adopted reads on all 6 videos (`z_video`, every `z_visual`, every
`z_speech`, speech presence, score curve: max |Δ| = 0.0). Full runs: `z_video` of A3–A6 and A5's first branch equal the
stored adopted reads on every video of every corpus (max |Δ| = 0.0). A5 on the 999 s HateMM video: 81 560 cached
tokens, peak 29.0 GiB. Decoders: the base re-decode equals the adopted decodes exactly (max |Δ| 0.0, both groups);
A6 decodes with one chain (`modalities ('z_joint',)`, k = 4). Segment windows (A4): HateMM 2283 windows (1794 with a
0.5 fps frame), HateClipSeg 1555 (1272), DeHate 10 221 (8104; 2 videos without transcript have no window); the other
windows asked the visual question alone (the base rule for a window without a frame).

**Table** (arm − base, 95 % paired-bootstrap interval over videos; videos with GT: HateMM 215 / within 84,
HateClipSeg 118 / 99, DeHate 1151 / 222). Old = the paper's Reader block (within). A4 is compared under the SPVL
composition, so its base is the adopted reads composed the same way (`spvl_fixed`: HateMM .8984 / .6900 / .6869,
HateClipSeg .7325 / .6883 / .6640, DeHate .6937 / .1634 / .6439). Full table with paths:
`runs/20261101_reader_ablation/table.md` (`table.py`); sources `analysis_main/`, `analysis_DeHate/`,
`analysis_asr_main/`, `analysis_asr_DeHate/` (`table.txt`, `summary.json`).

| arm | corpus | old within | new within | new pooled ROC | new pooled PR | arm ROC / PR / within |
|---|---|---|---|---|---|---|
| A1 no_transcript | HateMM | −.039 | −.016 [−.054, +.022] | **−.144** [−.216, −.083] | **−.178** [−.284, −.079] | .7545 / .5144 / .7638 |
| | HateClipSeg | +.001 | −.004 [−.025, +.016] | −.021 [−.073, +.030] | −.016 [−.067, +.030] | .7120 / .6734 / .6825 |
| | DeHate | −.049 | −.021 [−.046, +.003] | **−.068** [−.113, −.021] | −.027 [−.067, +.015] | .6262 / .1371 / .6392 |
| A2 no_prefix_frames | HateMM | −.060 | −.003 [−.030, +.024] | **−.017** [−.034, −.004] | −.017 [−.045, +.007] | .8817 / .6751 / .7767 |
| | HateClipSeg | −.109 | **−.024** [−.043, −.007] | **−.068** [−.110, −.030] | **−.041** [−.073, −.014] | .6650 / .6487 / .6626 |
| | DeHate | −.003 | **−.027** [−.051, −.004] | −.009 [−.045, +.024] | −.013 [−.051, +.013] | .6854 / .1509 / .6335 |
| A3 no_stance | HateMM | −.014 | −.006 [−.031, +.018] | −.001 [−.008, +.005] | −.001 [−.018, +.016] | .8981 / .6912 / .7743 |
| | HateClipSeg | −.011 | −.011 [−.033, +.007] | **−.014** [−.029, −.001] | −.018 [−.038, +.002] | .7187 / .6712 / .6756 |
| | DeHate | −.010 | +.004 [−.014, +.022] | +.002 [−.003, +.007] | +.002 [−.005, +.010] | .6962 / .1659 / .6641 |
| A4 asr_windows (SPVL comp.) | HateMM | −.089 | **−.110** [−.172, −.049] | −.012 [−.029, +.002] | −.020 [−.066, +.026] | .8863 / .6703 / .5765 |
| | HateClipSeg | −.052 | **−.134** [−.181, −.088] | −.025 [−.061, +.002] | −.037 [−.071, +.002] | .7071 / .6508 / .5304 |
| | DeHate | −.065 | **−.081** [−.123, −.040] | −.017 [−.035, +.001] | −.024 [−.055, +.002] | .6766 / .1398 / .5627 |
| A5 no_isolation | HateMM | −.014 | **−.057** [−.099, −.014] | **−.030** [−.056, −.008] | **−.064** [−.110, −.017] | .8690 / .6284 / .7229 |
| | HateClipSeg | +.011 | −.021 [−.046, +.005] | −.010 [−.043, +.019] | +.001 [−.032, +.031] | .7222 / .6909 / .6661 |
| | DeHate | −.031 | −.020 [−.060, +.021] | −.003 [−.022, +.016] | −.013 [−.033, +.007] | .6905 / .1511 / .6405 |
| A6 joint | HateMM | +.009 | −.016 [−.045, +.011] | −.008 [−.027, +.009] | −.019 [−.062, +.021] | .8913 / .6735 / .7637 |
| | HateClipSeg | −.074 | **−.058** [−.098, −.021] | **−.033** [−.058, −.010] | −.038 [−.067, +.004] | .7000 / .6513 / .6289 |
| | DeHate | +.027 | +.014 [−.018, +.048] | +.009 [−.003, +.021] | −.003 [−.023, +.012] | .7033 / .1610 / .6746 |

Bold: interval excludes 0. Base (adopted, r6_bma raw key): HateMM .8989 / .6921 / .7799, HateClipSeg
.7327 / .6895 / .6866, DeHate .6940 / .1636 / .6603.

**Label-free read diagnostics** (`diag_reads.py` → `runs/20261101_reader_ablation/diag_reads.txt`; arm reads vs the
adopted reads, no labels):
- A1: the verdict changes most (Spearman with the base verdict .52 / .56 / .55); window reads fall by 1.2–2.4 nats.
  The HateMM pooled loss (−.144 ROC) is the verdict; within changes little.
- A2: verdict Spearman .90 / .80 / .81; window reads keep their level (mean shift −0.2 to 0.0) and their order
  (median per-video Spearman .85 / .88 / .83).
- A3: windows keep their order best (Spearman .92 / .93 / .91) and fall by 1.6–2.4 nats.
- A5: window reads rise by 2.0–4.7 nats after the first branch (no trend with position: HateMM +2.6 / +3.4 / +2.9 by
  thirds of the video) and lose their order (Spearman .68 / .70 / .70). The read after a positive read is not more
  persistent than in the base order: P(read > 0 | previous read > 0) − P(read > 0 | previous ≤ 0) is .245 vs .271
  (HateMM), .142 vs .159 (HateClipSeg), .161 vs .126 (DeHate). So the loss is a general drift of the reads in a
  growing conversation, not a "stays positive" chain (only DeHate shows a small extra persistence).
- A6: Spearman .78 / .76 / .77 with the dual window z; 14–21 % of windows change sign.

**Reading.**
- Under the adopted reader the segment windows (A4) remain the largest reader loss on all three corpora
  (−.110 / −.134 / −.081 within, all intervals exclude 0), larger than in the old table on HateClipSeg.
- The isolation loss (A5) is now largest on HateMM (−.057, interval excludes 0); HateClipSeg and DeHate are −.021 and
  −.020 with intervals including 0. The old arm had no answers in the cache (§2), so the rows are not the same
  intervention.
- One joint branch (A6) loses on HateClipSeg (−.058, interval excludes 0), is within noise on HateMM (−.016) and
  DeHate (+.014): the old pattern, smaller in size.
- The prefix frames (A2) cost less than the old "all frames" row, because the visual branch keeps its own frames:
  −.024 on HateClipSeg and −.027 on DeHate (both exclude 0), −.003 on HateMM; pooled ROC falls on both main corpora.
- The prefix transcript (A1) now matters mostly for the verdict (pooled ROC −.144 HateMM, −.068 DeHate); within
  losses are −.016 / −.004 / −.021 with intervals including 0 (the speech branches still carry their aligned words).
- The stance turn (A3) is within noise on every corpus (−.006 / −.011 / +.004 within).

## 7. Test-read log (rule 10)

- Files read with test labels: only the evaluator outputs (`metrics.json`) and the paired-bootstrap tables
  (`analyze_dvd.py` reads the 4 fps GT to compute the metrics and the P(V) > .5 shares by class). No per-video labels
  were inspected; no design was changed after seeing any number (the arm definitions were committed in f93c8b1 before
  the first GPU job; the only later code change adds timing fields).
- Label-free diagnostics (`diag_reads.py`) read predictions only.
