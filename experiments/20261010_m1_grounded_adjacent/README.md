# M1 candidate 41: grounded acceptance of adjacent-frame reads

Declared 2026-10-10 before implementation and before any GPU run. User decision 2026-10-10: a per-window
accept-or-fall-back on the adjacent read, decided by the model's own cited evidence, does not violate rule 3.
Development-selected; frozen Qwen3-VL-8B, native G / own stance / S, original questions, 8 s windows, 4 fps,
sole evaluator, fixed r6 decoder; HateMM 215 and HateClipSeg 118, same constants and code for both.

## Evidence and hypothesis

- Placement controls (`experiments/20261007_m1_streamingtom/README.md`, 2026-10-10): showing the window's own
  prefix frames again right before the visual question (`adjacent_native`, no new image) gives within +.0224 /
  +.0128 over r6 with no loss beyond noise.
- Selection analysis (same README): 47 % / 45 % of the windows that change move away from the label. Accepting
  only the helpful changes (with labels, a ceiling) gives within +.053 / +.048 and moves the pooled metrics too.
  Signals known before the read (confidence, in-video rank, speech disagreement, frame coverage) do not beat
  size-matched random subsets. On HateMM the main harm is negatives pushed up (34 % of changed windows); on
  HateClipSeg positives pushed down (27 %) and negatives pushed up (18 %).
- Hypothesis: whether the adjacent read moved for a visible reason can be tested on that read's own content. A
  second probe on the same context asks the model to point at the shown frame that contains violating content,
  or say none. A rise the model cannot point at, or a fall the model contradicts by pointing at a frame, is not
  accepted; the window keeps its native read.

## Mechanism (exact, both corpora)

1. Native prefix (20 frames, full ASR, rules), G, own hard stance, speech reads S: r6, unchanged.
2. Adjacent read, for every window with at least one prefix frame inside [start, end): r6's visual question
   preceded in the same user turn by those frames shown again with their `[t=..s]` labels, exactly as
   `adjacent_native`; Yes/No margin z from the FP32 head. Windows with no prefix frame inside keep the native
   read v and get no probe.
3. Probe, same cache: the adjacent user turn is kept (the assistant header used for the margin read is
   removed), then one more user turn:
   `Among the frames shown for this window, which one, if any, contains content that violates the rules above?
   Reply with that frame's timestamp exactly as labelled, or "none".`
   Greedy decoding, at most 16 new tokens. The reply is grounded when it contains a number equal (within .05 s)
   to the time of one of the frames shown for this window.
4. Decision: final visual score = z when (z > v and grounded) or (z ≤ v and not grounded); otherwise v.
   Fusion max(V, S), G and the r6 decoder unchanged.

Constants: 16 new tokens; .05 s match tolerance; the probe text above. Nothing else; no labels in 1 to 4.

## Difference from earlier attempts (checked 2026-10-10)

- Explorer, candidate 18 (`archive/experiments/20261003_m1_explorer/`): a pre-read entropy gate decided whether
  to acquire new frames; the gate fired in 7344 / 7359 windows and HateMM within reached +.009. Here the frames
  are always shown and the decision is made after the read, on its content.
- HVL (`experiments/20260911_hvl/`): generated hypotheses verified by the model, with confirmation effects. Here
  nothing is generated to verify; the probe does not see the model's answer (fresh user turn) and only accepts
  or rejects a score change.
- Attributor (2026-10-02): attribution over cached values; Grounder (2026-10-02): access restriction; candidate
  27 verification (never run): factual re-observation before measurement; Provenance 25: entity graph; Interval
  Witness 26 (`archive/experiments/20261005_m1_interval_witness`): cited frame witnesses for factual fields, not
  for a window score. None accepts or rejects a window read by the model's own cited frame.

Independent proposal review (rule 4, 2026-10-10): PASS, `docs/reviews/20261010_m1_grounded_adjacent_proposal.md`.
Advisory from the review: this model's own citations localized near chance in HVL E0 and GLR section 7, so the
random-acceptance control is the decisive test.

## Inputs and cost

Nothing new to extract. Per covered window: the adjacent read (+4.7 min for 333 videos both corpora, measured)
plus the probe, about 45 prefill tokens and at most 16 greedy steps on the extended cache, estimated 0.2 to 0.4
s, so 5096 windows about 20 to 35 min. Whole run about 40 to 50 min on one 5090, peak under 20 GiB.

## Declared gates and controls (before any run)

- Performance (rule 8) against r6_bma: no metric below the noise floor (−.005 pooled, −.01 within) in either
  corpus, and one metric ≥ +.01 in both.
- Mechanism: within above accept-all (`adjacent_native`, stored in
  `runs/20261007_m1_streamingtom/placement_controls_main_decoded/adjacent_native/metrics.json`) by ≥ .01 in both
  corpora = supported; by ≥ .01 in one corpus with no loss beyond noise in the other = partial; otherwise not
  supported. The grounded rule must also be above a seeded random acceptance with the same per-corpus
  acceptance rate (built offline from the same reads, seeds 0 and 1) by more than the noise floor on within in
  both corpora. The inverted rule (accept where the rule rejects) is reported.
- Reported with the numbers: acceptance rate, share of probes answering none, share of cited times that are
  inside the window, parse failures.
- Pilot: the fixed five videos first, before the 333. If the probe answers none in more than 95 % or fewer than
  5 % of the probed windows, the probe text is revised and re-declared here before the 333 run (constants are
  fixed before the scoring run; no GT is read in the pilot).
- Everything development-selected. DeHate not part of this round.

## Run

Code `grounded.py` (reading), `analyze.py` (binding checks, offline controls, sole evaluator + fixed r6, report),
CPU fixture check `selfcheck.py`; launch `launch/lab1.sbatch`, `launch/lab2.sbatch`, `launch/lab3.sbatch`
(`SCOPE=smoke|main|both`). Outputs `runs/20261010_m1_grounded_adjacent/`.

## Pilot on the fixed five (2026-10-10, sc474397 / Slurm 353; no GT read)

76 probed windows: replies well formed (`[t=11.9s]`, `t=49.9s`, `none`), 0 parse failures; `none` in 84 %
(HateMM 81 %, HateClipSeg 88 %), a shown frame cited in 16 %; acceptance 75 %; the adjacent read moved up in 30 %
of the probed windows. The pilot rule (none above 95 % or below 5 %) is not triggered, so the probe text stays as
declared. Plumbing check `runs/20261010_m1_grounded_adjacent/smoke_analysis/plumbing_summary.json` PASS: native
reads equal r6, and the adjacent margin equals the stored `adjacent_native` read bit for bit (max |Δz| 0 over 76
windows). Peak 19.5 GiB; about 2 generated tokens per probe.

## Full run (2026-10-10, sc474397 / Slurm 354, `SCOPE=both`: fixed five, in-job plumbing check, then 333)

Binding (`runs/20261010_m1_grounded_adjacent/main_analysis/alignment.json`, PASS): native reads equal r6 in every
window; the adjacent read equals the stored `adjacent_native` read exactly in all 5096 covered windows (max |Δz| 0);
decisions re-derived from the stored replies. Acceptance 60.7 % (HateMM) / 66.1 % (HateClipSeg); probe answered
`none` in 87.2 % / 83.8 %, cited a shown frame in 12.7 % / 16.2 %; 1 parse failure (HateMM, 2738 probes); about 2
generated tokens per probe. Every cited number was a shown frame's time (349 / 376 citations, all inside the
window). Time for 333 videos: native reads 305 s + 247 s; adjacent read plus probe 337 s + 280 s, so the probe adds
about 5.6 min on top of the adjacent read's 4.7 min. Peak 19.5 GiB; job 20 min.

Sole evaluator + fixed r6 (`runs/20261010_m1_grounded_adjacent/analysis/summary.json`; metrics in
`main_decoded/<arm>/metrics.json`):

| arm | acceptance HMM / HCS | HateMM ROC / PR / within | HateClipSeg ROC / PR / within |
|---|---|---|---|
| r6_bma | – | .8971 / .6942 / .7508 | .7168 / .6711 / .6373 |
| accept_all (= `adjacent_native`, replay exact) | 1.00 / 1.00 | .8970 / .6938 / .7732 | .7252 / .6771 / .6501 |
| grounded (candidate 41) | 0.61 / 0.66 | .8977 / .6969 / .7717 | .7240 / .6762 / .6487 |
| random_0 (rate matched) | 0.61 / 0.66 | .8968 / .6926 / .7652 | .7226 / .6768 / .6512 |
| random_1 (rate matched) | 0.61 / 0.66 | .8965 / .6926 / .7656 | .7216 / .6746 / .6425 |
| inverted | 0.39 / 0.34 | .8964 / .6913 / .7555 | .7186 / .6729 / .6393 |

Declared gates:
- Performance vs r6: PASS (within +.0209 / +.0113, no loss) — the same gain `adjacent_native` already had.
- Mechanism vs accept_all: within −.0015 / −.0014, pooled within ±.003: **not supported**.
- Against random acceptance at the same rate: within +.0065 / −.0025 (seed 0) and +.0061 / +.0062 (seed 1); does
  not beat random in both corpora (floor .01). Inverted rule: within +.0047 / +.0019 vs r6, −.0177 / −.0108 vs
  accept_all, so the rule's rejected moves do contain most of the harm it was supposed to remove — see below for
  why that does not turn into a gain.

## Test error analysis (rule 10, 2026-10-10)

Files read: `runs/20261010_m1_grounded_adjacent/main/records` (traces with probe replies), `data/gt_4fps/*.npz`,
decoded predictions of the offline rules under `runs/20261010_m1_grounded_adjacent/error_analysis/decoded/`,
`runs/20260926_twolevel/r6_bma/predictions.jsonl`, and the 2026-10-10 selection-analysis oracle
`runs/20261007_m1_streamingtom/selection_analysis/decoded/adjacent_native__oracle_help/predictions.jsonl`. Scripts
`error_analysis.py`, `error_analysis_mixed.py`; outputs `error_analysis/{summary.json,within_eligible.json}`,
logs `error_analysis_run.log`, `error_analysis_mixed_run.log`. Window label = any positive 4 fps GT frame in the
window; "toward the label" = the visual margin moves up on a positive window or down on a negative one (the r6
decoder reads z_visual and z_speech as separate conditions, so the visual direction is the relevant one).

Findings:
1. Over the whole corpus the probe does carry label information, but at the video level. Among up moves on
   HateMM, windows where the probe cites a shown frame are positive in 60 %, windows where it says `none` in 24 %
   (HateClipSeg 73 % vs 57 %). The `none` answers sit mostly in all-negative videos.
2. The within metric only uses the 84 / 99 videos that contain both labels (100 HateClipSeg GT videos are mixed; one is not in the run). There, 65–77 % of the covered windows
   in every cell are positive and the probe barely separates them: up moves are positive in 67 % (`none`) vs 75 %
   (shown) on HateMM and 65 % vs 77 % on HateClipSeg; down moves 65 % vs 80 % and 57 % vs 77 %. So inside these
   videos the grounded rule rejects 378 / 620 up-`none` moves that go toward the label in 68 % / 65 % of cases,
   and accepts 541 / 1007 down-`none` moves that go away from it in 64 % / 57 %. Weighted by |Δz|, the moves it
   accepts are 57 % helpful on HateMM against 60 % for accepting everything.
3. Per-video decoded within, grounded minus accept_all: 18 videos better / 20 worse (HateMM), 40 / 35
   (HateClipSeg), means −.0015 / −.0014, single videos move by up to ±.18. accept_all minus r6 itself is 30 better /
   30 worse on HateMM (mean +.0224): the macro within is driven by a few videos either way.
4. Headroom check in the same videos: accepting only the visual moves toward the label gives +.0307 / +.0348 over
   accept_all (45 better / 5 worse on HateMM); accepting by the direction of the fused max(V, S) gives only
   +.0030 / +.0171, because moves that change V but not the max still reach the decoder. The headroom is
   window-level inside hateful videos; the probe's reply does not carry that information.
5. Other label-free rules on the same reads (decoded within vs r6, HateMM / HateClipSeg): accept only down moves
   +.0169 / +.0051; only up moves +.0152 / +.0128; down-and-`none` only +.0163 / +.0046; up-and-cited only
   +.0059 / +.0091; random at the down-move rate +.0198 / +.0117 and +.0155 / +.0015. None is above accept_all
   (+.0224 / +.0128); undecoded (raw window margins) the picture is the same (`error_analysis/<rule>/metrics.json`).

Design decision: no revision run. The one revision the analysis suggests — ask the probe for the most violating
frame of the whole video instead of only the shown frames, so that the answer is relative to the other windows —
re-tests a signal this model has already failed on: its whole-video citations localized within videos at
.580 / .526 (GLR §7) and .571 / .506 (HVL E0), near chance. Candidate 41's own contribution over its control
(accept_all) is within noise on both corpora; the ≥ .01 within gain over r6 belongs to the adjacent read that was
already known. Recommendation: archive under rule 9 (no gain of the mechanism); whether to spend the 3 revisions
is the user's decision. The adjacent read itself remains an input change (rule 5), adoption still the user's call.

## Revision 1 (declared 2026-10-11 before implementation and before any GPU run): probe before frames

User clarification 2026-10-11: the intended mechanism was never "add frames everywhere, then decide whether to keep
the result". It is: read every window natively first (r6, no extra frames); then one more operation on that
native read decides, per window, whether this window gets its frames; only those windows are re-read. The
candidate 41 main run tested the reversed order and is kept as a negative result above. This revision tests the
intended order. It counts as revision 1 of 3 under rule 9.

Why the model's own answer might mark the right windows: the placement controls showed that re-showing the
window's prefix frames next to the question helps even though no new image is added, i.e. the model has trouble
finding the in-window frames among the 20 by their time labels. If that failure is window-specific and visible
in what the model says when asked where the window's frames are, the windows where it fails are the windows
where re-showing should help; where it succeeds, re-showing changes nothing useful.

Mechanism (exact, both corpora, same constants):
1. r6 unchanged: prefix with 20 frames + full ASR + rules, G, own hard stance, per-window visual V and speech S
   from the prefix cache. Nothing new is shown to the model.
2. After the native visual read of window w (same cache, user turn kept, assistant header not kept; the model
   does not see its own answer), probe A, text only:
   `Which of the frames shown above have timestamps inside this window, from {start:.1f}s to {end:.1f}s? List
   their timestamps exactly as labelled, separated by commas, or reply "none".`
   Greedy, at most 48 new tokens. Numbers in the reply are matched to the prefix frame times (within .05 s +
   1e-9, as before). The reply is correct when the matched set equals the set of prefix frames whose time lies in
   [start, end) (an empty set is correct only if the reply contains "none" and no matched time). Window flagged
   by rule A when the reply is not correct.
3. Probe B, text only, on the same cache after the native read with the model's own hard answer appended as an
   assistant turn ("Yes" if V > 0 else "No"): `Which single frame shown above did you base that answer on? Reply
   with its timestamp exactly as labelled, or "none".` Greedy, at most 16 new tokens. Window flagged by rule B
   when no matched time lies inside [start, end) (it cites a frame outside the window, cites nothing, or says
   none).
4. Flagged windows take the extra visual read; unflagged windows keep V. The extra reads are the ones already
   stored for every window (`runs/20261007_m1_streamingtom/placement_controls_main/records`): arm a =
   `adjacent_native` (exists only for windows with a prefix frame inside; other windows keep V whatever the
   flag), arm l = `adjacent_local` (0.5 fps frames next to the question), arm E = `local_clean` (0.5 fps frames
   in the prefix). The selection is therefore evaluated offline; the GPU job runs only r6's reads plus the two
   probes. Fusion max(V, S), G, decoder: fixed r6.

Constants: the two probe texts; 48 and 16 new tokens; .05 s match tolerance; the rule definitions above. No label
enters steps 1 to 4. Cost: two text-only probes per window (about 270 question tokens re-encoded plus 45 to 50
probe tokens each, and the generated tokens); estimated 0.2 to 0.4 s per window, 7359 windows, 25 to 50 min on
one 5090 on top of r6's 9 min; peak under 20 GiB. In deployment arm a's own read costs 4.7 min for 333 videos, so
with arm a the probes cost more than the read they would save: this revision tests accuracy (does the rule find
the windows where the extra read helps), not cost.

Declared gates and controls (before any run):
- Pilot on the fixed five first, no GT read: if probe A is correct in more than 95 % or fewer than 5 % of the
  windows, or probe B cites an in-window frame in more than 95 % or fewer than 5 %, the probe text is revised and
  re-declared here before the 333 run. Binding: native V and S equal r6 in every window; the traces' in-window
  frame sets equal the stored `native_inside` of the placement records.
- Six selections are evaluated (rules A, B × arms a, l, E); primary = rule A on arm a. For each: within gain vs
  r6, compared with (i) the full arm and (ii) six rate-matched random selections (independent per video, seeds
  0 to 5, built offline from the same reads; the review of 2026-10-11 showed two seeds cannot locate the random
  band). A selection is supported when its within is above all six random controls by more than .01 in both
  corpora and no pooled metric is below r6 by more than .005 in either corpus. Reported alongside: flagged
  fraction, probe statistics (correct rate, none rate, inside rate, parse failures, generated tokens, time), the
  visual-direction oracle for reference. Six selections are tested, so a single selection meeting the gate on one
  corpus only is not a result.
- Everything development-selected; DeHate not part of this round.

Code: `probe_native.py` (reading job), `probe_native_analyze.py` (binding, selections, controls, sole evaluator +
fixed r6, report), `launch/lab{1,2,3}_probe.sbatch` (`SCOPE=smoke|main|both`). Outputs
`runs/20261010_m1_grounded_adjacent/probe_{smoke,main}/`, analysis `probe_analysis/`.

### Revision 1 pilot on the fixed five (2026-10-11, sc474397 / Slurm 372; no GT read)

158 windows (76 with a prefix frame inside). Binding PASS (`probe_smoke_analysis/plumbing_summary.json`): native V
and S equal r6 in every window; in-window frame sets equal the placement records' `native_inside`. Replies well
formed (`2.4s, 7.1s`, `[t=21.4s], [t=26.1s]`, `none`), 0 parse failures. Probe A correct in 64 % / 39 % of all
windows, but only 31 % / 35 % of the windows that do contain a prefix frame (the model lists a neighbouring frame,
one frame too many, or `none`); uncovered windows are answered `none` correctly in 83 % / 45 %. Probe B cites an
in-window frame in 14 % / 26 %, `none` mostly where the native answer was No. Neither pilot threshold (above 95 %
or below 5 %) is reached, so both probe texts stay as declared. Numbers within .05 s of a window bound appeared in
11 of 158 windows (recorded as `bounds_echo`). Probes 0.27 s per window (both probes, about 7 + 2 generated
tokens), so about 33 min for 7359 windows on top of r6's reads; peak 19.5 GiB.
