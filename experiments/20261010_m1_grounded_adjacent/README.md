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
