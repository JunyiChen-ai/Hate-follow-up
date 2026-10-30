# Window evidence in the reader: in-window frames + aligned words (adopted 2026-10-31)

## 1. What and why

The user adopted the "frames + aligned words" reader configuration on 2026-10-31 (it had been an open adoption
decision since candidate 43, `experiments/20261014_m1_audio_bound/README.md` §11; STATUS rule-11 summary).

- r6 names each window in its question by timestamps only. The visual branch has to find the window's frames among
  the 20 prefix frames; asked which prefix frames lie in the window, the reader is right on 36 % / 44 % of covered
  windows (candidate 41). The speech branch gets the transcript cut by proportion of Whisper segment times, which
  spills across window edges (segments average 11–16 s).
- Change: each window branch carries its own evidence after the shared verdict-conditioned context, right before
  its question. Visual: the window's 0.5 fps frames labelled `[t=..s]`. Speech: the words whose forced-aligned
  midpoint lies in the window. Prefix, verdict turn, questions, isolation and decoder are unchanged.
- Known on HateMM / HateClipSeg (development-selected; within vs r6_bma): +.029 / +.049; window-specific (in-video
  window shuffle control +.032 / +.045). It is an input change (rule 5), not a claimed mechanism.
- Open before this run: DeHate had no reads for this configuration.

Shared code promoted to `src/window_evidence.py` (frame sampler, aligned slice, frames-next-to-question read,
prediction record); `experiments/20261014_m1_audio_bound/align_words.py` gained the DeHate manifest.

## 2. How to run (uoa-lab1)

1. CPU inputs (audio for the aligner, 0.5 fps frames), detached:
   `setsid nohup .cache/envs/HateVLM/bin/python -u experiments/20261031_window_evidence/prep_inputs.py --dataset DeHate --jobs 12 > runs/20261031_window_evidence/prep_DeHate.log 2>&1 &`
2. Smoke on 2 + 2 HateMM / HateClipSeg videos; every read must equal the stored run it was promoted from
   (r6 base_gridA, placement `adjacent_local`, `vl_aligned`) within 1e-3:
   `SCOPE=smoke sbatch experiments/20261031_window_evidence/launch/lab1.sbatch`
3. DeHate (forced alignment, then all 1341 test videos): `SCOPE=dehate sbatch experiments/20261031_window_evidence/launch/lab1.sbatch`
4. Decode each read set with the current time level (r6_bma, raw key) and the single evaluator:
   `python experiments/20260926_twolevel/twolevel_r2.py --run runs/20261031_window_evidence/DeHate/<arm> --datasets DeHate --noleak --transform nscore --key raw --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2 --out-root runs/20261031_window_evidence/decoded --tag <arm>`

Read sets from one run: `r6_replay` (must equal the stored DeHate r6 reads), `frames`, `words`, `adopted`.

## 3. Results

(pending)
