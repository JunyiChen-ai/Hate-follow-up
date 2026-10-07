# Qwen2.5-VL-7B, per-window scoring without context (`winonly`), all three corpora

Label-free baseline, run_plan.md L8. No model is trained, nothing is tuned, no label is read.

## Setup

- Script: `experiments/20260910_spvl/spvl.py`, unchanged, arm `winonly` (same flags as
  `experiments/20260910_spvl/launch/run_mllm.sh`): `--model Qwen/Qwen2.5-VL-7B-Instruct --isolation cache --frames 0
  --no-transcript-context --branches joint --window-question rules --windows fixed --window-seconds 8`.
- What the model sees per 8 s window: system message, the 9 policy rules + spvl.py's reading instruction, then
  "Consider only window i of n, from a s to b s of this video. Transcript in this window: <Whisper text or
  (no speech)>" and "Does THIS window contain content that violates any of the above rules? Answer "Yes" or "No"."
  No frames, no whole-video transcript. Score = log P(Yes) − log P(No) at the answer position (spvl.py token sets,
  fp32 head).
- Transcripts: `data/asr_whisper_large_v3/<DS>/timestamped_chunks.jsonl` (spvl.py drops segments with no end
  time; kept unchanged so that all three corpora have the HateMM / HateClipSeg setup).
- Two read-outs of the same forward passes, evaluated separately (`finalize.py`):
  - `raw`: each 4 fps frame holds the log-odds of the window containing its centre.
  - `ispvl_rrank`: `compose.py --intercept spvl --residual rank`, the "per-window alone" row of
    `runs/20260910_spvl/mllm_table.md`. In this arm the whole-video question sees no video content, so its
    log-odds is the same constant for every video (−3.4996 on HateMM / HateClipSeg); the read-out keeps only the
    order of windows inside each video, so its pooled numbers are close to chance by construction.

## Commands and hosts

- HateMM, HateClipSeg: existing outputs `runs/20260910_spvl/mllm/q25vl-7b/winonly/` (spvl.py, 2026-09-10,
  uoa-lab3 sc474398, transformers 5.15.1). Coverage check: 215/215 and 118/118 cohort videos, every GT frame
  finite, no fallback. Re-evaluated on uoa-lab1 with
  `python experiments/20261008_baselines/qwen25vl_winonly/finalize.py --datasets HateMM HateClipSeg`.
- DeHate: `sbatch experiments/20261008_baselines/launch/lab2_winonly_dehate.sbatch` on uoa-lab2 (sc474399):
  spvl.py on the 1341 test videos of `data/manifests/DeHate_test.jsonl` →
  `runs/20261008_baselines/qwen25vl_winonly/DeHate_spvl/`, then compose.py, then `finalize.py --datasets DeHate`
  (evaluated on the 1151 cohort videos). compose.py also writes `DeHate_spvl/metrics_ispvl_rrank.json`; that
  file holds all 1341 predicted rows and is not the reported number.

## Results

Canonical evaluator, exact cohorts; pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC.

| read-out | HateMM | HateClipSeg | DeHate | source |
|---|---|---|---|---|
| raw | .7700 / .5701 / .6414 | .6182 / .6165 / .5662 | (pending) | `runs/20261008_baselines/qwen25vl_winonly/raw/<DS>/metrics.json` |
| ispvl_rrank | .5486 / .2721 / .6414 | .5397 / .5020 / .5662 | (pending) | `runs/20261008_baselines/qwen25vl_winonly/ispvl_rrank/<DS>/metrics.json` |

Frame pools: HateMM 116,975 frames / 215 videos (within defined on 84), HateClipSeg 113,002 / 118 (99).

DeHate status (2026-10-08 06:50): not run yet; it is the second half of Slurm 303 (`lab2_text_winonly.sbatch`),
pending on uoa-lab2 behind the account's 2-GPU limit. The job runs spvl.py, compose.py and `finalize.py --datasets
DeHate` itself; afterwards rsync `runs/20261008_baselines/qwen25vl_winonly/` back to uoa-lab1.
