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
- DeHate: run on **uoa-campus2 (foscsmlprd02), one NVIDIA A100-SXM4-80GB**, not on an RTX 5090 (the lab account may
  run only 2 GPU jobs at a time, so the pending lab2 job, Slurm 303, was moved to campus and cancelled). Second part of
  `sbatch experiments/20261008_baselines/launch/campus_qwen.sbatch` (Slurm 24594, 2026-10-08 10:30–10:47 NZDT, commit
  0d6012f), which runs the commands of `lab2_winonly_dehate.sbatch` unchanged: spvl.py on the 1341 test videos of
  `data/manifests/DeHate_test.jsonl` → `runs/20261008_baselines/qwen25vl_winonly/DeHate_spvl/` (1341 videos, 0 errors,
  904 s), then compose.py, then `finalize.py --datasets DeHate` (evaluated on the 1151 cohort videos). compose.py also
  writes `DeHate_spvl/metrics_ispvl_rrank.json`; that file comes from all 1341 predicted rows and is not the reported
  number. Environment: campus2's HateVLM env (torch 2.11.0+cu128, transformers 5.15.1, same package versions as
  lab2's HateVLM apart from pip and packaging). `Qwen/Qwen2.5-VL-7B-Instruct` was already in campus2's
  `.cache/hf` (same file names and sizes as lab2's cache; config, tokenizer, chat-template and index files
  byte-identical). Smoke test before the run: Slurm 24586 (2 videos, `runs/20261008_baselines/_smoke_campus/`).
  Outputs copied back to uoa-lab1 with `rsync -a`; re-checked there with
  `python experiments/20261008_baselines/campus_lab1_check.py runs/20261008_baselines/qwen25vl_winonly/<read-out>/DeHate:DeHate`
  (exact cohort, canonical evaluator into `metrics_lab1.json`; identical to the campus `metrics.json`).

## Results

Canonical evaluator, exact cohorts; pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC.

| read-out | HateMM | HateClipSeg | DeHate | source |
|---|---|---|---|---|
| raw | .7700 / .5701 / .6414 | .6182 / .6165 / .5662 | .6520 / .1443 / .6234 | `runs/20261008_baselines/qwen25vl_winonly/raw/<DS>/metrics.json` |
| ispvl_rrank | .5486 / .2721 / .6414 | .5397 / .5020 / .5662 | .5453 / .0873 / .6234 | `runs/20261008_baselines/qwen25vl_winonly/ispvl_rrank/<DS>/metrics.json` |

Frame pools: HateMM 116,975 frames / 215 videos (within defined on 84), HateClipSeg 113,002 / 118 (99), DeHate
441,345 / 1151 (222). No fallback was needed on any corpus.
