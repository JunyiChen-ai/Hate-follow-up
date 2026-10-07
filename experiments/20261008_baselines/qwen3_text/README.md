# Qwen3-8B reading Whisper transcript segments (run_plan.md L2), all three corpora

Label-free text-only baseline. No training, no tuning, no prompt selection, no label read.

## Setup

- Model: `Qwen/Qwen3-8B`, bf16, HF transformers 5.15.1 (HateVLM env), thinking off (`enable_thinking=False` in the
  chat template, so the first generated token is the answer).
- Unit: one Whisper large-v3 segment of `data/asr_whisper_large_v3/<DS>/timestamped_chunks.jsonl`, scored alone
  (no other segment, no frames, no title). Untimed segments are kept (run_plan.md §1.3): a missing end becomes the
  next segment's start or the video duration, a missing start the previous segment's end. Cohort segments:
  HateMM 2308, HateClipSeg 1582, DeHate 8709; every cohort video has at least one segment.
- Prompt: the shared hate query of `hate_query.md` §1 (rules copied verbatim from `spvl.py` `YOUTUBE_RULES`, the
  question verbatim), with spvl.py's system message:
  ```
  system: You are a content moderation analyst. Answer based strictly on observable evidence.
  user:   Based on this platform's hate speech policy, the following types of content are prohibited:
          <the 9 rules>

          Transcript of a speech segment from a video: <segment text>

          Does this segment contain content that violates any of the above rules? Answer "Yes" or "No".
  ```
  spvl.py's reading instruction (`READER_PRAG`, about judging the whole video and its visuals) is not included:
  `hate_query.md` fixes the rules and the question only, and the model sees one speech segment.
- Score: log-sum-exp over the Yes token ids minus log-sum-exp over the No token ids at the first answer position
  (id sets as spvl.py `binary_ids`: Yes, " Yes", yes, " yes", YES, " YES"; lm_head in fp32 on the last hidden
  state). One forward per segment, no padding.
- 4 fps grid: frame i (centre (i + 0.5)/4 s) takes the score of the segment containing its centre; overlaps take
  the maximum. F1: frames outside every segment take the score of the same prompt with "(no speech)" as the text
  (one constant). F4 (video without any segment) was not needed.

## Commands and host

Run on **uoa-campus2 (foscsmlprd02), one NVIDIA A100-SXM4-80GB**, not on an RTX 5090: the lab account may run only 2
GPU jobs at a time (QOS `gpu2`), so the pending lab2 job (Slurm 303, `lab2_text_winonly.sbatch`) was moved to campus
and cancelled once the campus job was processing videos.

- Job: `sbatch experiments/20261008_baselines/launch/campus_qwen.sbatch` on uoa-campus2, Slurm 24594, 2026-10-08
  10:16–10:47 NZDT (this text part 10:16–10:30), commit 0d6012f. The job is the lab2 file's two parts unchanged
  (this script, then the Qwen2.5-VL winonly DeHate run, see `../qwen25vl_winonly/README.md`); only the repository
  path and the environment differ (`launch/campus_env.sh`).
- Environment: campus2's `HateVLM` conda env (torch 2.11.0+cu128, transformers 5.15.1; its package list equals lab2's
  HateVLM apart from pip and packaging). Evaluator subprocess: `.cache/envs/lavad_tf449` (numpy 1.26.4,
  scikit-learn 1.5.2, as lab2's HateVideo).
- Model: `Qwen/Qwen3-8B` downloaded on campus2 into `.cache/hf` (`hf download`, log
  `runs/_setup_uoa-campus2/hf_download_qwen3.log`); same file names and sizes as lab2's cache, and the config,
  tokenizer and index files are byte-identical.
- Smoke test before the run: Slurm 24586 (2 cohort videos per corpus, no evaluation; outputs moved to
  `runs/20261008_baselines/_smoke_campus/qwen3_8b_text/`).
- Outputs, copied back to uoa-lab1 with `rsync -a`: `runs/20261008_baselines/qwen3_8b_text/<DS>/`
  (`segment_scores.jsonl` = every segment's score, `predictions.jsonl`, `coverage.json`, `config.json`,
  `metrics.json`, `run.log` whose first line is the host), `qwen3_8b_text/run.log`, `qwen3_8b_text/slurm_24594.out`.
- Re-check on uoa-lab1: `python experiments/20261008_baselines/campus_lab1_check.py
  runs/20261008_baselines/qwen3_8b_text/<DS>:<DS>` (exact cohort, finite score on every GT frame, canonical evaluator
  into `metrics_lab1.json`, comparison in `lab1_check.json`). All three corpora: exact cohort, numbers identical to
  the campus `metrics.json`.
- Chat template check (job log): the prompt ends in `<|im_start|>assistant\n<think>\n\n</think>\n\n`; Yes ids
  7414, 9454, 9693, 9834, 14004, 14080; No ids 902, 2152, 2308, 2753, 5664, 8996 (all one token).

## Results

Canonical evaluator, exact cohorts; pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC.

| corpus | ROC / PR / within | videos | frames | within defined on | segments | F1 frames | F4 / F2 videos | source |
|---|---|---|---|---|---|---|---|---|
| HateMM | .7540 / .5508 / .5673 | 215 | 116,975 | 84 | 2,308 | 14,344 | 0 / 0 | `runs/20261008_baselines/qwen3_8b_text/HateMM/metrics.json` |
| HateClipSeg | .5404 / .5515 / .5284 | 118 | 113,002 | 99 | 1,582 | 16,635 | 0 / 0 | `runs/20261008_baselines/qwen3_8b_text/HateClipSeg/metrics.json` |
| DeHate | .6162 / .1333 / .5643 | 1151 | 441,345 | 222 | 8,709 | 53,797 | 0 / 0 | `runs/20261008_baselines/qwen3_8b_text/DeHate/metrics.json` |

Empty-input score (F1 constant for frames outside every segment): −21.326880.
