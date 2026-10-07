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

`sbatch experiments/20261008_baselines/launch/lab2_qwen3_text.sbatch` on uoa-lab2 (sc474399): writes
`runs/20261008_baselines/qwen3_8b_text/<DS>/` (`segment_scores.jsonl` = every segment's score, `predictions.jsonl`,
`coverage.json`, `config.json`, `metrics.json`, `run.log`).

## Status (2026-10-08 06:50)

Not run yet. Slurm 303 (`lab2_text_winonly.sbatch` = this job, then the Qwen2.5-VL winonly DeHate job) is pending
on uoa-lab2. The user account may run only 2 GPU jobs at a time across the lab cluster (QOS `gpu2`), and both slots
are held by long jobs of other baseline runs. The job writes `metrics.json` per corpus itself; afterwards:
`rsync -a uoa-lab2:Hate-follow-up/runs/20261008_baselines/qwen3_8b_text/ runs/20261008_baselines/qwen3_8b_text/`.
CPU check done on uoa-lab2: the chat template ends in `<|im_start|>assistant\n<think>\n\n</think>\n\n`, and every Yes/No
variant is one token (Yes ids 9454, 7414, 9693, 9834, 14004, 14080; No ids 2753, 2308, 2152, 902, 8996, 5664).

## Results

(pending)
