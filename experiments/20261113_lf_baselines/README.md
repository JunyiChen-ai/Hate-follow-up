# Label-free baselines: naive prompts, and published methods on a modern MLLM (2026-11-13)

Concern (user, 2026-11-13): in the main table the strongest label-free baseline is a plain VLM (Qwen2.5-VL-7B
scoring each window), not a published method. Two steps, in the order the user set:

1. **Naive-prompt rows (§1).** The old LLM / VLM rows asked with our own design: the nine policy rules of our Reader
   (`experiments/20261008_baselines/hate_query.md` §1), our question wording, and, in the VLM row, our reading
   instruction (`READER_PRAG`). A naive baseline must ask a plain question. Rerun them that way, with the VLM row on
   Qwen3-VL-8B (the backbone of our method; user, 2026-11-13), then look at which label-free row is best. If the
   published rows already beat the naive rows, step 2 may not be needed (user).
2. **Published methods on Qwen3-VL-8B (§2), paused.** Re-run published training-free methods with the same MLLM and
   both modalities. Code for MARS (ICASSP 2026) is written (`mars_q3vl.py`) and has not been run.

Host for code and evaluation: uoa-lab1 (sc474397). Every number here is computed on the test split.

## 1. Naive-prompt LLM / VLM rows

Mechanism hypothesis: none; comparison rows.

**What the model sees** (`naive_prompt.py`; exact chat-template text in `runs/20261113_lf_baselines/naive/<arm>/<DS>/prompt.txt`).
No system message beyond the template default, no rules, no definition, no reading instruction, no context outside
the unit.

| arm | model | unit | input | question |
|---|---|---|---|---|
| `vlm` (table row) | Qwen/Qwen3-VL-8B-Instruct | fixed 8 s window | the window's 1 fps frames (at most 8, longer side 640) + `Transcript: <words>` | `Is this video clip hateful? Answer "Yes" or "No".` |
| `vlm_text` | Qwen/Qwen3-VL-8B-Instruct | fixed 8 s window | `Transcript: <words>` only (the input of the old Qwen2.5-VL row) | same |
| `llm` (table row) | Qwen/Qwen3-8B, thinking off | Whisper large-v3 segment (as the old Qwen3-8B row) | `Transcript: <segment text>` | `Is this speech hateful? Answer "Yes" or "No".` |

- Words: Whisper large-v3 segments (`src/lf_baselines.py` `transcript_segments`, untimed chunks kept), sliced
  proportionally to the window (`src/video_inputs.py` `window_text`); `(no speech)` when empty. The forced-aligned
  words of our Reader are not used.
- Score: log P(Yes) − log P(No) at the first answer token, from vLLM's top-20 log-probabilities (all case and space
  variants added; a side outside the top 20 is bounded by the smallest top-20 value and counted as `bounded`).
  Greedy, one token. vLLM 0.11.0 (`.cache/envs/vllm_q3vl`), bf16, prefix caching and the multimodal cache off, the
  multimodal hasher disabled (CLAUDE.md hash ban).
- 4 fps: window arms, a frame takes its window's score; `llm`, the maximum over the segments containing the frame
  centre, and outside every segment the score of the same question with `(no speech)` (the old row's rule F1).
- Fallback F2 (whole-video failure, e.g. no decodable frame): the corpus median frame score; more than 1 % of the
  cohort stops the corpus (`src/lf_baselines.py`).
- Nothing is selected on labels; the prompts above were fixed before any run.

**Comparison rule (declared before the runs).** For each corpus and metric, the best label-free cell is the maximum
over the label-free rows of the main table with the old LLM / VLM rows replaced by `llm` and `vlm`. The concern is
answered by the existing table if a published row holds that maximum on most cells; otherwise step 2 runs.

**How to run.**
```bash
# smoke (2 videos per arm, outside the run root)
LIMIT=2 LF_RUN_ROOT=$PWD/runs/20261113_lf_baselines/_smoke sbatch --export=ALL experiments/20261113_lf_baselines/launch/naive_lab1.sbatch vlm:HateMM vlm_text:HateMM llm:HateMM
# full runs (uoa-lab1 / uoa-lab2, after git pull), then on uoa-lab1:
sbatch experiments/20261113_lf_baselines/launch/naive_lab1.sbatch <arm>:<dataset> ...
python experiments/20261113_lf_baselines/naive_prompt.py raster --arm <arm> --dataset <dataset>   # HateVideo env
```

**Runs.** Code commit 3fcedae (2026-11-13). uoa-lab1 (sc474397) Slurm 2237: `vlm` DeHate, `llm` all three corpora;
uoa-lab2 (sc474399) Slurm 2239: `vlm` HateMM / HateClipSeg, `vlm_text` all three (Slurm 2238 failed at model load:
lab2's repository cache held a dangling link for Qwen3-VL-8B; the link was replaced by a copy of lab1's model
directory; log `slurm_2238_failed_no_weights.out`). Outputs copied back to uoa-lab1 with `rsync -a`; `raster` and the
evaluator ran on uoa-lab1. Every cohort video was scored (no F2, no bounded read); GPU time: `vlm` 9 / 9 / 38 min,
`vlm_text` 1 / 1 / 4 min, `llm` 1 / 1 / 3 min (HateMM / HateClipSeg / DeHate). A smoke run with vLLM log-probabilities
was discarded before the full runs: vLLM reads them from bf16 logits, which rounds the scores to steps of 0.25 and
creates ties; the reported runs use the fp32 output layer, as the old rows.

**Results** (pooled ROC / pooled PR / within, %; `python experiments/20261113_lf_baselines/compare.py`, which reads
the `metrics.json` files; table in `runs/20261113_lf_baselines/naive/compare.md`). Published rows are the variants
in the main table (`runs/20261008_baselines/<method>_av/<DS>/metrics.json`).

| row | HateMM | HateClipSeg | DeHate |
|---|---|---|---|
| best published cell (method) | 73.0 (LAVAD) / 50.2 (LAVAD) / 63.3 (VideoMind) | 63.6 (PRISM) / 61.0 (VideoMind) / 57.6 (PRISM) | 60.8 (LAVAD) / 12.4 (LAVAD) / 57.1 (VideoMind) |
| old Qwen3-8B, our rules | 75.4 / 55.1 / 56.7 | 54.0 / 55.2 / 52.8 | 61.6 / 13.3 / 56.4 |
| old Qwen2.5-VL-7B, our rules, transcript | 77.0 / 57.0 / 64.1 | 61.8 / 61.7 / 56.6 | 65.2 / 14.4 / 62.3 |
| **`llm`** Qwen3-8B, plain question | 78.5 / 56.4 / 57.8 | 55.4 / 55.6 / 52.0 | 63.7 / 13.5 / 57.2 |
| **`vlm`** Qwen3-VL-8B, plain question, frames + transcript | 81.0 / 55.3 / 61.8 | 70.6 / 67.4 / 61.2 | 69.1 / 17.7 / 60.5 |
| `vlm_text` same, transcript only | 76.3 / 56.0 / 63.1 | 61.2 / 60.3 / 57.4 | 65.2 / 15.2 / 61.2 |
| current method (main table) | 89.9 / 69.2 / 78.0 | 73.3 / 69.0 / 68.7 | 69.4 / 16.4 / 66.0 |

- By the comparison rule the concern stands: a new naive row holds the best label-free cell on 8 of 9 cells; the
  exception is HateMM within (VideoMind 63.3 against `vlm` 61.8).
- Our wording was not what made the naive rows strong. With the same input (transcript only), the plain question
  gives about the same as the old rules prompt (`vlm_text` vs the old Qwen2.5-VL row; different model), and the plain
  question raises Qwen3-8B on pooled HateMM. The `vlm` row is strong because the model is strong and sees the frames:
  frames add 9.4 / 7.1 pooled ROC / PR on HateClipSeg and 3.9 / 2.5 on DeHate over `vlm_text`.
- Consequence for the main table if `vlm` replaces the old VLM row: the margin of the current method over the best
  label-free cell shrinks to +8.9 / +12.8 / +14.7 (HateMM), +2.7 / +1.6 / +7.5 (HateClipSeg), +0.3 / −1.3 / +5.5
  (DeHate). On DeHate pooled PR the naive row is above the current method (17.7 vs 16.4), and pooled ROC is level.
- The published rows also carry our nine rules (LAVAD-AV and EventVAD-AV put them in their scoring prompt, PRISM in
  its description pools; `experiments/20261008_baselines/hate_query.md`). Not changed here.

## 2. Published methods on Qwen3-VL-8B (paused)

Planned arms, not run: MARS (ICASSP 2026, hate-specific, four VLM calls per window; `mars_q3vl.py`), EventVAD
(ACM MM 2025) with its existing event segmentation and Qwen3-VL-8B as the event scorer, LAVAD (CVPR 2024) with
Qwen3-VL-8B as captioner and LLM, and possibly LELA (arXiv 2602.09637, the only training-free hate localiser; no
code). Shared engine and read-outs: `vllm_common.py`. Waits for the §1 result and the user's decision.
