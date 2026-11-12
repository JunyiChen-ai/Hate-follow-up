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

**Results.** Pending.

## 2. Published methods on Qwen3-VL-8B (paused)

Planned arms, not run: MARS (ICASSP 2026, hate-specific, four VLM calls per window; `mars_q3vl.py`), EventVAD
(ACM MM 2025) with its existing event segmentation and Qwen3-VL-8B as the event scorer, LAVAD (CVPR 2024) with
Qwen3-VL-8B as captioner and LLM, and possibly LELA (arXiv 2602.09637, the only training-free hate localiser; no
code). Shared engine and read-outs: `vllm_common.py`. Waits for the §1 result and the user's decision.
