# Label-free baselines: naive prompts, and published methods on a modern MLLM (2026-11-13)

Concern (user, 2026-11-13): in the main table the strongest label-free baseline is a plain VLM (Qwen2.5-VL-7B
scoring each window), not a published method. Two steps, in the order the user set:

1. **Naive-prompt rows (§1).** The old LLM / VLM rows asked with our own design: the nine policy rules of our Reader
   (`experiments/20261008_baselines/hate_query.md` §1), our question wording, and, in the VLM row, our reading
   instruction (`READER_PRAG`). A naive baseline must ask a plain question. Rerun them that way, with the VLM row on
   Qwen3-VL-8B (the backbone of our method; user, 2026-11-13), then look at which label-free row is best. If the
   published rows already beat the naive rows, step 2 may not be needed (user).
2. **Published methods on Qwen3-VL-8B (§2).** The naive rows still held the best cell on 8 of 9 cells, so the user
   asked to give the published methods a better backbone (2026-11-13): re-run published training-free methods with
   Qwen3-VL-8B, the backbone of the naive VLM row and of our method, and both modalities. MARS and EventVAD done;
   both stay below the naive Qwen3-VL-8B row on most cells. LAVAD not run (decision pending).

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

## 2. Published methods on Qwen3-VL-8B

Mechanism hypothesis: none; comparison rows. Question: does a published training-free method, given the same MLLM as
the naive VLM row, beat that row?

**Shared rules** (`vllm_common.py`):
- MLLM: Qwen/Qwen3-VL-8B-Instruct for every call; frames resized to the image size of the naive row
  (`src/mllm_judge.py` MAX_PIXELS, `smart_resize` factor 32), so both see the same pixels.
- Words: Whisper large-v3, as the naive rows (not our forced-aligned words).
- Hate definition: each method keeps its own prompt; where it asks for an anomaly or a target concept, one plain
  sentence ("An anomaly is hateful content."), not the nine rules of our Reader.
- Two steps. (a) Generation with vLLM 0.11.0, greedy, the method's own text. Prefix caching and the multimodal cache
  off; vLLM's multimodal hasher replaced by a function that returns the request id we supply and raises on media
  (CLAUDE.md hash ban). (b) Read-out of the method's own final answer with one HF forward and an fp32 output layer on
  the prompt plus the generated text, because vLLM reads log-probabilities from bf16 logits (steps of 0.25, ties).
  The read-out uses the same number the method writes, at the position where it writes it; the method's own hard
  output is kept as a second, check read-out.
- 4 fps, F2 and the evaluator as §1.

| method | unit | calls per unit | primary read-out (`prob`) | check read-out |
|---|---|---|---|---|
| MARS (ICASSP 2026, `mars_q3vl.py`) | fixed 8 s window, 1 fps frames (8) + window words | 4 (description; evidence if hateful; evidence if not; decision) | log P("1") − log P("0") where turn 4 writes `"label":` | the 0/1 label (`label`) |
| EventVAD (ACM MM 2025, `eventvad_q3vl.py`) | events of the main-table EventVAD segmentation (`runs/20261008_baselines/eventvad/<DS>/events.jsonl`), 16 frames + event words (at most 512 tokens) | 1 (chain-of-thought anomaly score) | expected score at "\nTherefore, the final score is " after its answer (first digit, and the decimal after "0.") | the number its answer states (`text`) |

**How to run.**
```bash
# smoke (2 videos), then full runs on uoa-lab1 / uoa-lab2 after git pull
LIMIT=2 LF_RUN_ROOT=$PWD/runs/20261113_lf_baselines/_smoke sbatch --export=ALL experiments/20261113_lf_baselines/launch/mars_lab1.sbatch HateMM
sbatch experiments/20261113_lf_baselines/launch/mars_lab1.sbatch <dataset> ...        # or eventvad_lab1.sbatch
python experiments/20261113_lf_baselines/mars_q3vl.py raster --dataset <dataset>        # HateVideo env, uoa-lab1
python experiments/20261113_lf_baselines/compare.py
```

**Runs.**
- MARS: uoa-lab1 (sc474397) Slurm 2244, DeHate; uoa-lab2 (sc474399) Slurm 2245, HateMM then HateClipSeg. Code
  commit 162832f; the lab2 checkout was pulled to cd6a05a during the job, so HateClipSeg generation and every
  read-out ran under cd6a05a, which only adds functions used by EventVAD (`git diff 162832f cd6a05a`). GPU time,
  generation + read-out: HateMM 64 + 13 min, HateClipSeg 62 + 12 min, DeHate 250 + 51 min. Outputs copied back with
  `rsync -a`; raster and evaluator on uoa-lab1. Every window got a turn-4 label (no read-out fallback, no F2);
  turns that hit 2048 tokens: 95 / 59 / 174 windows; turn-1 JSON parse failures 86 / 50 / 168 (MARS passes the raw
  text on, as the release does).
- EventVAD: smoke Slurm 2246 (uoa-lab2, 2 HateMM videos, all events scored, answers end with a stated score). Full
  runs Slurm 2259 (uoa-lab1, DeHate, 16087 events) and 2260 (uoa-lab2, HateMM 4421 + HateClipSeg 4197 events),
  code cd6a05a (the DeHate read-out started under 483f785, which changes only this README and `compare.py`). GPU
  time, generation + read-out: HateMM 38 + 19 min, HateClipSeg 38 + 19 min, DeHate 143 + 71 min. Every event got a
  read-out (no F2); answers that hit 2048 tokens 47 / 55 / 156 events; answers with no stated score 45 / 57 / 154
  (0 in the `text` read-out, as the main-table rule C).

**Results** (pooled ROC / pooled PR / within, %; `runs/20261113_lf_baselines/naive/compare.md`).

| row | HateMM | HateClipSeg | DeHate |
|---|---|---|---|
| best published cell, main table | 73.0 / 50.2 / 63.3 | 63.6 / 61.0 / 57.6 | 60.8 / 12.4 / 57.1 |
| **MARS, Qwen3-VL-8B** (`prob`) | 79.0 / 52.8 / 60.5 | 67.4 / 63.9 / 58.7 | 69.1 / 16.4 / 60.8 |
| MARS, Qwen3-VL-8B, 0/1 label | 73.7 / 42.5 / 58.5 | 61.1 / 54.9 / 55.3 | 62.5 / 11.4 / 55.2 |
| **EventVAD, Qwen3-VL-8B** (`prob`) | 75.7 / 50.8 / 57.5 | 65.3 / 61.3 / 56.4 | 63.3 / 13.9 / 60.8 |
| EventVAD, Qwen3-VL-8B, stated score | 73.5 / 43.9 / 59.0 | 62.3 / 55.8 / 56.0 | 62.6 / 11.5 / 59.3 |
| naive `vlm` (§1) | 81.0 / 55.3 / 61.8 | 70.6 / 67.4 / 61.2 | 69.1 / 17.7 / 60.5 |
| current method (main table) | 89.9 / 69.2 / 78.0 | 73.3 / 69.0 / 68.7 | 69.4 / 16.4 / 66.0 |

- MARS on Qwen3-VL-8B is the best published row on 8 of 9 cells (all but HateMM within, still VideoMind).
- It stays below the naive `vlm` row on 6 of 9 cells. DeHate pooled ROC is level (69.14 vs 69.09) and DeHate within
  is above (60.8 vs 60.5). With the same model, frames and words, the decision at the end of the four-turn chain
  ranks windows worse than the answer to the plain question, at four times the calls.
- Its own 0/1 label is 2–10 points lower than the read-out at the same position; the read-out is the row to report.
- EventVAD on Qwen3-VL-8B is below MARS on 8 of 9 cells (DeHate within level, 60.83 vs 60.77) and below the naive
  `vlm` row on 8 of 9. It beats its main-table row (VideoLLaMA2.1-7B scorer) on all 9 cells, e.g. DeHate pooled ROC 63.3
  vs 53.5.
- Taken together: with Qwen3-VL-8B for every published method, the best label-free cell belongs to the naive `vlm`
  row on 6 of 9 cells and to a published row on 3 (HateMM within VideoMind 63.3; DeHate pooled ROC MARS 69.14 vs
  69.09, a tie; DeHate within EventVAD / MARS 60.8 vs 60.5). Both re-run methods add steps on top of the same model
  (four turns, or an event segmentation and a reasoning chain); on this task those steps rank windows worse than the
  model's direct answer to one plain question.
- LAVAD (CVPR 2024, captions + LLM scoring + refinement) on Qwen3-VL-8B: not started; five stages to rewrite
  (`experiments/20261008_baselines/lavad/lavad_chain.py`, copied, not imported). LELA (arXiv 2602.09637,
  training-free hate localiser) has no code.
