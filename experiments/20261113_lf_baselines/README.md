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

## 3. More published methods on Qwen3-VL-8B (gate 1, user 2026-11-14)

Gate (user, 2026-11-14; `experiments/20261114_exp_gates/README.md` §1): the best published label-free cell must be
>= the naive Qwen3-VL-8B row on at least 7 of 9 cells, the rest at most 1 point below. §2 left 3 of 9. Mechanism
hypothesis: none; comparison rows. Same shared rules as §2 (Qwen3-VL-8B for every call, frames at the naive row's
image size, Whisper large-v3 words, fp32 read-out of the method's own answer, 4 fps, F2, single evaluator).
Allowed per the brief: no component of our method goes into a baseline; the naive row is not renamed as a published
method; labels may be used only to choose a configuration, and every configuration tried is listed.

The gate table is `experiments/20261113_lf_baselines/gate.py` → `runs/20261113_lf_baselines/gate/gate.{md,json}`. It
reports four row sets: A all published rows and every configuration tried; B only configurations fixed before the
run; C only peer-reviewed methods; D peer-reviewed and fixed before the run. A configuration counts as one row only
when it exists for all three corpora (no per-corpus choice).

### 3.1 Probe-VAD (arXiv 2609.17211; preprint under review, not peer-reviewed)

**Excluded as a baseline (user, 2026-11-15). Kept as a record; the probing method is tested inside our method,
`experiments/20261114_exp_gates/README.md` §10.** (User: "不要这个 Probe-VAD 但是记下它的方法 它可以用来refine我们的method".)
`gate.py` no longer reads its rows; its runs stay on disk under `runs/20261113_lf_baselines/probevad/`.

Paper: Gu, Zhao, Guo, Zhong, Zhang, Lyu, Zhao, Xie, Shan, "Probe-VAD: Ordinal Likelihood Probing for Training-Free
Video Anomaly Detection", https://arxiv.org/abs/2609.17211 (v1, 15 Sep 2026; arXiv Comments field: "Under Review";
no venue). Code: https://github.com/yvestine/COVAS-VAD (now served as github.com/yvestine/Probe-VAD; MIT). Checked
2026-11-15 by opening the arXiv abstract page and the GitHub page (both HTTP 200; the abstract links the repository,
the repository README links the arXiv page). The local clone `third_party/COVAS-VAD` is at commit dafcdd7d
(2026-11-05). Status: **preprint under review, not peer-reviewed**; it counts only in gate-table sets A and B.

Release github.com/yvestine/COVAS-VAD (`docs/METHOD.md`, `research_code/src/prompt_configs.py`). Training-free: a
frozen video LLM answers ten ordered YES / NO questions per clip, "is the severity at least tau", tau = 0.1 … 1.0;
p_tau = softmax over the YES / NO logits; the ten values are projected onto a non-increasing profile (pool adjacent
violators); clip score = their mean; Gaussian smoothing over the clip sequence; a clip's score goes to the frames from
its centre to the next centre. Code `probevad_q3vl.py`; the ten questions share one encoded prefix per clip.

Kept: clip window [c − 5 s, c + 5 s] clipped to the video, 2 fps sampling with at most 10 frames, the ten thresholds,
single-token YES / NO at T = 1, the projection, the mean, Gaussian smoothing (scipy default, reflect), the frame
expansion rule.

Every deviation from the release (all fixed before any run):
1. Backbone Qwen3-VL-8B (release default VideoLLaMA3-7B; the paper also reports Qwen3-VL-8B); frames at the naive
   row's image size; YES / NO logits from an fp32 output layer.
2. The clip's Whisper words are added to the user turn (`Transcript: <words>`, `(no speech)` when empty), as in every
   row of §1–§2. The release sees frames only.
3. System prompt and question: the release's sentences with the surveillance anomaly-severity scale replaced by a
   hatefulness scale, sentence by sentence (exact text `runs/20261113_lf_baselines/probevad/<DS>/s4/prompt.txt`).
4. Clip stride 4 s (release: one clip per 16 frames, about 0.53 s at 30 fps), for cost: about 7.5 times fewer clips.
5. The smoothing sigma is kept in seconds (release: 10 clips of 16 frames = 5.33 s at 30 fps), i.e. 1.33 clips at
   the 4 s stride.
6. The ten questions are scored in one batched pass over a shared prefix cache. In-job check on the first video of
   each corpus: largest |difference| of a YES − NO logit difference against a full forward per question 0.37–0.42
   (logit units), from bf16 attention; not a change of method.

Configurations (all three corpora each; raster `python experiments/20261113_lf_baselines/probevad_q3vl.py raster
--dataset <DS> --sigma 5.33333 0`):

| configuration | status | HateMM ROC / PR / within | HateClipSeg | DeHate |
|---|---|---|---|---|
| `s4/sig5.33333` (release sigma) | fixed before the run (gate default) | 84.1 / 58.3 / 62.4 | 72.3 / 69.6 / 63.9 | 71.5 / 17.6 / 64.0 |
| `s4/sig0` (no smoothing) | extra configuration, counted as test-selected | 81.0 / 57.6 / 62.7 | 70.8 / 68.0 / 61.6 | 69.3 / 15.7 / 63.7 |

Runs: uoa-lab3 (sc474398) Slurm 2302, code e53a25a (2026-11-14), HateVLM env; GPU time HateMM 26 min, HateClipSeg
24 min, DeHate 98 min (7422 / 7128 / 28182 clips). Every cohort video scored, no F2. Before the projection, 78–86 % of
clips have a YES probability that rises somewhere along the thresholds (`raster_stats.json`
`clips_with_order_violation`); the projection is the release's fix for this. Outputs copied to uoa-lab1 with
`rsync -a`; raster and evaluator on uoa-lab1.

### 3.2 CoReVAD (ICPR 2026; peer-reviewed)

Paper: Lim, Hur, "CoReVAD: A Contextual Reasoning Framework for Training-Free Video Anomaly Detection",
https://arxiv.org/abs/2605.23116 (v1, 22 May 2026; arXiv Comments field: "Accepted to ICPR 2026"). Code:
https://github.com/Muk-00/CoReVAD (README: "(ICPR 2026)"). Checked 2026-11-15 by opening the arXiv abstract page and
the GitHub page (both HTTP 200; the abstract links the repository). The ICPR 2026 proceedings page itself was not
opened; acceptance rests on the authors' arXiv comment and README. Local clone `third_party/CoReVAD` at commit 76ea2a0
(2026-04-28).

Release github.com/Muk-00/CoReVAD (`src/ucf/LRC.py`, `src/ucf/eval_ucf.py`, `P_VLM_format.txt`). One frozen VLM per
segment writes a binary decision line and a one-sentence description; the decisions are refined with frozen CLIP
ViT-B/16 features: (1) Local Response Cleaning (a segment takes the response of the neighbour in [i − 1, i + 1] whose
text embedding is closest to its mean frame embedding); (2) softmax refinement over all segments' descriptions,
tau = 0.05; (3) Gaussian smoothing, kernel 9, sigma 5 segments; (4) multiplication by a Gaussian position weight
centred on the middle of the video (s.d. half its length). Code `corevad_q3vl.py`.

Every deviation from the release (all fixed before any run):
1. Backbone Qwen3-VL-8B instead of InternVL2-8B; frames at the naive row's image size.
2. The segment's Whisper words are added to the prompt input (`Transcript: <words>`).
3. Prompt wording: "anomaly" replaced by "hateful content"; decision lines "Hateful scenes." / "Normal scenes."
4. Segment length 8 s with 8 frames (1 fps; the naive row's unit) instead of 30 frames (about 1 s), for cost.
5. The assistant turn starts with the opening code fence, so the decision is the first token written.
6. Primary read-out `prob` = P(Hateful) from the fp32 output layer at the decision position (§2 shared rule); the
   release's 0 / 1 decision is kept as `label` (declared before the run as the check read-out).
7. CLIP frame embedding of a segment = mean over its 8 sampled frames (the release averages all frames).
8. CLIP features are computed with the projection heads applied to the pooled outputs
   (`text_projection(text_model(...).pooler_output)`, `visual_projection(vision_model(...).pooler_output)`): the
   installed transformers 5.15.1 `get_*_features` returns an output object, not the projected tensor. Same features as
   the release.

Configurations: `u8/prob_lrc1_tau0.05_g9-5_pos1` (gate default, release constants) and
`u8/label_lrc1_tau0.05_g9-5_pos1` (the release's hard decision; counted as an extra configuration by `gate.py`).
Smoke: uoa-lab1 Slurm 2303 (2 HateMM videos, 17 segments; every answer had both lines). Full run: uoa-lab1 Slurm
2305, code 10aec17, host sc474397, HateVLM env (vLLM generation, then fp32 HF read-out). GPU time (generation +
read-out): HateMM 15 + 20 min, HateClipSeg 16 + 20 min, DeHate 62 + 95 min (3768 / 3591 / 14375 segments). Every
cohort video scored (215 / 118 / 1151), every answer had a decision line, no video truncated (`raster_stats.json`).
Raster and evaluator on uoa-lab1 (`corevad_q3vl.py raster`); metrics from the evaluator's `metrics.json` under
`runs/20261113_lf_baselines/corevad/<DS>/u8/<config>/`.

| configuration | status | HateMM ROC / PR / within | HateClipSeg | DeHate |
|---|---|---|---|---|
| `u8/prob_lrc1_tau0.05_g9-5_pos1` | fixed before the run (gate default) | 76.5 / 44.5 / 72.9 | 70.2 / 66.8 / 57.1 | 63.9 / 12.1 / 58.1 |
| `u8/label_lrc1_tau0.05_g9-5_pos1` | extra configuration, counted as test-selected | 76.2 / 44.1 / 71.7 | 70.1 / 66.6 / 58.2 | 62.9 / 11.8 / 55.5 |

Against the naive Qwen3-VL-8B row (81.0 / 55.3 / 61.8, 70.6 / 67.4 / 61.2, 69.1 / 17.7 / 60.5) CoReVAD is higher only
on HateMM within (+11.2); HateClipSeg pooled ROC / PR are 0.3 / 0.6 points below. Gate 1 after adding CoReVAD
(`runs/20261113_lf_baselines/gate/gate.md`): 3 of 9 cells in all four row sets (A–D), largest shortfall 2.53 points
(HateMM pooled PR); gate 1 fails. CoReVAD supplies the best published cell on HateMM within and HateClipSeg pooled
ROC / PR.

### 3.3 AnyAnomaly (WACV 2026; method and configuration written before the run)

Paper: Ahn, Jo, Lee, Kwon, Hong, Park, "AnyAnomaly: Zero-Shot Customizable Video Anomaly Detection with LVLM", WACV
2026, arXiv 2503.04504; code github.com/SkiddieAhn/Paper-AnyAnomaly (local clone `third_party/AnyAnomaly`, commit
ec7e9fc, 2026-02-27; README: "Our paper has been accepted to WACV 2026", links the CVF open-access page). Venue taken
from the release README; the arXiv and CVF pages were not opened in this session. Chosen by the planning session
(brief 4 H4).

Method (release `vad_proposed_Qwen_vllm.py`): training-free customizable VAD. Per clip and a user-defined anomaly text,
CLIP (ViT-B/32) builds three images and an LVLM scores each on 0–1 ("does the image include <keyword>"): the key frame
(KFS: CLIP similarity, then grouping), a position-context image (key frame masked by multi-scale WinCLIP patch
similarity) and a temporal-context 2×2 grid (the patch most similar to the keyword across four key frames). Clip score
= .6 key + .3 position + .1 temporal; Gaussian smoothing, sigma 15 frames; min-max normalisation. Code
`anyanomaly_q3vl.py`.

Configuration fixed before the run (the release's constants): keyword "hateful content"; KFS 4; avenue scales (the
release's default dataset: image 240, patches 48 / 80 / 240, stride = size); template ensemble on; alpha .6, beta .3,
gamma .1; sigma 15 frames. The release's default `--grid_search True` searches sigma and the three weights on the test
labels; it is not used.

Every deviation from the release:
1. Backbone Qwen3-VL-8B (release default Qwen2.5-VL-3B-Instruct); greedy decoding, at most 50 new tokens (release:
   temperature .1, top_p .001, repetition penalty 1.05, 50 tokens). Images resized by the naive row's rule
   (smart_resize, src/mllm_judge.py) so generation and the fp32 read-out see the same pixels.
2. The clip's Whisper words are added before each instruction (`Transcript: <words>`; `(no speech)` when empty).
3. Clip = 8 s window with 24 frames from a 3 fps decode (release: 24 consecutive native frames, about 1 s at 25 fps),
   for cost; a short last window uses the frames it has (the release drops an incomplete last clip).
4. Class adaptation off. The release softmaxes the patch similarity over the dataset's anomaly classes; with one
   keyword the softmax is 1 everywhere and the attention normalisation divides by zero. The release's
   `class_adaption False` path is used.
5. CLIP ViT-B/32 from transformers (openai/clip-vit-base-patch32, same weights as the `clip` package); KFS
   preprocessing as `clip.load` (resize 224 bicubic, centre crop, CLIP normalisation); patches go to the image
   encoder unnormalised, as in the release.
6. Read-outs: `prob` (primary) = expected score from the fp32 output layer at the position where the answer writes its
   first number (if there is none, after the answer + "\nScore: "); `text` (check) = the release's extract_score
   (first number, 0 if none).
7. Smoothing per video on the 4 fps curve with sigma 0.6 s (15 frames at 25 fps; release smooths the concatenated
   test set); global min-max omitted (monotone, no effect on any metric).
8. The two CLIP-built images per window are stored as JPEG (quality 95) under the run directory between generation and
   read-out (both stages read the same files) and removed after the raster (`clean`).

Cost: 3 generations + 3 read-out forwards per 8 s window, plus about 200 CLIP image encodings per window.
Run plan: smoke (2 HateMM videos), then the three corpora on uoa-lab1 after the ordinal read (Slurm 2312):
`sbatch experiments/20261113_lf_baselines/launch/q3vl_lab1.sbatch anyanomaly HateMM HateClipSeg DeHate`, raster
`python experiments/20261113_lf_baselines/anyanomaly_q3vl.py raster --dataset <DS>`, gate row in `gate.py`
(`anyanomaly`, default `u8/prob`).

**Result.** Smoke: uoa-lab1 Slurm 2317 (2 HateMM videos, 17 windows, every answer had a number). Full run: uoa-lab1
(sc474397) Slurm 2321, code 7fed039, HateVLM env; 7 h 37 min. GPU time (generation + read-out): HateMM 66 + 19 min,
HateClipSeg 54 + 18 min, DeHate 232 + 66 min (3768 / 3591 / 14375 windows). Every cohort video scored (215 / 118 /
1151), no answer without a number; answers that reached the 50-token limit: 68 / 94 / 257 generations
(`raster_stats.json` `finish_length`; their first number is still read). Context images removed after the raster
(`clean`). Metrics from the evaluator's `metrics.json` under `runs/20261113_lf_baselines/anyanomaly/<DS>/u8/<config>/`.

| configuration | status | HateMM ROC / PR / within | HateClipSeg | DeHate |
|---|---|---|---|---|
| `u8/prob` | fixed before the run (gate default) | 79.1 / 53.7 / 63.3 | 64.9 / 61.8 / 58.2 | 66.4 / 15.3 / 60.2 |
| `u8/text` | release read-out, counted as test-selected | 73.7 / 47.0 / 58.3 | 62.1 / 57.5 / 55.9 | 60.4 / 12.4 / 53.2 |

Against the naive Qwen3-VL-8B row AnyAnomaly is higher only on HateMM within (+1.5). It becomes the best published
row on HateMM pooled ROC (79.06, MARS 79.05) and pooled PR (53.66, MARS 52.80), both still below naive (−1.93 /
−1.67). Gate 1 after adding AnyAnomaly (`runs/20261113_lf_baselines/gate/gate.md`): 3 of 9 cells in all four row sets,
largest shortfall 2.51 points (HateClipSeg within); gate 1 fails.
