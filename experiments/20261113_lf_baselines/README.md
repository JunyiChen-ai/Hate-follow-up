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

### 1.1 `vlm_tax` on HateClipSeg (planning-session brief 5, item K3, 2026-11-16; reference row only)

Same as `vlm` (Qwen3-VL-8B, frames + transcript, 8 s windows, fp32 Yes/No read-out), with the question replaced by the
HateClipSeg category list: "Does this video clip contain content of any of these kinds: hateful, insulting, sexual,
violence, or self-harm? Answer \"Yes\" or \"No\"." (`naive_prompt.py`, arm `vlm_tax`, commit 5af9561). The prompt was
written knowing that the HateClipSeg ground truth marks offensive content of these kinds, so the row is
development-selected and is a reference only: it is not in `gate.py`, and the frozen `vlm` row is unchanged.

Run: uoa-lab1 (sc474397), Slurm 2348, 9 min GPU; raster on uoa-lab1. 118 test videos, 3591 windows, no failed videos.
Metrics: `runs/20261113_lf_baselines/naive/vlm_tax/HateClipSeg/metrics.json`.

| row (HateClipSeg) | pooled ROC | pooled PR | within |
|---|---|---|---|
| `vlm`, plain hateful question | 70.6 | 67.4 | 61.2 |
| `vlm_tax`, category list | 73.7 | 70.5 | 63.9 |
| current method (main table) | 73.3 | 69.0 | 68.7 |

- Naming the HateClipSeg categories adds 3.2 / 3.1 / 2.7 over the plain question.
- With this prompt the naive row is above the current method on HateClipSeg pooled ROC (+0.4) and pooled PR (+1.5);
  within stays 4.8 below.

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

#### 3.2.1 Hyperparameter grid (planning-session brief 5, item K1, 2026-11-15; test-selected)

CPU only, on the stored read-outs (no new inference): `corevad_q3vl.py raster` with the `prob` read-out, LRC on,
tau ∈ {0.05, 0.1, 0.5, 1.0} (softmax refinement), Gaussian (size, sigma) ∈ {(9, 5) release, (5, 1), (3, 1), none
(`g1-1`, size 1 = no smoothing)}, position weight ∈ {on (release), off}: 32 configurations, every one on all three
corpora. tau / size / sigma are command-line arguments of the release (`src/ucf/eval_ucf.py`); turning the position
weight off is not a release option (deviation 9 below). The selection rule was fixed by the planning session before
the grid was read: the one configuration, the same on all three corpora, with the most cells ≥ the naive row, ties
by the smallest largest shortfall; reported once among release-option configurations (position weight on) and once
among all 32. Every number here was chosen by reading test metrics: **test-selected (development-selected)**.

Deviation 9 (K1 only): rows named `..._pos0` skip step 4 (the multiplication by the position weight). The release has
no switch for this; such rows carry "position prior off (not a release option)" in their gate.py name.

Run: uoa-lab1 (sc474397), CPU, `setsid nohup`, 2026-11-15, commit 5af9561, script and logs in
`runs/20261113_lf_baselines/corevad_k1/` (`run.sh`, `raster_<DS>.log`, `select.py`, `grid.md`, `selected.json`).
The release-default configuration was rastered again in the same call; its metrics.json equals the earlier one on all
three corpora. Metrics from the evaluator's metrics.json under `runs/20261113_lf_baselines/corevad/<DS>/u8/<config>/`.

| configuration | HateMM ROC / PR / within | HateClipSeg | DeHate | cells >= naive | largest shortfall |
|---|---|---|---|---|---|
| naive Qwen3-VL-8B | 81.0 / 55.3 / 61.8 | 70.6 / 67.4 / 61.2 | 69.1 / 17.7 / 60.5 | — | — |
| `prob_lrc1_tau0.05_g9-5_pos1` | 76.5 / 44.5 / 72.9 | 70.2 / 66.8 / 57.1 | 63.9 / 12.1 / 58.1 | 1 | 10.86 |
| `prob_lrc1_tau0.05_g9-5_pos0` | 76.7 / 49.5 / 72.0 | 70.5 / 67.5 / 56.3 | 64.1 / 12.6 / 55.2 | 2 | 5.79 |
| `prob_lrc1_tau0.05_g5-1_pos1` | 76.3 / 44.0 / 71.9 | 70.7 / 67.2 / 58.5 | 64.5 / 12.8 / 57.2 | 2 | 11.28 |
| `prob_lrc1_tau0.05_g5-1_pos0` | 76.5 / 50.3 / 70.8 | 70.7 / 67.9 / 58.0 | 64.7 / 13.7 / 56.6 | 3 | 5.06 |
| `prob_lrc1_tau0.05_g3-1_pos1` | 76.2 / 44.0 / 71.7 | 70.8 / 67.3 / 58.7 | 64.5 / 12.8 / 57.6 | 2 | 11.35 |
| `prob_lrc1_tau0.05_g3-1_pos0` **(selected, all 32)** | 76.5 / 50.7 / 68.7 | 70.7 / 67.9 / 57.7 | 64.8 / 13.7 / 57.1 | 3 | 4.59 |
| `prob_lrc1_tau0.05_g1-1_pos1` | 75.8 / 43.5 / 72.2 | 70.7 / 67.3 / 59.5 | 64.7 / 12.8 / 59.3 | 2 | 11.83 |
| `prob_lrc1_tau0.05_g1-1_pos0` | 76.0 / 49.5 / 54.9 | 70.5 / 67.2 / 55.4 | 64.8 / 13.6 / 54.2 | 0 | 6.83 |
| `prob_lrc1_tau0.1_g9-5_pos1` | 77.5 / 45.6 / 72.3 | 70.2 / 66.6 / 57.7 | 64.0 / 12.2 / 57.9 | 1 | 9.70 |
| `prob_lrc1_tau0.1_g9-5_pos0` | 77.7 / 50.7 / 72.2 | 70.4 / 67.0 / 56.2 | 64.2 / 12.6 / 54.5 | 1 | 6.03 |
| `prob_lrc1_tau0.1_g5-1_pos1` **(selected, release options)** | 77.3 / 45.2 / 71.7 | 70.6 / 67.0 / 58.2 | 64.6 / 12.9 / 56.4 | 2 | 10.10 |
| `prob_lrc1_tau0.1_g5-1_pos0` | 77.5 / 51.4 / 71.8 | 70.6 / 67.3 / 58.3 | 64.7 / 13.7 / 56.8 | 1 | 4.36 |
| `prob_lrc1_tau0.1_g3-1_pos1` | 77.3 / 45.2 / 71.8 | 70.6 / 67.1 / 58.3 | 64.7 / 12.9 / 56.8 | 2 | 10.15 |
| `prob_lrc1_tau0.1_g3-1_pos0` | 77.5 / 51.8 / 69.6 | 70.5 / 67.2 / 58.5 | 64.8 / 13.6 / 56.8 | 1 | 4.29 |
| `prob_lrc1_tau0.1_g1-1_pos1` | 76.9 / 44.7 / 72.3 | 70.6 / 67.1 / 58.8 | 64.7 / 12.9 / 59.1 | 2 | 10.64 |
| `prob_lrc1_tau0.1_g1-1_pos0` | 77.1 / 50.6 / 56.3 | 70.3 / 66.6 / 55.4 | 64.8 / 13.6 / 54.8 | 0 | 5.80 |
| `prob_lrc1_tau0.5_g9-5_pos1` | 78.5 / 46.8 / 71.9 | 69.8 / 66.1 / 57.7 | 64.2 / 12.3 / 57.6 | 1 | 8.52 |
| `prob_lrc1_tau0.5_g9-5_pos0` | 78.6 / 51.7 / 71.6 | 70.1 / 66.7 / 56.7 | 64.4 / 12.7 / 53.9 | 1 | 6.62 |
| `prob_lrc1_tau0.5_g5-1_pos1` | 78.3 / 46.4 / 71.3 | 70.1 / 66.4 / 57.7 | 64.7 / 13.0 / 55.5 | 1 | 8.94 |
| `prob_lrc1_tau0.5_g5-1_pos0` | 78.4 / 52.2 / 72.8 | 70.2 / 66.8 / 58.5 | 64.8 / 13.6 / 55.9 | 1 | 4.64 |
| `prob_lrc1_tau0.5_g3-1_pos1` | 78.3 / 46.3 / 71.4 | 70.2 / 66.5 / 57.7 | 64.8 / 12.9 / 55.8 | 1 | 9.01 |
| `prob_lrc1_tau0.5_g3-1_pos0` | 78.4 / 52.7 / 69.3 | 70.1 / 66.7 / 58.6 | 64.9 / 13.6 / 56.7 | 1 | 4.22 |
| `prob_lrc1_tau0.5_g1-1_pos1` | 77.9 / 45.8 / 71.8 | 70.1 / 66.4 / 57.8 | 64.8 / 13.0 / 58.2 | 1 | 9.52 |
| `prob_lrc1_tau0.5_g1-1_pos0` | 77.9 / 51.4 / 56.8 | 69.9 / 66.0 / 55.3 | 64.8 / 13.5 / 55.0 | 0 | 5.91 |
| `prob_lrc1_tau1_g9-5_pos1` | 78.6 / 47.0 / 71.8 | 69.8 / 66.0 / 57.6 | 64.2 / 12.3 / 57.5 | 1 | 8.34 |
| `prob_lrc1_tau1_g9-5_pos0` | 78.8 / 51.9 / 71.6 | 70.0 / 66.5 / 56.7 | 64.4 / 12.7 / 53.7 | 1 | 6.84 |
| `prob_lrc1_tau1_g5-1_pos1` | 78.4 / 46.6 / 71.2 | 70.1 / 66.3 / 57.6 | 64.8 / 13.0 / 55.4 | 1 | 8.76 |
| `prob_lrc1_tau1_g5-1_pos0` | 78.5 / 52.3 / 71.9 | 70.1 / 66.7 / 58.3 | 64.9 / 13.6 / 55.9 | 1 | 4.57 |
| `prob_lrc1_tau1_g3-1_pos1` | 78.4 / 46.5 / 71.2 | 70.1 / 66.3 / 57.6 | 64.8 / 12.9 / 55.8 | 1 | 8.84 |
| `prob_lrc1_tau1_g3-1_pos0` | 78.5 / 52.8 / 69.7 | 70.1 / 66.6 / 58.5 | 64.9 / 13.5 / 56.4 | 1 | 4.21 |
| `prob_lrc1_tau1_g1-1_pos1` | 78.1 / 46.0 / 71.6 | 70.0 / 66.3 / 57.6 | 64.9 / 13.0 / 58.0 | 1 | 9.35 |
| `prob_lrc1_tau1_g1-1_pos0` | 78.0 / 51.5 / 57.1 | 69.8 / 65.9 / 55.2 | 64.8 / 13.5 / 55.0 | 0 | 6.02 |

Selected among release options (position weight on): `prob_lrc1_tau0.1_g5-1_pos1` (2 cells, shortfall 10.10).
Selected among all 32: `prob_lrc1_tau0.05_g3-1_pos0` (3 cells, shortfall 4.59).

Gate 1 (`runs/20261113_lf_baselines/gate/gate.md`, regenerated; `gate.py` reads only the release default, the label
read-out and the two selected configurations of CoReVAD (`ONLY`); the two selected rows are test-selected candidates,
so they enter sets A and C and not B / D):
- A and C: 5 of 9 cells ≥ naive (was 3). The two new cells are HateClipSeg pooled ROC 70.72 (+0.16) and pooled PR
  67.90 (+0.46), both from `prob_lrc1_tau0.05_g3-1_pos0` (position prior off). Largest shortfall unchanged, 2.51
  (HateClipSeg within, MARS). Gate 1 still fails.
- With only the release-option selection added (`prob_lrc1_tau0.1_g5-1_pos1`, position weight on), A and C would have
  4 of 9 (HateClipSeg pooled ROC +0.02; pooled PR −0.39), largest shortfall 2.51.
- B and D unchanged: 3 of 9, largest shortfall 2.51.
- Remaining failing cells (A): HateMM pooled ROC −1.93 and PR −1.67 (AnyAnomaly), HateClipSeg within −2.51 (MARS),
  DeHate pooled PR −1.30 (MARS).
- Turning the position weight off raises HateMM pooled PR in all 16 pairs (+4.9 to +6.7; release setting 44.5 →
  49.5) and DeHate pooled PR in all 16 (+0.4 to +0.9); HateClipSeg pooled PR moves by −0.5 to +0.7. Without both
  position weight and smoothing (`g1-1_pos0`) within falls sharply (HateMM 72.2 → 54.9 at tau 0.05). No configuration
  reaches the naive row on HateMM pooled ROC / PR or on DeHate pooled ROC / PR.

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

#### 3.3.1 The release's own grid search on the stored scores (planning-session brief 5, item K4, 2026-11-15; test-selected)

Why: (a) the release's default protocol is `--grid_search True`: it searches sigma (`--sigma_range 1,30`) and the
three context weights (`--weight_step 0.1`) on the test labels of each dataset; the run above fixed alpha .6 / beta .3
/ gamma .1 and searched nothing. (b) Deviation 7 kept sigma in seconds (15 frames at 25 fps = 0.6 s), but our clip is
8 s, not about 1 s; in clip units the release's sigma 15 frames is 15 / 24 of a clip = 5 s here, so the run above had
almost no smoothing.

CPU only, no new inference: the stored `prob` read-outs of the three context images per window (`readout.jsonl`
`key` / `wa` / `tc`; all present for every window of every cohort video). Code `anyanomaly_k4.py`. The enumeration
follows the release (`third_party/AnyAnomaly/functions/eval_func.py` `evaluate_auc`): `for sigma in range(1, 30)`
(1 … 29; the release's range excludes 30), `for a1, a2 in np.arange(0.0, 1.1, 0.1)`, `a3 = 1 - a1 - a2` kept if
`0 <= a3 <= 1`. With the release's floating-point arithmetic this keeps 60 weight triples, not 66 (six triples with
a1 + a2 = 1 get a3 slightly below 0 and are dropped, as in the release). 29 × 60 = 1740 configurations per corpus.
sigma s release frames is applied as s / 24 windows = s / 3 seconds on the 4 fps curve, per video (as our port).
Selection rules fixed by the planning session before the grid was read. Check: the code reproduces the stored `prob`
metrics.json (HateMM and HateClipSeg to 2e-8; DeHate within differs by 0.0004 from floating-point rounding: many
AnyAnomaly curves have plateaus of near-equal values, so within-video ROC changes with the last bits; the curves agree
to 4e-16). Selection uses the grid values; reported numbers are the evaluator's metrics.json of the written rows.

Run: uoa-lab1 (sc474397), CPU, `setsid nohup`, 2026-11-15, code at commit 26bcd29 plus `anyanomaly_k4.py`
(committed with this section). Full grid (5220 lines: corpus, sigma, weights, ROC / PR / within) in
`runs/20261113_lf_baselines/anyanomaly_k4/grid.jsonl`; selections and ranges in `summary.json`; log `run.log`.
A second launch started by mistake while the first was still running was stopped; its partial per-sigma files were
deleted, and every number here is from the first run.

Rows written (`runs/20261113_lf_baselines/anyanomaly/<DS>/u8/<name>/metrics.json`); all three are test-selected:

| row | configuration | HateMM ROC / PR / within | HateClipSeg | DeHate | cells ≥ naive |
|---|---|---|---|---|---|
| naive Qwen3-VL-8B | | 81.0 / 55.3 / 61.8 | 70.6 / 67.4 / 61.2 | 69.1 / 17.7 / 60.5 | |
| run above (`u8/prob`) | .6 / .3 / .1, sigma 0.6 s | 79.1 / 53.7 / 63.3 | 64.9 / 61.8 / 58.2 | 66.4 / 15.3 / 60.2 | 1 |
| `k4_clipunit_default` (defined before its numbers were seen) | .6 / .3 / .1, sigma 15 (5 s) | 80.9 / 53.9 / 63.7 | 66.8 / 63.0 / 59.0 | 67.0 / 16.0 / 58.8 | 1 |
| `k4_grid_common` (K1 rule) | .2 / .4 / .4, sigma 14 (4.7 s) | 82.0 / 55.9 / 62.8 | 67.2 / 63.6 / 58.8 | 66.9 / 15.6 / 60.3 | 3 |
| `k4_grid_per_corpus` = "AnyAnomaly, release protocol (per-dataset grid search on test labels)" | HateMM 0 / .2 / .8, sigma 29; HCS .2 / 0 / .8, sigma 29; DeHate .5 / .5 / 0, sigma 15 | 84.0 / 56.8 / 60.1 | 68.8 / 64.4 / 57.5 | 67.4 / 15.2 / 59.6 | 2 |

- `k4_grid_common`: 4 of 9 cells ≥ naive in the grid values, largest shortfall 3.80; the written metrics.json gives 3
  (DeHate within: 60.57 in the grid, 60.35 in the written metrics.json, naive 60.51; the same rounding effect as above,
  here 0.22 points, decides this cell).
  Next in the ranking: .1 / .5 / .4 sigma 14 (shortfall 3.90), .2 / .3 / .5 sigma 10 (3.91).
- `k4_grid_per_corpus`: selected per corpus by pooled ROC only (the release selects by frame AUC); the other two
  metrics are those of the same configuration. On HateMM and HateClipSeg the best sigma is 29, the end of the
  release's range.
- Grid range per corpus (lowest / highest over the 1740 configurations): HateMM ROC 77.5–84.0, PR 50.7–57.3, within
  56.4–64.8; HateClipSeg ROC 62.9–68.8, PR 60.4–64.5, within 56.4–59.4; DeHate ROC 64.9–67.4, PR 13.8–17.3, within
  56.1–61.8. No configuration reaches the naive row on HateClipSeg ROC / PR / within or on DeHate ROC / PR.

Gate 1 (`runs/20261113_lf_baselines/gate/gate.md`, regenerated; the three rows enter sets A and C, not B / D; the
per-corpus row is allowed only because it is the method's published protocol, stated under the tables in gate.md):
- A and C: 7 of 9 cells ≥ naive (was 5). New: HateMM pooled ROC 84.04 (+3.05) and PR 56.84 (+1.51), both from the
  per-corpus row. Without the per-corpus row, A / C are still 7 of 9 (`k4_grid_common`: HateMM ROC +1.04, PR +0.59).
- Gate 1 still fails: largest shortfall 2.17 > 1.0. Failing cells: HateClipSeg within −2.17 (best is now
  `k4_clipunit_default`, 59.01; was MARS −2.51) and DeHate pooled PR −1.30 (MARS).
- B and D unchanged: 3 of 9, largest shortfall 2.51.

### 3.4 Cog-VADU (TMLR 2026; method and configuration written before the run; planning-session brief 5, item H5)

Paper: Wani, Atito, Kittler, Awais, "Cog-VADU: A Training-Free Cognitive Reasoning Framework for Video Anomaly
Detection and Understanding", Transactions on Machine Learning Research (2026), https://openreview.net/forum?id=QcuSMNG7J8.
Code: local clone `third_party/Cog-VADU` at commit b440861 (2026-09-01). Venue from the release README (badge "TMLR
Accepted" linking the OpenReview forum; BibTeX `journal={Transactions on Machine Learning Research}`, year 2026).
Opened: the release README and code only. Not opened: the OpenReview page and the PDF; acceptance rests on the
release README.

**What the release code does** (scripts `Scripts/01`–`05`, `evaluate/06_evaluate.sh`; every constant as the scripts
set it):
1. Raw stage (`MY_MODEL/Raw_CoADTP_STAGE1.py`, backbone VideoLLaMA3-7B). Clip centres every `frame_interval` = 16
   frames (30 fps, so every 0.53 s); each clip = the `clip_duration` = 10 s around the centre (300 frames, clipped
   at the video ends), 10 frames uniformly sub-sampled, written as a 1 fps video and given to the model at
   `fps_vlama` 1. System message "You are a helpful assistant specialized in video analysis and anomaly detection.";
   user text = the chain-of-thought prompt (`build_cot_prompt`): role and 0.000–1.000 score task, the ANOMALY
   CATEGORIES list (`anomaly_db`: NORMAL + 13 UCF-Crime classes), then, from the second clip of a video on, a
   PREVIOUS CLIP CONTEXT block with the previous clip's DESCRIPTION, ANOMALY_TYPE and (smoothed) score (`%.3f`) and
   the instruction to keep temporal consistency, then the output format DESCRIPTION / REASONING / EXPLANATION /
   ANOMALY_SCORE / ANOMALY_TYPE and the scoring guidelines. `max_new_tokens` 512 (script; argparse default 150).
   Parsing (`_parse_videollama_output`): an answer containing a refusal phrase reuses the previous analysis
   unchanged; otherwise the line starting `anomaly_score:` gives the first number, clamped to [0, 1] and rounded to
   3 decimals (no such line: the previous clip's score); a type not in `anomaly_db` becomes NORMAL. Temporal
   smoothing: score = 0.3 × previous score + 0.7 × this score (`temporal_consistency_weight` 0.3, a constructor
   default, not a command-line argument), rounded to 3 decimals; the smoothed analysis is what the next clip's
   prompt carries. The chain restarts for every video.
2. `extract_raw_scores_summaries.py`: raw score = the smoothed score; summary = the DESCRIPTION.
3. `get_visual_embeddings.py`: ImageBind-huge VISION embedding per clip centre (same 10 s window, 10 frames,
   `load_and_transform_video_data`).
4. `create_summary_index_FAST.py`: per video, a FAISS inner-product index of the L2-normalised ImageBind TEXT
   embeddings of the descriptions; identical descriptions share one entry (index entry → every centre with that
   description).
5. `video_text_TEST3_fast.py` (script: N = 10, `num_neighbors` 10; defaults `initial_k` None → max(30, 3 × 10) = 30,
   `text_weight` 0.5, `visual_weight` 0.5, `temporal_window` 5): per clip centre, the L2-normalised video embedding
   searches the index for `initial_k` descriptions (inner product D); each candidate gets text_sim = 1 / (1 + D) and
   visual_sim = the largest, over the centres sharing that description, of cos(query video embedding, that centre's
   video embedding) × w, with w = 1 + 0.5 exp(−|Δ| / (2 × 16 frames)) if |Δ| ≤ 5 × 16 frames, else 1; combined =
   0.5 text_sim + 0.5 visual_sim; the top `num_neighbors` by combined score are kept with their raw scores and their
   D.
6. `eval2_k_exp.py` (script loops `--num_neighbors` 2 … 10 and reports each; argparse default 2): clip score =
   Σ softmax(D) × raw score over the top `num_neighbors` kept neighbours by D; each centre score is repeated for 16
   frames.

**Port** (`cogvadu_q3vl.py`; shared rules of §2–§3):
- Stages: `infer` (vLLM env: the chain, generation), `readout` (HateVLM env: fp32 read-out of the score), `embed`
  (HateVideo env: ImageBind VIDEO and TEXT embeddings with the release's own copy `third_party/Cog-VADU/ImageBind`,
  weights `data/assets/imagebind/imagebind_huge.pth`), `raster` (CPU: stages 4–6, 4 fps, F2, evaluator).
- Configuration fixed now (`u8/prob_default`; `u8/text_default` = the check read-out, same configuration):
  `initial_k` 30, `text_weight` 0.5, `visual_weight` 0.5, `temporal_window` 5, retrieval `num_neighbors` 10, eval
  `num_neighbors` 10, chain smoothing 0.3. Eval `num_neighbors` 10 is chosen because stage 5 retrieves 10 and the
  release's evaluation loop ends at 10; the other values (2 … 9) are in the grid of §3.4.1.

Deviations (all fixed before any run):
1. Backbone Qwen3-VL-8B instead of VideoLLaMA3-7B; frames given as images at the naive row's image size (not as a
   video file); vLLM greedy generation, `max_tokens` 512.
2. Unit: non-overlapping 8 s segments (`fixed_windows`), 8 frames at 1 fps, instead of 10 s windows every 0.53 s
   (about 15 times more calls); the same 8 frames go to the LVLM and to ImageBind (release: 10 frames per window).
   One segment = one release step in every step-counted constant: the chain (one previous segment, smoothing 0.3 per
   step), `temporal_window` 5 steps → 5 segments, the decay 2 × `frame_interval` → 2 segments. (In seconds the
   release's window is 5 × 0.53 = 2.7 s, less than one of our segments, so a seconds conversion would leave only
   the segment itself boosted; that case is `temporal_window` 0 in the §3.4.1 grid.)
3. Words: `Transcript: <words>` (Whisper large-v3 slice of the segment, `(no speech)` when empty) before the release
   prompt.
4. Anomaly definition: the ANOMALY CATEGORIES list (NORMAL + 13 UCF-Crime classes) is replaced by two entries,
   NORMAL (release text, unchanged) and HATEFUL: "An anomaly is hateful content.". Everything else in the prompt is
   the release text, typos included; the ANOMALY_TYPE line keeps the literal text `{', '.join(anomaly_types)}`,
   because in the release that part of the prompt is a plain (not an f-) string.
5. Read-outs. `prob` (primary): fp32 expected score at the position where the answer writes its ANOMALY_SCORE
   number (first digit, and the decimal after "0."; §2 rule), on the same prompt and images; the per-segment value
   then goes through the same chain rule as the release score (0.3 × previous + 0.7 × this; a refusal reuses the
   previous value), without rounding. An answer with no ANOMALY_SCORE number is read after the answer plus
   "\nANOMALY_SCORE: ". `text` (check): the release's parsed and smoothed number. The chain itself (what the next
   prompt carries) is always the generated text, as in the release.
6. Retrieval when a video has fewer distinct descriptions than `initial_k` or `num_neighbors`: candidates and kept
   neighbours are limited to the entries that exist (FAISS would return id −1, which the release code would read as
   the last entry). With 8 s segments most videos have fewer than 30 segments, so the text search usually returns
   every description of the video.
7. Stages 4–6 are run in numpy (inner products of normalised vectors, the same arithmetic as `faiss.IndexFlatIP`);
   the t-SNE plot and the per-video visualisation are skipped (no effect on scores).
8. 4 fps: a frame takes the score of the segment containing its centre (`units_to_4fps`), F2, single evaluator.

Cost (new video): one generation per 8 s segment (up to 512 tokens, the chain makes a video's segments sequential;
segments of different videos are batched), one fp32 forward for the read-out, one ImageBind VIDEO and TEXT
embedding per segment. Segments: HateMM 3768, HateClipSeg 3591, DeHate 14375.

Plan: smoke on 2 HateMM videos (uoa-lab1), then HateMM, HateClipSeg, DeHate; raster; `gate.py` row in sets A–D for
`u8/prob_default` (the text read-out as an extra, test-selected configuration).

#### 3.4.1 Grid over the release's re-ranking arguments (defined before the full run; CPU only; test-selected)

Arguments the release exposes for stages 5–6 (command-line arguments of `video_text_TEST3_fast.py` and
`eval2_k_exp.py`) and the ranges (the release's own range where it gives one; otherwise the default halved and
doubled twice, rounded to integers with Python `round`):

| argument | release default / script value | grid |
|---|---|---|
| `eval2_k_exp.py --num_neighbors` (neighbours averaged) | loop 2 … 10 in `06_evaluate.sh` | 2, 3, …, 10 (the release's range) |
| `video_text_TEST3_fast.py --num_neighbors` (neighbours kept) | 10 | 2, 5, 10, 20, 40 |
| `--initial_k` | None → 30 | 8, 15, 30, 60, 120 |
| `--text_weight` | 0.5 | 0.125, 0.25, 0.5, 1, 2 |
| `--visual_weight` | 0.5 | 0.125, 0.25, 0.5, 1, 2 |
| `--temporal_window` (segments) | 5 | 1, 2, 5, 10, 20, and 0 (the seconds conversion of deviation 2) |

Not in the grid: the chain smoothing 0.3 (`temporal_consistency_weight` is a constructor default, not an argument)
and the decay 2 × `frame_interval` (a constant in the code). Only the ratio of the two weights changes the ranking,
so the 25 weight pairs give 9 distinct results; an eval `num_neighbors` above the kept number averages all kept
neighbours. Read-out `prob`. Selection: the K1 rule (one configuration for all three corpora; most cells ≥ the naive
row, ties by the smallest largest shortfall, then grid order). The selected row enters gate sets A and C as
test-selected.

Code: `cogvadu_grid.py` (written before the full run). Grid order for ties: eval `num_neighbors`, kept
`num_neighbors`, `initial_k`, `text_weight`, `visual_weight`, `temporal_window`, each in the order of the table
(33 750 configurations). The script splits `refine_video` into retrieval and averaging; on the smoke videos the split
equals `refine_video` exactly on four configurations, and at start it checks again on the full data against
`refine_video` and the stored `prob_default` metrics. Output: `runs/20261113_lf_baselines/cogvadu_grid/`
(`grid.jsonl`, `summary.json`) and `cogvadu/<DS>/u8/grid_selected/metrics.json`.

#### 3.4.2 Results (2026-11-16)

Runs: uoa-lab1 (sc474397). Smoke Slurm 2349 failed at the ImageBind stage (pytorchvideo `FrameVideo` needs OpenCV,
missing in HateVideo; `opencv-python-headless==4.10.0.84` installed with `--no-deps`, numpy 1.26.4 kept, recorded in
`environment_HateVideo.yml` and `runs/_setup_lab1/pip_installs.log`); smoke Slurm 2352 passed (2 HateMM videos, 17
segments). Full run Slurm 2353, 6 h 12 min GPU (generation 33 / 24 / 103 min, read-out 17 / 13 / 54 min,
ImageBind 23 / 20 / 84 min for HateMM / HateClipSeg / DeHate). Grid: `cogvadu_grid.py` on uoa-lab1 CPU, 16
processes, 19 min. No failed videos. Segments 3768 / 3591 / 14375; answers without an ANOMALY_SCORE number 0 / 1 / 0
(read after the appended prefix); refusal steps that reuse the previous value 0 / 1 / 6; answers cut at 512 tokens
0 / 1 / 0.

Metrics (evaluator `metrics.json` under `runs/20261113_lf_baselines/cogvadu/<DS>/u8/<config>/`), pooled ROC / pooled
PR / within:

| row | HateMM | HateClipSeg | DeHate | cells >= naive |
|---|---|---|---|---|
| naive `vlm` | 81.0 / 55.3 / 61.8 | 70.6 / 67.4 / 61.2 | 69.1 / 17.7 / 60.5 | — |
| `prob_default` (release defaults; fixed before the run; sets A–D) | 70.7 / 44.4 / 56.1 | 68.9 / 64.0 / 54.3 | 55.0 / 11.6 / 51.7 | 0 |
| `text_default` (the release's own number; check, test-selected) | 69.6 / 43.9 / 54.0 | 68.2 / 63.2 / 54.2 | 54.7 / 10.9 / 51.8 | 0 |
| `grid_selected` (§3.4.1, K1 rule; test-selected; sets A, C) | 70.9 / 43.1 / 53.5 | 67.2 / 61.8 / 50.7 | 56.4 / 11.3 / 52.0 | 0 |

- `grid_selected` = eval `num_neighbors` 3, kept `num_neighbors` 10, `initial_k` 15, `text_weight` 2,
  `visual_weight` 0.125, `temporal_window` 0. No configuration of the 33 750 is at or above the naive row on any cell,
  so the K1 rule decides on the smallest largest shortfall (12.7 points, DeHate pooled ROC).
- Ranges over the grid: HateMM 67.5–73.2 / 40.6–47.0 / 47.5–57.8; HateClipSeg 65.7–69.4 / 60.2–65.5 / 47.3–57.0;
  DeHate 53.5–56.5 / 9.2–12.2 / 48.3–55.6. Best single cells stay below the naive row on all nine cells.
- Gate 1 after adding the three rows (`runs/20261113_lf_baselines/gate/gate.md`): unchanged. A / C 7 of 9 (fails:
  HateClipSeg within −2.17, AnyAnomaly clip-unit default; DeHate pooled PR −1.30, MARS); B / D 3 of 9. No Cog-VADU
  row is the best published row on any cell.

### 3.5 Gaussian temporal smoothing of the stored scores (planning session, 2026-11-16; written before the run)

From here on the planning session does the work itself (user, 2026-11-16).

Why. Gate 1 fails on two cells: HateClipSeg within (best published 59.01, naive 61.18) and DeHate pooled PR (best
16.40, naive 17.69). The releases of AnyAnomaly and Probe-VAD end with a Gaussian filter over time; on our corpora
that filter alone moved HateClipSeg within by +0.8 (AnyAnomaly, §3.3.1) and DeHate pooled PR by +1.9 (Probe-VAD,
§3.1). MARS, EventVAD, CoReVAD and Cog-VADU have no such step. MARS is a video-level detector; how its per-window
decision becomes a frame score is part of our port, not of the method. So the port gets a post-processing step:
a Gaussian filter over the 4 fps score curve of each video. This is **not** an option of any of the four releases
and every row below says so. No component of our method is used. Mechanism hypothesis: none; comparison rows.

Input (no new inference, CPU): the stored `predictions.jsonl` (4 fps curves) of
- MARS `mars/<DS>/prob` (score = log-odds),
- EventVAD `eventvad/<DS>/prob`,
- CoReVAD `corevad/<DS>/u8/prob_lrc1_tau0.05_g9-5_pos1` (release default),
- Cog-VADU `cogvadu/<DS>/u8/prob_default` (release default),
- reference only, not a gate row: the naive row `naive/vlm/<DS>` (score = log-odds).

Operation: per video `scipy.ndimage.gaussian_filter1d(curve, sigma = 4 * sigma_s)` (reflect boundary, as §3.3.1).

Grid (fixed now):
- `sigma_s` (seconds): 2, 4, 5, 8, 12, 16, 24, 32 (0.25 to 4 of our 8 s windows; 5 s is the AnyAnomaly clip-unit
  default of §3.3.1).
- score space: `raw` (the stored score) for every method; for the two log-odds rows (MARS, naive reference) also
  `prob` = 1 / (1 + exp(−score)), because averaging log-odds and averaging probabilities are different operations.
- Order: sigma ascending, `raw` before `prob`. MARS 16 configurations, the other three 8 each.

Rows written per method (evaluator `metrics.json` under `runs/20261113_lf_baselines/smooth/<method>/<DS>/<name>/`):
- `s5_raw`: sigma 5 s on the stored score; fixed before its numbers were seen.
- `sel`: one configuration for all three corpora, chosen on test labels by this rule: add the configuration's row
  to gate set A as it stands at commit 2439878; most cells >= naive for the enlarged set; ties by the smallest
  largest shortfall of the enlarged set; then by the row's own number of cells >= naive; then grid order.
Both rows enter gate sets A and C only, labelled "+ Gaussian smoothing (not in the release) [test-selected
candidate]" (the step itself was added after the gate result was known, so no smoothing row counts as fixed before
the run). The naive reference with the same filter is reported here and is not a gate row; the gate's naive row
stays the frozen unsmoothed row.

Metrics in the grid: the evaluator functions `pooled`, `within_video_macro` (`src/eval/evaluate.py`) on the test
split; the script first checks that the unsmoothed curves reproduce each stored `metrics.json`. Reported numbers
are the evaluator's `metrics.json` of the written rows. Code `smooth_grid.py`; output
`runs/20261113_lf_baselines/smooth/{grid.jsonl,summary.json,run.log}`. Cost: CPU minutes on uoa-lab1.
