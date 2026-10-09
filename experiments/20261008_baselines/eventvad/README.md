# EventVAD on the 4 fps protocol (HateMM, HateClipSeg, DeHate)

Label-free baseline (ACM MM 2025, arXiv 2504.13092, github.com/YihuaJerry/EventVAD, no licence). Nothing is trained;
released or public checkpoints only (CLIP ViT-B/16 OpenAI, RAFT `raft-things.pth`, VideoLLaMA2.1-7B-16F).
Mechanism hypothesis: none; this is a comparison row.

## Setup

- Method code: the reconstruction in `scripts/reproduction_baselines/eventvad/` (config, graph, boundary, video_io,
  prompt, features), imported unchanged. The release cannot run as published: `graph_propagation` is imported but
  not defined, the scoring prompt is the placeholder string `"prompt"`, the RAFT path is a placeholder and
  `src/evaluate.py` does not compile (`scripts/reproduction_baselines/DESIGN_EVENTVAD.md`).
- Driver: `eventvad_4fps.py` (`segment` = CLIP + RAFT + dynamic graph + boundaries, GPU; `score` = VideoLLaMA2 per
  event, GPU; `raster` = 4 fps curves + evaluator, CPU). Shared cohort / coverage / evaluator plumbing:
  `../lf_common.py`.
- Graph constants: the released code's `config.py` values (`--preset upstream`): CLIP weight alpha = 0.8, time
  decay gamma = 0.05 per frame index, the released trailing-window moving average, raw CLIP magnitudes in the node
  features. The paper states alpha = 0.75 and gamma = 0.6 (coordinator decision 2026-10-08: use the code constants,
  record the paper values). Savitzky-Golay 2 s, MAD k = 3, minimum gap 2 s, one GAT iteration, k = 64.
- Decode rate: min(native, 30) fps, the paper's FPS = 30 (plan L5, decision D5). Frames are capped at 1280 x 720.
- Prompt (**prompt reconstructed**): the Figure 2 reconstruction of `prompt.py` (`paper` arm) with the nine rules of
  `../hate_query.md` in its anomaly slot. Exact text:

  ```
  #Question: Are there any obvious or potential anomalies in the video? An anomaly is content that violates any of the following rules:
  1. Encourage violence against groups based on protected status
  ...
  9. Deny or minimize major violent events
  #Instruction: Let's think step by step to judge the anomaly of the video. Finally, output the anomaly score based on the thinking process.
  ```
  16 frames per event (VideoLLaMA2 uniform sampler), greedy, at most 2048 new tokens.
- Score parsing (the 2026-08 HateMM run left 41 % of events unparsed; here every event gets a score):
  A. the legacy parser and range rule of `prompt.py` unchanged, plus "X out of N" / "X/N" read as X / N;
  B. if A finds no number: answer extraction with the same model, frames and prompt, the model's own answer followed
     by `"\nTherefore, the final score is"` (the closing sentence of the paper's Figure 2 example; the second stage
     of zero-shot chain-of-thought prompting), greedy, at most 8 new tokens, first number read and range-normalised;
  C. if B gives no number: 0.0 (the legacy fill, plan F3). Counts per rule are in `<corpus>/raster_stats.json`.
- 4 fps: an event covers decoded frames `[s, e)` = seconds `[s/fps, e/fps)`; frame i takes the event containing
  `(i + 0.5) / 4` s; frames past the last decoded frame take the last event. Whole-video failure: plan F2 (median of
  the corpus's scored frames), listed in `run.log`; more than 1 % F2 stops the corpus.
- Engineering changes with the same arithmetic: RAFT runs on batches of 8 adjacent frame pairs; CLIP preprocessing
  runs in a thread pool; CLIP and RAFT features are cached per video (`<corpus>/features/<id>.npz`); CLIP is loaded
  from the local file (no download, no checksum). The SHA-256 checks in the port (`features.py`,
  `smoke_cpu_eventvad.py`) were deleted (hash ban); the checkpoint is checked by parsing it.
- Environments (the pinned torch 2.1/2.2 does not support the RTX 5090): lab-server
  `.cache/envs/eventvad` (Python 3.12, torch 2.8.0+cu128, transformers 4.57.1, setuptools < 81 for the vendored CLIP's
  `pkg_resources`); uoa-lab1 `~/venvs/SafetyContradiction` (same torch/transformers, the 2026-08 run's venv).

## Commands

```bash
# on lab-server (sc448960, user junyi), repo ~/Hate-follow-up
sbatch experiments/20261008_baselines/launch/eventvad_labserver.sbatch DeHate                     # job 289
sbatch experiments/20261008_baselines/launch/eventvad_labserver.sbatch DeHate HateMM HateClipSeg  # job 296 (queued)
# when both stages are complete, on uoa-lab1 after rsync of runs/20261008_baselines/eventvad/<corpus>/:
python3 experiments/20261008_baselines/eventvad/eventvad_4fps.py raster --dataset DeHate
```

Both GPU stages append one line per video and skip finished videos, so a job stopped by the partition's 1-day limit
is resubmitted unchanged.

## Runs

| corpus | host | job | status |
|---|---|---|---|
| DeHate (1151) | lab-server (sc448960) | 289, then 296 | running: segmentation since 2026-10-08 03:52 NZDT at about 48 decoded frames/s (about 20 h projected), then scoring (about 11 h projected); the 1-day limit stops 289, and the queued job 296 resumes it |
| HateMM (215) | lab-server | 296 | queued after DeHate (same job) |
| HateClipSeg (118) | lab-server | 296 | queued after HateMM (same job) |

A first DeHate job (283/284) used the paper preset (alpha 0.75, gamma 0.6); it was cancelled after 11 videos when the
coordinator fixed the released constants. Its outputs are kept in
`runs/20261008_baselines/eventvad/_aborted_paper_preset_DeHate/` on lab-server and are not used.

## Results

None yet.

## Transcript (audio-visual) variant, 2026-10-09

User decision 2026-10-09: the label-free baselines must see both the picture and the speech. This variant gives
EventVAD's scorer the transcript of each event's time span. Mechanism hypothesis: none; this is a comparison row
(EventVAD + speech).

**What changes.** Only the scoring prompt. The first line of the user turn (after VideoLLaMA2's video token) is the
event's transcript; the rest is the visual-only prompt unchanged:

```
Speech during this segment: <text>
#Question: Are there any obvious or potential anomalies in the video? An anomaly is content that violates any of the following rules:
1. Encourage violence against groups based on protected status
...
9. Deny or minimize major violent events
#Instruction: Let's think step by step to judge the anomaly of the video. Finally, output the anomaly score based on the thinking process.
```

**What stays the same.** The events (read from the visual-only `runs/20261008_baselines/eventvad/<DS>/events.jsonl`,
not recomputed; the raster stage checks that every scored event equals that file's), the 16 frames per event,
greedy decoding with at most 2048 new tokens, parsing rules A/B/C (rule B's extraction call uses the same prompt,
transcript included), the 4 fps rasterisation, F2.

**Transcript.**
- Source: `data/asr_whisper_large_v3/<DS>/timestamped_chunks.jsonl` (Whisper large-v3 segments). Untimed chunks are
  kept as in `../qwen3_text/text_llm.py`: a missing start is the previous segment's end; a missing end is the next
  chunk's start (if later) or the video duration (run_plan.md §1.3).
- Span: event `[s, e)` in decoded frames = seconds `[s/fps, e/fps)`; text by `src/video_inputs.py` `window_text`
  (the Reader's helper: segments inside the span whole, segments crossing it sliced proportionally at word
  boundaries); runs of whitespace collapsed to one space.
- Empty span: `(no speech)` (run_plan.md §1.3 F1).
- Cap: **512 tokens** of the model's own tokenizer (Qwen2), fixed before any run: the longest whole-word prefix that
  fits. Chosen from the model's context (VideoLLaMA2.1 was trained with 4,096 tokens; 16 frames plus the prompt and up
  to 2,048 answer tokens must fit) and checked against the input only: span lengths of the visual-only events
  (HateMM median 13 tokens, 99th percentile 172; DeHate 13 / 165) put about 0.1 % of events above 512. No result was
  looked at; the number is not tuned.

**Code.** `eventvad_4fps.py score --transcript` / `raster --transcript` (default off: the visual-only path is
unchanged; its prompt equals the one in the DeHate run's `score_config.json`). Outputs:
`runs/20261008_baselines/eventvad_av/<DS>/` (`event_scores.jsonl` with each event's `speech`, `speech_tokens`,
`speech_truncated`; `score_config.json`; `raster_stats.json` with no-speech and truncated event counts;
`predictions.jsonl`, `coverage.json`, `metrics.json`; method name `eventvad_av`). CPU check of the prompts without a
model: `eventvad_4fps.py prompts --dataset HateMM --limit 1`.

```bash
# on the GPU machine, repo root, after git pull (lab-server: *_labserver.sbatch; uoa-lab1: *_lab1.sbatch)
mkdir -p runs/20261008_baselines/eventvad_av
sbatch experiments/20261008_baselines/launch/eventvad_av_labserver.sbatch HateMM HateClipSeg DeHate
# on uoa-lab1 after rsync of runs/20261008_baselines/eventvad_av/<DS>/:
python3 experiments/20261008_baselines/eventvad/eventvad_4fps.py raster --transcript --dataset <DS>
```
