# VideoMind-7B as a zero-shot hate localizer (HateMM, HateClipSeg, DeHate)

Label-free baseline (ICLR 2026, arXiv 2503.13444, github.com/yeliudev/VideoMind, BSD-3-Clause; cloned 2026-10-08,
upstream last commit 2026-02-08). Nothing is trained: released checkpoint `yeliudev/VideoMind-7B` (LoRA roles on
`Qwen/Qwen2-VL-7B-Instruct`). The grounder and verifier were trained by the authors on public temporal-grounding
data (QVHighlights, DiDeMo, TACoS, ...), so the row carries the "pretrained on other labelled data" footnote.
Mechanism hypothesis: none; comparison row.

## Setup

- Driver `videomind_hate.py`, stages `infer` (GPU) and `raster` (CPU, 4 fps curves + evaluator via `../lf_common.py`).
- Pipeline = the authors' temporal-grounding evaluation (`videomind/eval/infer_auto.py` as run by
  `scripts/evaluation/eval_auto_7b.sh`): planner not called; grounder over the whole video at 1 fps with at most
  150 frames (videos longer than 150 s are sampled below 1 fps), 36..64 x 28 x 28 pixels per frame; up to 100
  proposals after the model's NMS (IoU 0.75); verifier on the top 5 proposals (2 fps, at most 64 frames, proposal
  widened by half its length on each side, segment tokens inserted), score = sigmoid(logit "Yes" - logit "No").
  float16, SDPA attention, greedy.
- Query (`../hate_query.md` section 2, grounder and verifier):
  `hateful content that attacks or demeans people based on protected attributes`.
- Frame score (task brief): the maximum verifier score over the verified proposals covering the 4 fps frame centre,
  0 where none covers it. One deviation: infer_auto.py skips the verifier when the grounder returns a single
  proposal; here it is run on that proposal too. Proposal counts per video are in `<corpus>/raster_stats.json` and
  in each prediction row (`extra.n_proposals`); the verified intervals and their scores are in `intervals`.
- Decoding: decord (VideoMind's reader). A file decord cannot open is re-encoded once to H.264 at the same size and
  rate (`lf_common.transcode_h264`) and listed (on HateClipSeg: `yt_5yZByxbH8cg`, `yt_jNY3ZXSTBb8`).
- Fallback F2 (whole-video failure): median frame score of the corpus; more than 1 % stops the corpus.
- Environment (the pinned torch 2.4 does not support the RTX 5090): `.cache/envs/videomind`, Python 3.11,
  torch 2.8.0+cu128, transformers 4.45.2 (VideoMind patches this version), peft 0.14.0, nncore 0.4.5, decord 0.6.0.

## Commands

```bash
# uoa-lab1, repo ~/Hate-follow-up
sbatch experiments/20261008_baselines/launch/videomind_lab1.sbatch            # HateMM, HateClipSeg, DeHate
python3 experiments/20261008_baselines/videomind/videomind_hate.py raster --dataset HateMM   # then HCS, DeHate
```

- Memory fix: transformers 4.45.2's SDPA vision attention builds a dense block-diagonal mask over all patches of the
  video (about 15k patches for 150 frames) and ran out of memory on the 32 GB card for 131 of 215 HateMM videos in
  job 298. `patch_vision_attention` runs the same attention per temporal block (checked equal to the masked version,
  max difference 2e-16 in float64). HateClipSeg and DeHate ran with the patch; the HateMM output of job 298 was moved
  to `runs/20261008_baselines/videomind/_superseded/` and HateMM was rerun in full with the patch, as a step inside
  the same allocation after DeHate (`srun --jobid=298 --overlap`, log `step_in_298_HateMM.log`); the queued retry
  job 306 was then cancelled.

## Runs

| corpus | host | job | status |
|---|---|---|---|
| HateMM (215) | uoa-lab1 (sc474397) | 298 (step) | done, 11.6 min |
| HateClipSeg (118) | uoa-lab1 (sc474397) | 298 | done, 10.5 min |
| DeHate (1151) | uoa-lab1 (sc474397) | 298 | done, 73.1 min |

About 1.6 GPU-h in all on one RTX 5090 (one grounder call and five verifier calls per video).

## Results

Pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC, transcribed from `metrics.json` written by
`src/eval/evaluate_four_datasets.py`; exact cohort and every GT frame scored (`coverage.json`, `n_videos_overlap`).

| corpus | ROC | PR | within | source |
|---|---:|---:|---:|---|
| HateMM | .5781 | .3459 | .5599 | `runs/20261008_baselines/videomind/HateMM/metrics.json` |
| HateClipSeg | .6174 | .5766 | .5336 | `runs/20261008_baselines/videomind/HateClipSeg/metrics.json` |
| DeHate | .5511 | .0978 | .5760 | `runs/20261008_baselines/videomind/DeHate/metrics.json` |

Proposals per video (grounder, after NMS, cap 100; `raster_stats.json`): HateMM mean 72.5 (median 100, min 3);
HateClipSeg 100 for every video; DeHate mean 60.8 (median 60, min 3). No video had a single proposal and the grounder
never failed to emit a proposal. The verified top 5 cover on average 43 % / 32 % / 48 % of a video's frames. No F2
fallback on any corpus. Re-encoded for decord: HateClipSeg `yt_5yZByxbH8cg`, `yt_jNY3ZXSTBb8`.

These numbers are baseline outputs selected without labels (default hyper-parameters, fixed query); they are not
development-selected.

## Transcript (audio-visual) variant, 2026-10-09

User decision 2026-10-09: the label-free baselines must see both the picture and the speech. `videomind_hate.py --av`
(default off; the visual-only run and its outputs are unchanged) writes
`runs/20261008_baselines/videomind_av/<corpus>/`, method name `videomind_av`.

- **Where the transcript goes**: in both the grounder and the verifier prompt, right after the query sentence
  (`Now I give you the query: '<query>'. `) and before the closing instruction (`Please think carefully ...`):

  ```
  ... Now I give you the query: 'hateful content that attacks or demeans people based on protected attributes'. Subtitles:
  [0.0-2.6s] Now, let's say the numbers backwards.
  [3.3-9.4s] 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
  ...
  Please think carefully and provide your response.
  ```

  - Grounder: every Whisper segment inside `[0, duration]` of the decoded video, times in video seconds.
  - Verifier: the segments inside its clip `[s1, e1]` (the proposal widened by half its length on each side), cut at
    word boundaries at the clip edges, times counted from the clip start (the verifier sees only the clip).
  - No speech in the span: `Subtitles: (no speech)`.
- **Transcript source**: `data/asr_whisper_large_v3/<corpus>/timestamped_chunks.jsonl` (Whisper large-v3 segments),
  untimed chunks kept with the rule of `qwen3_text/text_llm.py` (`lf_common.transcript_segments`); partial segments cut
  with `src/video_inputs.py` `window_text` (`lf_common.span_text`).
- **Token cap**: at most 2048 subtitle tokens (Qwen2-VL tokenizer) per prompt (`SUB_CAP`). Lines are kept in time
  order; the line that crosses the cap keeps its first tokens and later lines are dropped. The cap was fixed before any
  run from the transcript lengths only: it holds the whole transcript of all but 8 / 1 / 4 of the 215 / 118 / 1151
  cohort videos (HateMM / HateClipSeg / DeHate; computed on uoa-lab2 with the Qwen2-VL tokenizer; mean grounder subtitle
  length 569 / 775 / 409 tokens). Per-video counts are in `raw.jsonl` (`subtitles`) and `raster_stats.json`.
- **Unchanged**: query, model, frames (grounder 1 fps / at most 150 frames; verifier 2 fps / at most 64 frames),
  proposals, verifier top 5, frame score (max verifier probability over covering verified proposals, 0 elsewhere), F2.
- **Host**: uoa-lab2 (sc474399), which holds the VideoMind env and weights (uoa-lab1's copy of the weights is gone).

```bash
# uoa-lab2, repo ~/Hate-follow-up
sbatch experiments/20261008_baselines/launch/videomind_av_lab2.sbatch       # HateMM, HateClipSeg, DeHate
# uoa-lab1, after rsync -a uoa-lab2:Hate-follow-up/runs/20261008_baselines/videomind_av/ to the same path
~/miniconda3/envs/HateVideo/bin/python experiments/20261008_baselines/videomind/videomind_hate.py raster --av --dataset HateMM   # then HCS, DeHate
```

### Runs (transcript variant)

| corpus | host | job | wall time |
|---|---|---|---|
| HateMM (215) | uoa-lab2 (sc474399), RTX 5090 | 342 | 20.1 min |
| HateClipSeg (118) | uoa-lab2 | 342 | 18.2 min |
| DeHate (1151) | uoa-lab2 | 342 | 128.7 min |

Job 342 ran 2026-10-09 23:30 to 2026-10-10 02:18 (about 2.8 GPU-h, code commit 161e693); outputs copied to uoa-lab1
with `rsync -a`, then `raster --av` and the evaluator ran on uoa-lab1.

### Results (transcript variant)

Pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC, transcribed from `metrics.json` written by
`src/eval/evaluate_four_datasets.py`; exact cohort, every GT frame scored (`coverage.json`, `n_videos_overlap`).

| corpus | visual only ROC / PR / within | transcript variant ROC / PR / within | source (variant) |
|---|---|---|---|
| HateMM | .5781 / .3459 / .5599 | .6775 / .3969 / .6326 | `runs/20261008_baselines/videomind_av/HateMM/metrics.json` |
| HateClipSeg | .6174 / .5766 / .5336 | .6205 / .6102 / .5448 | `runs/20261008_baselines/videomind_av/HateClipSeg/metrics.json` |
| DeHate | .5511 / .0978 / .5760 | .5660 / .0892 / .5711 | `runs/20261008_baselines/videomind_av/DeHate/metrics.json` |

- Fallbacks: no F2 video on any corpus. The grounder emitted no proposal for one HateMM video (`non_hate_video_221`,
  15 s; empty response); the code then uses infer_auto.py's own default proposals (five fixed spans), as in the
  release.
- Subtitles (`raster_stats.json`): mean grounder block 568 / 774 / 407 tokens (HMM / HCS / DeHate); grounder prompts
  cut by the 2048-token cap: 8 / 1 / 4 videos; videos with no speech in `[0, duration]`: 1 / 0 / 4; verifier prompts
  cut by the cap: 18 of 1070 / 1 of 590 / 10 of 5729; verifier clips with no speech: 50 / 6 / 173.
- With subtitles the grounder proposes much longer moments: the verified top-5 proposals average 47 / 48 / 46 % of the
  video (visual only 17 / 13 / 19 %), and the verified proposals cover 88 / 87 / 88 % of a video's frames on average
  (visual only 43 / 32 / 48 %).
- Re-encoded for decord: HateClipSeg `yt_5yZByxbH8cg`, `yt_jNY3ZXSTBb8` (as in the visual-only run).

These numbers are baseline outputs selected without labels (fixed query, cap fixed before the run); they are not
development-selected.
