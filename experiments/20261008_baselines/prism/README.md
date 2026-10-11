# PRISM on HateMM, HateClipSeg, DeHate

Label-free baseline (ICML 2026, "PRISM: Training-Free Video Anomaly Detection via Intrinsic Statistical Modeling",
github.com/ytC2026/ICML2026-PRISM, MIT; cloned 2026-10-08, upstream last commit 2026-10-02). Nothing is trained;
the backbone is the public `Qwen/Qwen3-VL-Embedding-2B` with the release's `Qwen3VLEmbedder`. Mechanism hypothesis:
none; comparison row.

## Setup

- Driver `prism_hate.py`: `extract` (GPU, window embeddings), `score` (text axes + window scores + 4 fps curves),
  `eval` (coverage check + canonical evaluator through `../lf_common.py`).
- Release files followed: `feature_extraction/extract_qwen_xd.py` and `Code/Qwen_xd.py`, method M6 "PRISM (Full)".
  - Video: decord, every 16th native frame, windows of 7 samples with stride 1, instruction
    "Represent the video content for anomaly detection.", `Qwen3VLEmbedder(max_pixels=600*1000)` (its frame sampler
    expands each 7-frame list to 64 frames, as in the release).
  - Text: normal pool with the prefix "Video footage of ", class pools with
    "Real-world video footage of anomalous event: " (the release's XD templates; the UCF template
    "A surveillance video of" is not used, coordinator 2026-10-08). Whitening of the text covariance with
    lambda = 0.01, one axis per class, softmax temperature 0.2, Gaussian smoothing over windows with sigma = 16.
  - Engineering changes, same arithmetic: windows of a video are embedded in batches of 8 (equal sizes, no padding);
    bfloat16 weights (the checkpoint dtype; the release passes no dtype).
- Description pools (`hate_descriptions.json`, written by `make_descriptions.py` from the policy wording only; no
  video, transcript or label): 10 classes = the 9 rules of `../hate_query.md` and its short positive query, each the
  verbatim text + 20 generated descriptions; normal pool = "normal content" + 20 generated ordinary scenes.
  Generator: Qwen3-VL-8B-Instruct, greedy, run on lab-server (sc448960) on 2026-10-08 inside the EventVAD job's
  allocation (`runs/20261008_baselines/prism/make_descriptions.log`).
- Mapping to frames (coordinator 2026-10-08): the release scores window i at its start and leaves the last 6 sample
  positions unscored. Here window i covers native frames `[16 i, 16 (i + 7))` and each native frame takes the mean
  smoothed score of the windows covering it; frames after the last covered one (fewer than 16) hold the last value;
  4 fps frame j takes the native frame at `(j + 0.5) / 4` s.
- Fallbacks: videos with fewer than 7 samples (under 97 native frames) or a failed decode get plan F2 (median frame
  score of the corpus); listed in `run.log` and `score_stats.json`; more than 1 % stops the corpus. A file decord
  cannot open is re-encoded once to H.264 (`lf_common.transcode_h264`) and listed.
- Environment: `.cache/envs/prism` on uoa-lab3, Python 3.12, torch 2.8.0+cu128, transformers 4.57.1,
  qwen-vl-utils 0.0.14, decord 0.6.0.

## Commands

```bash
# uoa-lab3, repo ~/Hate-follow-up
sbatch experiments/20261008_baselines/launch/prism_lab3.sbatch        # extract + score, HateMM HateClipSeg DeHate
# after rsync -a uoa-lab3:Hate-follow-up/runs/20261008_baselines/prism/<corpus>/ to the same path on uoa-lab1:
python3 experiments/20261008_baselines/prism/prism_hate.py eval --dataset HateMM
```

## Runs

Job 295 on uoa-lab3 (sc474398) started HateMM with the first version of the extraction (release preprocessing, about
1.05 windows/s, 18 h projected for HateMM alone). After 6 videos its batch shell was paused and its extraction
process stopped, and the same sbatch script was started as a step inside the same allocation
(`srun --jobid=295 --overlap`, log `runs/20261008_baselines/prism/step_in_295.log`) with the per-frame cache, about
2.37 windows/s. The cache gives processor inputs identical to the release path (`prism_hate.py check`, run on the
first and last windows of one video per corpus), so the 6 features from the first version are kept. Projected cost
at 2.37 windows/s: HateMM about 6 h, HateClipSeg about 6 h, DeHate about 23 h (about 0.29 M windows in all); the
partition's 1-day limit will stop the job inside DeHate, which then resumes in a new job.

| corpus | host | job | status |
|---|---|---|---|
| HateMM (215) | uoa-lab3 (sc474398) | 295 (step) | running |
| HateClipSeg (118) | uoa-lab3 | 295 (step) | after HateMM |
| DeHate (1151) | uoa-lab3 | 295, then 308 (queued; resumes extraction, re-runs the cheap score stages) | after HateClipSeg |

## Results

None yet.

## Transcript (audio-visual) variant, 2026-10-09

User decision 2026-10-09: the label-free baselines must see both the picture and the speech. `prism_hate.py --av`
(default off; the visual-only run and its outputs are unchanged) writes `runs/20261008_baselines/prism_av/<corpus>/`,
method name `prism_av`.

- **Path taken: one mixed input item per window.** The release's `Qwen3VLEmbedder.format_model_input(video=...,
  text=...)` accepts a video and a text in the same item: both go into one user turn (video first, then text) and the
  item gets one last-token embedding. This is supported, so it is used; no second text item and no score fusion.
- **Exact input of window i** (system turn = the unchanged instruction):
  - system: `Represent the video content for anomaly detection.`
  - user: the same 7 sampled frames as the visual-only run (expanded to 64 by the embedder), then the text
    `Subtitles: <transcript of the window's span>`, or `Subtitles: (no speech)` when nothing is spoken in the span.
- **Span**: native frames `[16 i, 16 (i + 7))` (clipped to the video) divided by the decoded frame rate, i.e. the
  native frames that window i is mapped to in the frame mapping below. Span text: `lf_common.span_text`, which is
  `src/video_inputs.py` `window_text` on the Whisper large-v3 segments of
  `data/asr_whisper_large_v3/<corpus>/timestamped_chunks.jsonl` (a segment inside the span enters whole, a segment
  that overlaps it is cut proportionally at word boundaries). Untimed chunks are kept with the rule of
  `qwen3_text/text_llm.py` (`lf_common.transcript_segments`): a missing end becomes the next chunk's start if later,
  else the manifest duration; a missing start the previous chunk's end.
- **Token cap**: the span text keeps at most 128 tokens of the embedder's tokenizer (`AV_TEXT_CAP`). The video part of
  a window is up to about 7,990 tokens (64 frames + timestamps; measured 7,603 to 7,998 with the subtitle on the first
  video of each corpus), and the processor cuts every input at the embedder's `max_length` of 8192 tokens; the cap keeps
  every input below it. Each video's longest input is in `extract_meta.jsonl` (`max_tokens`); windows whose text hit
  the cap are counted (`n_windows_capped`). The cap was fixed before any run.
- **Unchanged**: description pools, prefixes, whitening (lambda 0.01), temperature 0.2, Gaussian smoothing over
  windows (sigma 16), window-to-frame mapping, F2 fallback for videos with no window or a failed decode.
- **Check**: `prism_hate.py check --av` (run on uoa-lab3 CPU, then at the start of every corpus in the job) confirms
  that the per-frame cache gives the release path's processor inputs exactly, with the text item present.
- **Environments**:
  - HateMM and HateClipSeg ran on uoa-campus3 (foscsmlprd03, NVIDIA H200). `.cache/envs/prism` there was built by
    `launch/campus_prism_env.sh` (log `runs/_setup_uoa-campus3/prism_env.log`) with
    `launch/campus_prism_env_constraints.txt` (= `pip freeze` of the uoa-lab3 env) as pip constraints: torch
    2.8.0+cu128, transformers 4.57.1, qwen-vl-utils 0.0.14, decord 0.6.0. The only package differences are `packaging`
    (conda-forge build) and `wheel` 0.48.0. uoa-campus3 has no system ffmpeg, so `lf_common.transcode_h264` uses a
    separate env with conda-forge ffmpeg 6.1.2 (`.cache/envs/ffmpeg`, same script; uoa-lab3 has /usr/bin/ffmpeg 6.1.1).
    Model files: the uoa-lab3 snapshot of `Qwen/Qwen3-VL-Embedding-2B`, copied with rsync (uoa-lab3 to uoa-lab1 to
    uoa-campus3), so every run reads the same files. Cohort videos, transcripts, manifests and cohort lists were copied
    from uoa-lab1 with rsync.
  - DeHate ran on uoa-lab3 (sc474398, RTX 5090) in the env of the visual-only run.
  - Each corpus ran whole on one machine.

```bash
# uoa-campus3, repo /data/jehc223/Hate-follow-up (budget check: squeue -u jehc223 summed over campus1-3 < 2)
sbatch experiments/20261008_baselines/launch/campus_prism_av.sbatch [CORPUS ...]   # check + extract + score
# uoa-lab3, repo ~/Hate-follow-up
sbatch --export=ALL,DATASETS=DeHate experiments/20261008_baselines/launch/prism_av_lab3.sbatch
# uoa-lab1, after rsync -a <host>:<repo>/runs/20261008_baselines/prism_av/<corpus>/ to the same path
~/miniconda3/envs/HateVideo/bin/python experiments/20261008_baselines/prism/prism_hate.py eval --av --dataset HateMM   # HCS, DeHate
```

### Runs (transcript variant)

| corpus | host | Slurm jobs | extraction time |
|---|---|---|---|
| HateMM (215) | uoa-campus3 (H200) | 23446 | 4.22 h |
| HateClipSeg (118) | uoa-campus3 (H200) | 23446, then 23452 for the 3 failed videos and the score stage | 4.23 h |
| DeHate (1151) | uoa-lab3 (RTX 5090) | 355, then 376 (resumed) | 29.11 h |

- Job 23446 (code commit 2bcf98d) started 2026-10-10 06:38 after about 10 h in the campus queue and ran HateMM
  (`CORPUS_DONE` 10:53), then HateClipSeg. On HateClipSeg two webm files that decord cannot open
  (`yt_5yZByxbH8cg`, `yt_jNY3ZXSTBb8`) failed with `FileNotFoundError: 'ffmpeg'`, so 3 of 118 videos (with
  `yt_8sIY4W1hBQM`, as in the visual-only run) needed F2. That is more than 1 %, so the score stage stopped the corpus
  (`STOP: 3 F2 videos > 1% of 118`) and the job ended before DeHate.
- Job 23452 (commit fba2185, ffmpeg env added) re-tried only the 3 failed HateClipSeg videos (the other 115 were kept):
  the two webm files were re-encoded and embedded, `yt_8sIY4W1hBQM` failed in decord again (F2, as in the
  visual-only run), and the score stage ran (`CORPUS_DONE HateClipSeg` 2026-10-11 05:11). The job was then cancelled
  before any DeHate video, because DeHate was by then most of the way through on uoa-lab3.
- DeHate: job 352 on uoa-lab3 (4 CPUs) was cancelled after 8 min and its 6 videos deleted, when the campus job looked
  faster. Job 355 (8 CPUs, commit fba2185) started 2026-10-10 15:07 and reached the partition's 1-day time limit at
  2026-10-11 15:07 after 980 videos. Job 376 (submitted by the coordinator; the same job file, with the PRISM code
  unchanged since fba2185) resumed at video 981 and finished at 20:16. No video was embedded twice
  (`window_texts.jsonl` has 1148 rows for 1148 distinct videos).
- Rates: 3.3-3.5 windows/s on the H200; 1.9-2.1 on the RTX 5090. The 5090 rate is about 15-20 % below the visual-only
  run on the same DeHate videos. 8 CPUs did not change it (the process used about one core), so the GPU is the limit.
  A likely cause is that windows in a batch now differ in text length, so each batch is padded and the attention takes
  a mask. That cause was not tested.
- GPU time: about 8.6 H200-hours (campus) plus 29.3 RTX 5090-hours (lab3), plus 8 min of the cancelled job 352.
- Outputs copied to uoa-lab1 with `rsync -a`. Evaluation ran there. The campus copies of the window features and the
  DeHate videos copied there were deleted after the rsync.

### Results (transcript variant)

Pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC, transcribed from `metrics.json` written by
`src/eval/evaluate_four_datasets.py`; exact cohort and every GT frame scored (`coverage.json`, `n_videos_overlap`
215 / 118 / 1151).

| corpus | visual only ROC / PR / within | transcript variant ROC / PR / within | source (variant) |
|---|---|---|---|
| HateMM | .6918 / .4226 / .5046 | .7224 / .4505 / .6258 | `runs/20261008_baselines/prism_av/HateMM/metrics.json` |
| HateClipSeg | .6118 / .5805 / .5536 | .6359 / .6034 / .5760 | `runs/20261008_baselines/prism_av/HateClipSeg/metrics.json` |
| DeHate | .4995 / .1275 / .5265 | .5412 / .1116 / .5649 | `runs/20261008_baselines/prism_av/DeHate/metrics.json` |

Visual-only numbers: `runs/20261008_baselines/prism/<corpus>/metrics.json`.

- F2 (median frame score of the corpus): HateMM 1 (`non_hate_video_141`, fewer than 7 samples); HateClipSeg 1
  (`yt_8sIY4W1hBQM`, decord error); DeHate 3 (`7439757774583549206`, `_m7jhIjuafc`, `hGb9vA7LNTA`, fewer than 7 samples).
  These are the same videos as in the visual-only run, and every share is below 1 %.
- Windows: 51,587 / 53,397 / 213,065, the same counts as the visual-only run. Windows with speech: 89.4 % / 82.3 % /
  89.0 %. Windows whose text hit the 128-token cap: 229 / 217 / 657. The longest input was 8,111 / 8,112 / 8,114
  tokens, so every input stayed under the 8192 limit. `n_windows_speech`, `n_windows_capped` and `max_tokens` per
  video are in `extract_meta.jsonl`; the window texts are in `window_texts.jsonl`.
- Re-encoded for decord: HateClipSeg `yt_5yZByxbH8cg`, `yt_jNY3ZXSTBb8`.

These numbers are baseline outputs selected without labels (release hyper-parameters, cap fixed before the run); they
are not development-selected.
