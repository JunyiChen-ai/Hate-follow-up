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
