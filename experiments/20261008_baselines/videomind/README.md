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
  to `runs/20261008_baselines/videomind/_superseded/` and HateMM is rerun in full with the patch (job 306).

## Runs

| corpus | host | job | status |
|---|---|---|---|
| HateMM (215) | uoa-lab1 (sc474397) | 306 | queued (full rerun with the attention patch) |
| HateClipSeg (118) | uoa-lab1 (sc474397) | 298 | done, 10.5 min |
| DeHate (1151) | uoa-lab1 (sc474397) | 298 | running |

## Results

Pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC, transcribed from `metrics.json` written by
`src/eval/evaluate_four_datasets.py`; exact cohort and every GT frame scored (`coverage.json`, `n_videos_overlap`).

| corpus | ROC | PR | within | source |
|---|---:|---:|---:|---|
| HateClipSeg | .6174 | .5766 | .5336 | `runs/20261008_baselines/videomind/HateClipSeg/metrics.json` |

HateClipSeg: the grounder returned 100 proposals for every video (its cap after NMS); the verified top 5 cover on
average 32 % of a video's frames (`raster_stats.json`); 0 F2; `yt_5yZByxbH8cg` and `yt_jNY3ZXSTBb8` were re-encoded.
