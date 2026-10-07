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

## Runs and results

Filled in when the runs finish.
