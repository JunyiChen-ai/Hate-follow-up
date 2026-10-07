# ZS-CLIP on DeHate (setup of the HateMM / HateClipSeg rows)

Label-free baseline (LAVAD's "ZS CLIP" row). No training, no tuning, no label read. HateMM and HateClipSeg are the
re-evaluated Retrieval-hate campaign outputs (`runs/20261008_baselines/zs_clip/`, see `../README.md`); this
directory adds DeHate with the same setup (`hate_query.md`: one setup per method across the three corpora).

## Setup (identical to the campaign's)

- Port of Retrieval-hate `scripts/repro_campaign/extract_dense.py` (visual channel) and `zs_clip.py` (stages
  `prompts`, `visual`), uoa-lab2 `~/Retrieval-hate` read only: `zs_clip.py`.
- Frames: ffmpeg `fps=4,scale=w=336:h=336:force_original_aspect_ratio=increase:flags=bicubic,crop=336:336`, rgb24,
  /255, CLIP mean/std. Backbone `openai/clip-vit-large-patch14-336` `CLIPVisionModel`, fp16, `pooler_output`,
  batch 48; joint-space image embedding = `visual_projection(pooler_output)`.
- Prompts: "a normal video frame" / "a hateful video frame" (campaign set `main`), using the campaign's own text
  embeddings and projection matrix (`prompt_emb.npz`, copied to `data/retrieval_hate_repro/repro_zs_clip/`).
- Score per frame: cos(img, hateful) − cos(img, normal) (the campaign reported sigmoid(100·x); same ranking).
- Grid: native 4 fps (frame i = t i/4 s), used as is; frames past the last decoded frame hold its value.
- Fallback: F2 for a video that does not decode (run_plan.md §1.3).

## Reproduction check

`zs_clip.py verify` re-extracted 5 HateMM test videos (hate_video_1, _10, _101, _102, _114) on uoa-lab2 and compared
with the campaign's cached features: same frame count, max |feature difference| 0.0, max |score difference|
3e-8 (`runs/20261008_baselines/zs_clip/DeHate/verify.json`).

## Commands and host

`sbatch experiments/20261008_baselines/launch/lab2_zs_clip.sbatch` on uoa-lab2 (sc474399, partition
`local-sc474399`), HateVideo env (torch 2.7.1, transformers 4.57.6): `verify`, `extract` (all 1341 DeHate test
videos of `data/manifests/DeHate_test.jsonl` → `data/dense4fps_clipL336/DeHate/`), `score` (1151 cohort videos →
`runs/20261008_baselines/zs_clip/DeHate/`).

## Results

Canonical evaluator (`src/eval/evaluate_four_datasets.py`), pooled frame ROC-AUC / pooled frame PR-AUC /
within-video macro ROC-AUC. Not development-selected (nothing was selected).

| corpus | ROC | PR | within | videos / frames (within defined) | source |
|---|---:|---:|---:|---|---|
| HateMM | .5368 | .2774 | .5156 | 215 / 116,975 (84) | `runs/20261008_baselines/zs_clip/metrics.json` (campaign output) |
| HateClipSeg | .4984 | .4627 | .5201 | 118 / 113,002 (99) | same file |
| DeHate | .5117 | .0765 | .5038 | 1151 / 441,345 (222) | `runs/20261008_baselines/zs_clip/DeHate/metrics.json` |

DeHate coverage (`coverage.json`): exact cohort, 1151/1151 videos scored, no fallback, 0 decode failures over the
1341 extracted videos. 748 frames (0.17 % of the curve frames) lie past the last
decoded 4 fps frame and hold its value. DeHate base rate is .076, so PR .0765 is chance level.
