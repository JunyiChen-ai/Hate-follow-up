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

## Transcript (audio-visual) variant, 2026-10-09

User decision 2026-10-09: the label-free baselines must see the picture and the speech. `zs_clip_av.py` adds a
transcript score to the frame score; the frame-only row above is unchanged and stays in its own directory.

What the model sees, per 4 fps frame i:
- **Frame** (unchanged): the frame-only run's value x = cos(img, "a hateful video frame") − cos(img, "a normal video
  frame"), read from its `predictions.jsonl` (HateMM / HateClipSeg `runs/20261008_baselines/zs_clip/predictions.jsonl`,
  DeHate `runs/20261008_baselines/zs_clip/DeHate/predictions.jsonl`), so the frame part is exactly the frame-only row.
  CLIP zero-shot softmax over the two prompts with logit scale 100 (the model's `exp(logit_scale)` = 100.000008):
  p_frame = sigmoid(100 x).
- **Transcript**: the Whisper large-v3 text (`data/asr_whisper_large_v3/<DS>/timestamped_chunks.jsonl`) of the 8 s
  window [8w, 8w + 8) s containing the frame (w = floor(i / 32)). Segments are loaded with
  `qwen3_text/text_llm.py` `load_segments` (untimed chunks kept: missing end = next chunk's start or the video
  duration) and cut to the window with `src/video_inputs.py` `window_text`
  (`../transcript_windows.py`). CLIP ViT-L/14-336 text tower, the campaign's own encoder
  (`CLIPTokenizerFast`, padding, truncation to 77 tokens, `get_text_features`, unit norm, float32); re-encoding the
  campaign's two frame prompts reproduces `prompt_emb.npz` to 1.2e-7. Anchors: `normal content` and
  `hateful content that attacks or demeans people based on protected attributes` (`hate_query.md` §2).
  p_text = softmax(100 · [cos(t, normal), cos(t, hateful)])_hateful.
- **Score**: (p_frame + p_text) / 2 in a window with speech; p_frame alone in a window without speech.
  No corpus statistic, no label, nothing selected.

Host and commands: uoa-lab1 (sc474397), CPU only (no Slurm job; `setsid nohup`, HateVideo env, torch 2.7.1,
transformers 4.57.6; CLIP-L/336 downloaded into the repository's `.cache/hf`, file list in
`runs/_setup_uoa-lab1/hf_download_clipL336_llama2tok.log`), 2026-10-09, commit fbb664a:

```bash
python experiments/20261008_baselines/zs_clip/zs_clip_av.py text --cpu   # 19,680 window texts, ~10 min
python experiments/20261008_baselines/zs_clip/zs_clip_av.py score        # combination + canonical evaluator
```

Outputs: `runs/20261008_baselines/zs_clip_av/<DS>/` (`transcript_windows.jsonl` = text, cosines and p_text of every
window with speech; `predictions.jsonl`, `coverage.json`, `config.json`, `metrics.json`, `run.log`);
`zs_clip_av/run.log`, `zs_clip_av/text_stage.out`, `zs_clip_av/anchor_text_emb.npy`.

Results (canonical evaluator, exact cohorts; pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro
ROC-AUC). Nothing selected; not development-selected.

| corpus | frame + transcript | frame only (above) | videos / frames (within defined) | windows with speech | frames in a window with speech | videos without speech | F2 |
|---|---|---|---|---:|---:|---:|---:|
| HateMM | .5054 / .2357 / .4843 | .5368 / .2774 / .5156 | 215 / 116,975 (84) | 3,442 | 107,990 (92 %) | 1 | 0 |
| HateClipSeg | .4792 / .4499 / .5202 | .4984 / .4627 / .5201 | 118 / 113,002 (99) | 3,141 | 99,466 (88 %) | 0 | 0 |
| DeHate | .5043 / .0755 / .5050 | .5117 / .0765 / .5038 | 1151 / 441,345 (222) | 13,097 | 407,246 of 442,461 curve frames (92 %) | 2 | 0 |

Sources: `runs/20261008_baselines/zs_clip_av/<DS>/metrics.json`; frame-only numbers from the table above.
Truncation at 77 tokens: 138 / 72 / 293 windows. No fallback was needed (every cohort video has a frame-only row;
a window without speech is the method's rule, not a fallback).

Why adding the transcript lowers the numbers here (the distributions below use no label): p_text is much lower than p_frame on average
(HateMM windows: mean p_text .088, frames: mean p_frame .429; DeHate windows: mean p_text .136), so averaging lowers
every frame inside a window with speech, and the pooled ranking partly becomes "speech present → lower score". The
CLIP text tower also matches topic words: the highest-p_text windows are news and commentary that talk about hate
("motivated by bias and hate", "stirring up racial hatred"), not hateful speech.
