# ZS-ImageBind (audio channel) on DeHate (setup of the HateMM / HateClipSeg rows)

Label-free baseline. No training, no tuning, no label read. HateMM and HateClipSeg are the re-evaluated
Retrieval-hate campaign outputs (`runs/20261008_baselines/zs_imagebind_audio/`, see `../README.md`); this
directory adds DeHate with the same setup.

## Setup (identical to the campaign's)

- Port of Retrieval-hate `scripts/repro_campaign/extract_imagebind.py` (audio channel) and the scoring of
  `eval_frame.imagebind_curves`: `zs_imagebind_audio.py`.
- Audio: ffmpeg `-map 0:a:0 -ac 1 -ar 16000` from the video file; 2 s clips tiling the track (stride 2 s, a clip
  under 400 samples zero-padded to 400); ImageBind `waveform2melspec` numerics (kaldi fbank, 128 mel bins, target
  length 204), normalised (mean −4.268, std 9.138).
- Model: ImageBind-Huge (`third_party/lavad/libs/ImageBind`, `data/assets/imagebind/imagebind_huge.pth`), float16,
  batch 16; embeddings stored float16.
- Text: "normal" / "hateful", the campaign's own embedding file (`imagebind_text_normal_hateful.npy`, copied to
  `data/retrieval_hate_repro/imagebind_audio/`).
- Score per clip: softmax over (normal, hateful) of the unit-normalised audio embedding times the text embeddings,
  p(hateful). Native 0.5 fps; 4 fps frame i takes clip floor(i/8); frames past the last clip hold its value.
- Fallback F1 (run_plan.md §1.3): a video without an audio stream gets the embedding of digital silence of its
  duration (the campaign dropped such videos; on HateMM / HateClipSeg no cohort video was affected).

## Reproduction check

`zs_imagebind_audio.py verify` re-embedded HateMM test videos on uoa-lab2 and compared with the campaign's cached
embeddings (`runs/20261008_baselines/zs_imagebind_audio/DeHate/verify.json`): hate_video_1, _10, _101, _102, _114
give the same clip counts, mean cosine 1.0 and max |score difference| 0.0.

## Commands and host

`sbatch experiments/20261008_baselines/launch/lab2_imagebind_audio.sbatch` on uoa-lab2 (sc474399), HateVideo env:
`verify`, `extract` (1341 DeHate test videos → `data/imagebind_audio_2s/DeHate/`; silence embeddings of F1 videos
in `runs/20261008_baselines/zs_imagebind_audio/DeHate/silence_embeddings/`), `score` (1151 cohort videos).

## Results

Canonical evaluator, pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro ROC-AUC. Nothing was selected.

| corpus | ROC | PR | within | videos / frames (within defined) | source |
|---|---:|---:|---:|---|---|
| HateMM | .5654 | .2905 | .5256 | 215 / 116,975 (84) | `runs/20261008_baselines/zs_imagebind_audio/metrics.json` (campaign output) |
| HateClipSeg | .5652 | .5122 | .5218 | 118 / 113,002 (99) | same file |
| DeHate | .5302 | .0820 | .4819 | 1151 / 441,345 (222) | `runs/20261008_baselines/zs_imagebind_audio/DeHate/metrics.json` |

DeHate coverage: exact cohort, 1151/1151 scored, no fallback (every one of the 1341 test videos has an audio stream,
so F1 was never used). 797 curve frames lie past the last 2 s clip and hold its value.
Extraction: 361.6 s for 1341 videos (Slurm 297).
