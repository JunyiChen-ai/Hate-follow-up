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

## Transcript (audio-visual) variant, 2026-10-09

User decision 2026-10-09: the label-free baselines must see the picture and the speech. `zs_imagebind_av.py` scores
three ImageBind modalities against the same text pair as the audio-only row ("normal" / "hateful", the campaign's
`imagebind_text_normal_hateful.npy`) and averages them. The audio-only row above is unchanged.

What the model sees, per 4 fps frame i (each modality → p(hateful) = softmax over (normal, hateful) of the
unit-normalised embedding times the text embeddings, as in the audio-only row; the text embeddings carry ImageBind's
learned logit scale, norm 100):
- **Vision**: per-frame ImageBind-Huge VISION embedding at 4 fps, the campaign's `image` channel
  (`extract_imagebind.py`: ffmpeg `fps=4`, short side 224 bicubic + centre crop 224, rgb24, /255, CLIP mean/std,
  float16, batch 16). HateMM / HateClipSeg: the campaign's cache, cohort files copied to
  `data/retrieval_hate_repro/imagebind_image/<DS>/` (see its `PROVENANCE.md`). DeHate: extracted with the ported
  pipeline (`extract-image` stage) for all 1341 test videos → `data/imagebind_image_4fps/DeHate/`. Frame i ← embedding
  i; frames past the last decoded frame hold its value.
- **Audio**: the audio-only row's 2 s clip embeddings, unchanged (frame i ← clip floor(i / 8)).
- **Transcript**: ImageBind TEXT embedding of the Whisper large-v3 text of the 8 s window [8w, 8w + 8) s containing
  the frame (w = floor(i / 32)); segments via `qwen3_text/text_llm.py` `load_segments` (untimed chunks kept), window
  text via `src/video_inputs.py` `window_text` (`../transcript_windows.py`). Tokenizer: ImageBind's CLIP BPE with
  OpenAI-CLIP truncation (start token + first 75 text tokens + end token). ImageBind's own `SimpleTokenizer` cuts a
  long text at 77 tokens without keeping the end token, and the text head then pools an arbitrary token; the
  truncation here avoids that. Re-encoding "normal" / "hateful" reproduces the campaign's text embeddings
  (cosine 1.0000, norms 100.0000).
- **Score**: mean of p over the modalities available at the frame: vision and audio always, transcript only in a
  window with speech. No corpus statistic, no label, nothing selected.

Hosts and commands (code as committed in fbb664a; `--modalities` added in 7df9b45):
- Transcript embeddings and scoring: uoa-lab1 (sc474397), CPU only (`setsid nohup`, HateVideo env, torch 2.7.1),
  2026-10-09:
  ```bash
  python experiments/20261008_baselines/zs_imagebind_audio/zs_imagebind_av.py text --cpu --batch 64   # 19,681 windows, 22 min
  python experiments/20261008_baselines/zs_imagebind_audio/zs_imagebind_av.py score
  python experiments/20261008_baselines/zs_imagebind_audio/zs_imagebind_av.py score --modalities vision        # comparison
  python experiments/20261008_baselines/zs_imagebind_audio/zs_imagebind_av.py score --modalities vision,audio  # comparison
  ```
- DeHate vision embeddings: **uoa-campus2 (foscsmlprd02), one NVIDIA A100-SXM4-80GB**, Slurm 24651
  (`launch/campus_av.sbatch`, step `imagebind`, 2026-10-10 21:27–22:00 NZDT, commit 27e1c27), env
  `.cache/envs/lavad_tf449`, ffmpeg 6.0-static. The 1341 DeHate test videos were copied from uoa-lab1 into a
  temporary directory of the campus checkout (`runs/20261008_baselines/zs_imagebind_av/_staging_videos/DeHate/`) and
  deleted after the step; the embeddings were copied back to uoa-lab1 `data/imagebind_image_4fps/DeHate/` (see its
  `PROVENANCE.md`) and deleted on campus2. Lab GPUs were all taken (QOS limit of 2 jobs, held by other agents' jobs);
  the job waited about 25 h in the campus2 queue.
  ```bash
  python experiments/20261008_baselines/zs_imagebind_audio/zs_imagebind_av.py extract-image \
    --video-root runs/20261008_baselines/zs_imagebind_av/_staging_videos/DeHate \
    --out-dir runs/20261008_baselines/zs_imagebind_av/features/imagebind_image_4fps/DeHate --decoders 6   # 1932 s
  ```
  The job's own `verify-image` step failed (the text-embedding file was not copied to campus2), so the check was
  done on uoa-lab1 on the CPU instead (`verify-image --cpu`, float32; logs `zs_imagebind_av/verify_cpu_lab1*.out`):
  1. port vs the campaign cache, 5 HateMM cohort videos, ffmpeg 6.1.1 (uoa-lab1's): same frame counts, mean cosine
     ≥ .99993 per video, max |Δp| .029 (`verify_image_HateMM_sc474397_cpu_ffmpeg-6.1.1-3ubuntu5.json`);
  2. the same with campus2's ffmpeg 6.0-static binary (copied to `.cache/tools/ffmpeg-6.0-static-campus2/`):
     identical numbers, so the two ffmpeg builds decode the same frames
     (`verify_image_HateMM_sc474397_cpu_ffmpeg-6.0-static.json`);
  3. the campus2 A100 float16 DeHate embeddings vs a CPU float32 re-extraction with the campus ffmpeg, 3 DeHate
     cohort videos (0nmB0Mkx8Lb8, 13Oi7NWW3M4, 264CG_YSdWk): same frame counts, mean cosine ≥ .99994, max |Δp| .018
     (`verify_image_DeHate_sc474397_cpu_ffmpeg-6.0-static.json`).

Outputs: `runs/20261008_baselines/zs_imagebind_av/<DS>/` (`transcript_windows.jsonl` = text and p of every window
with speech; `predictions.jsonl`, `coverage.json`, `config.json`, `metrics.json`, `run.log`); comparison rows in
`<DS>/modalities_vision/` and `<DS>/modalities_vision_audio/`; `zs_imagebind_av/run.log`, `text_stage.out`.

Results (canonical evaluator, exact cohorts; pooled frame ROC-AUC / pooled frame PR-AUC / within-video macro
ROC-AUC). Nothing selected; not development-selected. The existing row of this method is audio-only, so two
comparison rows of the same script are added: vision only, and vision + audio (= the AV row without the transcript).

| corpus | vision + audio + transcript | audio only (row above) | vision only | vision + audio | source |
|---|---|---|---|---|---|
| HateMM | .6285 / .3400 / .5514 | .5654 / .2905 / .5256 | .5920 / .3142 / .5406 | .5959 / .3248 / .5274 | `runs/20261008_baselines/zs_imagebind_av/HateMM/{,modalities_vision/,modalities_vision_audio/}metrics.json` |
| HateClipSeg | .6035 / .5597 / .5345 | .5652 / .5122 / .5218 | .5923 / .5536 / .5298 | .6090 / .5582 / .5394 | same, `HateClipSeg/` |
| DeHate | .5616 / .0934 / .5213 | .5302 / .0820 / .4819 | .5541 / .0946 / .5223 | .5526 / .0919 / .4973 | same, `DeHate/` (audio only: `runs/20261008_baselines/zs_imagebind_audio/DeHate/metrics.json`) |

Frame pools: HateMM 116,975 frames / 215 videos (within defined on 84), HateClipSeg 113,002 / 118 (99), DeHate
441,345 / 1151 (222). Windows with speech: 3,443 / 3,141 / 13,097 (cut at 77 tokens: 138 / 72 / 293). Curve frames
(ceil(4 · duration), at least the GT length) in a window with speech: HateMM 108,139 of 117,186, HateClipSeg 99,534 of
113,116, DeHate 407,246 of 442,461. Videos without speech (transcript absent everywhere): 1 / 0 / 2. Videos without a
vision embedding: 0 / 0 / 0. F2: 0 / 0 / 0 (no fallback needed). Curve frames past the last decoded frame, holding
it: 205 / 146 / 748; past the last audio clip: 8 / 24 / 797.
