# SAGE scored per 8-s window (weakly supervised group)

SAGE: "Synergistic Adaptive Gating of Experts for Hateful Video Detection", ACL 2026
(https://aclanthology.org/2026.acl-long.817/), code github.com/XinLiao04/SAGE, cloned unchanged to
`third_party/SAGE` at commit `3545f6d`. A video-level detector: text (XLM-R twitter-sentiment, 128 tokens), audio
(MFCC 128 x 2048 frames) and video (VideoMAE-base on 16 frames) experts, global deliberation, gated decision.

Mechanism hypothesis: none (a baseline). Group: weakly supervised (trained with video-level labels of the target
corpus's train split; user decision 2026-10-08).

## What is run

1. **Train as published** on each corpus's train split with video-level labels, epoch chosen on val macro-F1 (the
   authors' `Trainer.train_model` rule), authors' hyper-parameters (`config/config.yaml`: 30 epochs, batch 16, AdamW
   lr 1e-4, wd 5e-5, 3 warm-up epochs then cosine, losses fusion + 0.5 expert + 0.05 gate). Seeds 2025, 234, 3407
   (training is cheap).
2. **Score each test window as a short video.** Windows are consecutive 8-s windows from 0 (the last may be shorter),
   the grid of our own method. Window score = softmax(logits_fusion)[1] = P(hateful).
3. **4 fps:** frame i (centre (i + 0.5)/4 s) takes the score of the window containing it (window i // 32); curve
   length ceil(4 * duration), cut to the GT length by the evaluator.

## Exact per-window input

| input | whole video (train / val) | 8-s window [a, b) (test) |
|---|---|---|
| frames | the authors' `frame-extract.py` sampler on the whole stream: `frame_count / 16` interval, 16 frames, saved as JPEG (cv2 default quality 95), read back as `read_image` | the same sampler applied to the frames whose timestamp i / fps lies in [a, b); 16 frames per window |
| video transform | `VideoTransform` train branch (resize 224, RandomResizedCrop 0.8-1, flip 0.5, normalise) | `VideoTransform` eval branch (resize 224, normalise) |
| audio | 16 kHz mono wav of the video, `AudioEncoder`: MFCC 128, first 2048 frames (20.48 s), zero-padded | the wav samples [round(16000 a), round(16000 b)), same MFCC, 800 frames + zero padding to 2048 |
| text | Whisper large-v3 transcript of the video (`text`), first 128 XLM-R tokens | the transcript inside [a, b): Whisper segments sliced proportionally at word boundaries (the rule of `experiments/20260910_spvl/spvl.py` `window_text`); empty string when nobody speaks |

Train/test mismatch to keep in mind: training sees the first 20.48 s of audio and the first 128 tokens of each whole
video; a window sees 8 s of audio and its own words.

Text recipe: the authors' HateMM recipe (transcript only) for all three corpora; MultiHateClip's title + description
recipe is not used because those fields are video-level (they would put the same text in every window) and HateMM /
HateClipSeg have none.

## Changes needed to run the release (recorded, nothing else changed)

- `Trainer.evaluate` reads `batch_size` before assignment and `train_model` passes it a DataLoader instead of a
  dataset; the release cannot train as is. The training loop is re-written in `sage_run.py` with the same steps
  (shuffle, forward, `CustomLoss`, AdamW step, val macro-F1 per epoch, save when strictly better, scheduler step per
  epoch).
- The release passes the single config flag `train: True` to the train, val and test datasets, so val and test frames
  also get random crop / flip. Val and test use the eval branch of the authors' `VideoTransform` here.
- The frozen encoders (VideoMAE, XLM-R, MFCC) run batched in the collate step instead of once per sample; each
  sample's arithmetic is the authors'. From 2026-10-08 06:45 the CPU part (frame load + `VideoTransform`, wav load)
  runs in 3 DataLoader workers (checked identical to the synchronous path on HateMM val samples); DeHate seed 2025
  ran with the synchronous loop (job 299), later runs with the workers. Only the augmentation random streams differ.
- `nn.MultiheadAttention` in GED is called with `need_weights=False`: the release materialises the (discarded)
  attention weights, which does not fit a 32 GB GPU at batch 16 (OOM, job 293); the output is the same (max abs
  difference 7e-8 on a random input). Resized 224 x 224 frames are cached (resize is the first transform op, and
  resizing a 224 x 224 frame again is the identity, checked).
- Kept as published: frames are normalised with ImageNet mean/std without dividing by 255 (`read_image` gives uint8);
  the first epoch runs at learning rate 0 (warm-up step 0).
- `preprocess/audio-extract.py` in the release is a copy of the Whisper script. The wavs come from the CLARA authors'
  ffmpeg command (16 kHz, mono, PCM16) into `data/wav16k_mono/<DS>/`; 40 / 40 HateMM files checked sample-identical
  to the existing 16 kHz wavs (`~/Retrieval-hate/data/AV2A_wav`).
- Environment: torch 2.7.1+cu128 / transformers 4.57.6 (HateVideo env) + opencv-python-headless 4.12.0.88 in
  `.cache/envs/hv_cv2`; the authors pin torch 2.6+cu124, which has no RTX 5090 (sm_120) kernels.
- AV1 videos (31 HateClipSeg webm/mkv) cannot be decoded by OpenCV's bundled FFmpeg; they are read with PyAV
  (libdav1d) in the same sequential frame order.
- Whisper: the authors use Whisper turbo; the existing Whisper large-v3 transcripts are reused
  (`data/asr_whisper_large_v3/<DS>/all_splits_chunks.jsonl`).

Train / val videos the authors' loader would drop (fewer than 16 frames or no audio file) are dropped: HateMM
non_hate_video_559 / 585 (no audio stream), DeHate 7310121949189999878, 7384178426288164139, 8bAfN6vXoIZp (fewer
than 16 frames) — see `runs/20261008_baselines/sage/<DS>/prep/prep_report.json` and the train `run.log`.

Fallbacks (`run_plan.md` §1.3): F1 tail — a window with fewer than 16 readable frames (only a last window shorter
than about 0.6 s) takes the previous window's score; a video without an audio stream gets the authors' zero MFCC.
F2 is never expected (every cohort video decodes).

## Splits (video-level labels only; selection on val)

`data/weaksup_video_splits/<DS>.json` (`experiments/20261008_baselines/detwin/prepare.py`, provenance in that
directory). HateMM 744 / 109 (Retrieval-hate `hatemm_{train,val}.txt`, the earlier weakly supervised split; SAGE
released no HateMM split), HateClipSeg p11 237 / 39 (label = offensive union, the HCS GT rule), DeHate official
4680 / 668. Test = the exact cohorts (215 / 118 / 1151). train ∪ val ∩ test = ∅ for all three, checked by id
comparison: `runs/20261008_baselines/sage_clara_splits/split_check.json`.

## Commands

```bash
# CPU (setsid nohup), on the machine that holds the corpus's videos
python experiments/20261008_baselines/detwin/prepare.py splits; python experiments/20261008_baselines/detwin/prepare.py asr  # lab1 once
.cache/envs/hv_cv2/bin/python experiments/20261008_baselines/sage/sage_run.py prep --dataset <DS> --workers 8
# GPU
sbatch experiments/20261008_baselines/launch/sage_<machine>.sbatch <DS>   # train 3 seeds, infer, evaluate
```

Hosts: HateMM and HateClipSeg on uoa-lab1 (sc474397), DeHate on uoa-lab2 (sc474399).
Outputs: `runs/20261008_baselines/sage/<DS>/seed<k>/` (`run.log` first line = host, `config_snapshot.yaml`,
`config.json`, `train_history.json`, `predictions.jsonl`, `coverage.json`, `metrics.json` from
`src/eval/evaluate_four_datasets.py`); window scores of all seeds in `runs/20261008_baselines/sage/<DS>/infer/`.

## Results

Pending (jobs queued 2026-10-08; see the coordinator's report).
