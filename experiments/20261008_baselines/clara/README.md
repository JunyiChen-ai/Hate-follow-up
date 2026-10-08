# CLARA scored per 8-s window (weakly supervised group)

CLARA: "Clip-Level Multimodal Alignment with VLM-Derived Rationales for Hateful Video Detection", ACM MM 2026
(acceptance stated by the authors; the DOI does not resolve yet), code github.com/yuchenzhang-1/CLARA, cloned
unchanged to `third_party/CLARA` at commit `468a6bc`. A video-level detector: speech-segment clips (Whisper audio,
ViT frames, BERT transcript and OCR) fused per clip by a mixture of experts, a gated video transformer over the clips
plus 8 rationale tokens from a two-step Qwen3-VL-8B read of the whole video, clip-segment contrastive loss.

Mechanism hypothesis: none (a baseline). Group: weakly supervised (video-level labels of the target corpus's train
split; user decision 2026-10-08).

## What is run

1. **Train as published** on each corpus's train split (whole videos, video-level labels), with `run_clara.sh`'s
   settings (BUDGET 40, text model bert, rationale Qwen, batch 32, 50 epochs, lr 3e-5, wd 1e-4, warm-up 0.05,
   early stopping 10 on val accuracy and `load_best_model_at_end`, MoE 8 experts top-3, GVT 4 layers, contrastive
   weight 0.3, tau 0.1, bf16). Seeds 2025, 234, 3407.
2. **Score each test window as a short video** (consecutive 8-s windows from 0, last one shorter; our own grid);
   window score = softmax(logits)[1].
3. **4 fps:** every frame takes its window's score (window i // 32).
4. **Check (coordinator, 2026-10-08):** the paper's "w/o Rationale" ablation, `--gvt_rationale_mode none`, same
   seeds, reported as `CLARA_win8_norationale` (`runs/20261008_baselines/clara_norationale/`).

## Exact per-window input

| input | whole video (train / val) | 8-s window [a, b) (test) |
|---|---|---|
| clips | authors' `get_video_clip.py` functions on the video's Whisper segments (speech clip per segment, silences >= 1 s as silent clips, shorter gaps merged, wav duration authoritative) | the same functions on the window's segments: Whisper segments cut at a and b, text sliced proportionally at word boundaries (`spvl.py` `window_text` rule), times relative to a, total duration b - a |
| frames | budget 40 allocated to clips in proportion to duration, bin-centre frames (`get_frame.py`), JPEG quality 95 | the video's own sampling rate: n = min(40, max(1, round(40 (b - a) / D))) frames, D = the video's wav duration, allocated to the window's clips by the same rule, bin-centre frames |
| audio | per clip: wav slice, Whisper large-v3 encoder over the 30-s padded input, mean | per window clip: the wav slice at absolute times a + start .. a + end, same encoder |
| transcript | per clip, BERT CLS (empty / silent clips masked) | per window clip, BERT CLS of the sliced text |
| OCR | PaddleOCR (PP-OCRv5 server det + rec, paddleocr 3.4.0) on the clip's budget-40 frames, words with score >= 0.5, BERT CLS mean | the same on the window clip's frames |
| rationale | one two-call Qwen3-VL-8B read per video: 20 frames, VIDEO_TITLE, VIDEO_DESCRIPTION, full transcript; 8 fields, BERT CLS | **the video's rationale, shared by all its windows** (coordinator decision: the rationale is a video-level read; regenerating it per window would cost 2 VLM calls per window). Every window therefore carries a video-level term; the w/o-rationale check measures it. |

Rationale details: the 20 frames are taken uniformly (k = 20) from the video's 100-frame budget in clip order
(`get_frames_for_rationale.py`, budget dir `frame_100` "for my model"). The release's defaults would index k = 32
frames and the rationale script would then keep the first 20, i.e. only the first 62 % of the video; k = 20 follows
the prompt's own wording ("20 frames sampled uniformly in temporal order"). Title / description: DeHate
`DeHate_labels.csv` `title` / `desc`; HateMM and HateClipSeg have none and get "N/A" (the authors' value for a
missing field). Generation: temperature 0.2, top-p 0.9, max 2048 new tokens, top-k 20 and repetition penalty 1.0
from the model's generation_config (as HF `generate` applies them), served with vLLM 0.11.0.

## Changes needed to run the release (recorded)

- Hash ban: the collator seeds each video's contrastive segment sampling with md5(video_id). The seed term is the
  index of the id in the sorted sample-id list instead; the md5 helpers are replaced by functions that raise.
  vLLM runs with prefix caching and the multimodal processor cache off; requests are added with
  `tokenization_kwargs=None` (vLLM's `LLM.generate` passes `{"truncation": False}`, which makes it key every image
  by a content hash even when an id is given); a guard makes any call to vLLM's content hasher fail once the engine
  is up (its start-up memory profiling keys synthetic dummy images only).
- DeepSpeed is installed in the HateVideo env without a CUDA toolkit; accelerate imports it inside
  `Trainer.__init__` and the import fails on a GPU node (Slurm 324). accelerate is told DeepSpeed is unavailable
  (it is not used).
- The packs store features in fp16; under `--bf16` autocast, `torch.cat` of two fp16 tensors in
  `FeatureProjector` raises ("Unexpected floating ScalarType in at::autocast::prioritize"). Features are upcast to
  fp32 when loaded (values unchanged).
- Rationale throughput on DeHate (2026-10-09). Many DeHate videos are 720p-1080p vertical; at native resolution
  (the authors' processor setting, unchanged) 20 frames give 17.6k-40.8k image tokens per prompt (HateMM /
  HateClipSeg frames: about 6k), against a 55k-token KV cache on the 5090. vLLM v1 admits requests whose prompts
  fit and, when the cache runs out during decoding, silently preempts a request and recomputes it from zero; with
  several large prompts this repeated (Slurm 318: 4 s/video on small frames, then 37, 90, 98 and 185 s/video per
  64-video batch; image-token throughput fell from 2.5k to 0.25k tokens/s). Generation lengths were normal (step A
  median 1870 characters, no run to the 2048-token cap) and transcripts were shorter than earlier ones, so neither
  output length nor text length was the cause. Fix: requests are admitted only when prompt + 2048 new tokens fit the
  KV cache not reserved by running requests (no preemption can occur), step B is queued as soon as step A finishes,
  and frames are decoded only on admission. Inputs, prompts, sampling and seeds are unchanged; the 896 rationales of
  Slurm 318 are kept and the job resumed from them (Slurm 328).
- Whisper clips of one sample are encoded in a batch (same per-clip arithmetic). The Qwen3-Embedding text variants
  are not computed (unused by the bert configuration). OCR runs only on the frames that enter the embeddings.
- `dataloader_num_workers` 8 -> 4; the per-microbatch MoE / GVT diagnostic JSONL logs are not written.
- Environments: `.cache/envs/hv_cv2` (torch 2.7.1+cu128), `.cache/envs/vllm_q3vl` (vllm 0.11.0, torch 2.8.0+cu128,
  transformers 4.57.6), `.cache/envs/paddleocr` (paddlepaddle-gpu 3.2.0 cu129, paddleocr 3.4.0); built by
  `setup_envs.sh`. RTX 5090 needs cu128+ builds.
- Transcripts: existing Whisper large-v3 segments (`data/asr_whisper_large_v3/<DS>/all_splits_chunks.jsonl`), the
  model the authors use; untimed chunks kept (end = next start or duration).
- AV1 videos (HateClipSeg webm/mkv) are decoded with PyAV when OpenCV cannot.

Train / val videos the authors' extractor skips (no wav, no clips, no rationale) are skipped
(`runs/20261008_baselines/clara/<DS>/embed/embed_report.json`). Test fallbacks (`run_plan.md` §1.3 F1): a test video
without an audio stream gets digital silence; a failed rationale becomes the all-blank rationale (masked).

## Splits

As SAGE: `data/weaksup_video_splits/<DS>.json`; HateMM 744 / 109, HateClipSeg p11 237 / 39 (offensive union),
DeHate official 4680 / 668; test = exact cohorts 215 / 118 / 1151; train ∪ val ∩ test = ∅ by id comparison
(`runs/20261008_baselines/sage_clara_splits/split_check.json`). CLARA's released 5-fold splits are not used.

## Commands

```bash
bash experiments/20261008_baselines/clara/setup_envs.sh                         # once per machine (CPU)
.cache/envs/hv_cv2/bin/python experiments/20261008_baselines/clara/clara_prep.py --dataset <DS> --workers 8   # CPU
.cache/envs/paddleocr/bin/python experiments/20261008_baselines/clara/clara_ocr.py --dataset <DS> --device cpu   # optional CPU head start
sbatch experiments/20261008_baselines/launch/clara_<machine>.sbatch <DS>      # rationale, OCR rest, embed, 2 x 3 trainings, eval
```

Hosts: HateMM and HateClipSeg: clips, frames and OCR (CPU) on uoa-lab3 (sc474398), copied to uoa-lab1 (sc474397)
where the GPU stages ran (the lab3 GPU was held by a long job of another agent); DeHate entirely on uoa-lab2
(sc474399).
Caches: `data/clara_raw/<DS>/` (clips, frames, OCR), `data/clara_rationale/<DS>/Qwen/`, `data/clara_emb/<DS>/`.
Outputs: `runs/20261008_baselines/clara{,_norationale}/<DS>/seed<k>/` (`run.log`, `config_snapshot.json`,
`config.json`, `train_history.json`, `window_scores.json`, `predictions.jsonl`, `coverage.json`, `metrics.json`).

## Cost

VLM calls: 2 per video (train + val + test), the published cost: HateMM 1068 videos, HateClipSeg 394, DeHate 6499.
OCR: about 40 frames per video plus the window frames of test videos (about 40 per test video).

## Results

Weakly supervised (video-level labels), 8-s windows, exact cohorts; transcribed from
`runs/20261008_baselines/{clara,clara_norationale}/<DS>/seed<k>/metrics.json` (summary:
`runs/20261008_baselines/sage_clara_summary.json`).

| corpus | variant | seed | frame ROC-AUC | frame PR-AUC | within-video ROC-AUC |
|---|---|---|---|---|---|
| HateMM (215) | CLARA | 2025 / 234 / 3407 | .8753 / .8755 / .8657 | .6377 / .6336 / .6208 | .5439 / .5335 / .5478 |
| HateMM | CLARA | mean (sd) | .8722 (.0056) | .6307 (.0088) | .5417 (.0074) |
| HateMM | w/o rationale | 2025 / 234 / 3407 | .5405 / .5679 / .5476 | .2735 / .2883 / .2759 | .5705 / .5526 / .5282 |
| HateMM | w/o rationale | mean (sd) | .5520 (.0142) | .2792 (.0079) | .5504 (.0212) |
| HateClipSeg (118) | CLARA | 2025 / 234 / 3407 | .5441 / .5430 / .5367 | .5154 / .5161 / .5180 | .5048 / .5022 / .5079 |
| HateClipSeg | CLARA | mean (sd) | .5413 (.0040) | .5165 (.0013) | .5050 (.0028) |
| HateClipSeg | w/o rationale | 2025 / 234 / 3407 | .5360 / .5410 / .5350 | .5117 / .5147 / .5072 | .5233 / .5271 / .5192 |
| HateClipSeg | w/o rationale | mean (sd) | .5373 (.0032) | .5112 (.0037) | .5232 (.0039) |

The w/o-rationale check shows where CLARA's pooled score comes from on HateMM: the shared video-level rationale
tokens lift pooled ROC-AUC from .55 to .87 and PR-AUC from .28 to .63, while within-video ROC-AUC stays at
.54-.55 either way. The rationale is identical for all windows of a video, so it ranks videos, not moments.

HateMM: rationale (1068 videos, 1.8 h) and embeddings in Slurm 324 on uoa-lab1; training in Slurm 324 failed on the
DeepSpeed import (fixed), and the six trainings then ran on the uoa-lab1 CPU (`clara_train.py --cpu`, bf16 autocast on
CPU, all settings unchanged; `runs/20261008_baselines/clara/train_cpu_HateMM_lab1.log`) because no GPU slot was free
under the per-user 2-GPU limit. Selected epochs (val accuracy): CLARA 12 / 14 / 23 (.851 / .860 / .869), w/o rationale
4 / 8 / 2 (.664 / .673 / .664). Coverage 215 / 215 videos, 3768 windows, no fallback; 2 train videos without an
audio stream skipped as in the authors' extractor.

HateClipSeg: uoa-lab1, Slurm 324, 2026-10-08 18:28-19:41 (rationale 394 videos in 50 min). On the p11 val split
(34 hateful / 5 normal) every run predicts all videos hateful from the first epoch (val accuracy .8718), so early
stopping (patience 10 on val accuracy) keeps the epoch-1 checkpoint in all six runs; this is the published selection
rule applied to this split. Coverage 118 / 118 videos, 3591 windows, no fallback.

DeHate: Slurm 318 on uoa-lab2, started 2026-10-08 19:40 (rationale about 5 s/video for 6499 videos, then the OCR
not yet done by the parallel CPU pass, embeddings, six trainings, evaluation); not finished at the time of writing.
