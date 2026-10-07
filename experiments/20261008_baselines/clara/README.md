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
  vLLM runs with prefix caching and the multimodal processor cache off and explicit image ids, and a guard makes any
  call to vLLM's content hasher fail.
- The packs store features in fp16; under `--bf16` autocast, `torch.cat` of two fp16 tensors in
  `FeatureProjector` raises ("Unexpected floating ScalarType in at::autocast::prioritize"). Features are upcast to
  fp32 when loaded (values unchanged).
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

Hosts: HateMM and HateClipSeg on uoa-lab3 (sc474398), DeHate on uoa-lab2 (sc474399).
Caches: `data/clara_raw/<DS>/` (clips, frames, OCR), `data/clara_rationale/<DS>/Qwen/`, `data/clara_emb/<DS>/`.
Outputs: `runs/20261008_baselines/clara{,_norationale}/<DS>/seed<k>/` (`run.log`, `config_snapshot.json`,
`config.json`, `train_history.json`, `window_scores.json`, `predictions.jsonl`, `coverage.json`, `metrics.json`).

## Cost

VLM calls: 2 per video (train + val + test), the published cost: HateMM 1068 videos, HateClipSeg 394, DeHate 6499.
OCR: about 40 frames per video plus the window frames of test videos (about 40 per test video).

## Results

Pending (jobs queued 2026-10-08; see the coordinator's report).
