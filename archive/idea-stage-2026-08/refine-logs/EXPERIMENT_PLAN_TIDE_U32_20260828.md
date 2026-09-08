# TIDE U32 Mechanism Pilot

## Claim under test

Temporal isolation converts a frozen MLLM from a video-level judge into a
within-video hateful-event localizer, and aligned local visual evidence adds
information beyond timestamped transcript alone.

## Development cohort

`results/idea_discovery/melt/fresh_held32_manifest.jsonl` (8 videos per
dataset). This cohort has been inspected before and is development-only.

## Runs

1. `u32_joint`: local three-frame visual context plus local timestamped ASR.
2. `u32_visual`: identical local visual context without ASR.
3. `u32_text`: local ASR with a blank visual carrier.
4. `u32_unmasked_joint`: full-video canvas and full transcript supplied while
   asking about each center cell; this is the temporal-contamination control.

All arms use the same frozen Qwen3-VL-8B, 32 uniform center cells, content-free
Yes/No logit correction, zero threshold, and canonical 4 FPS rasterization.
No ground truth is read during inference.

## Metrics and gates

- Primary mechanism metric: within-video macro ROC-AUC.
- Boundary metrics: interval F1 at tIoU 0.3/0.5/0.7.
- Diagnostics: pooled ROC/PR, constant-curve rate, all-empty/all-full rate.

Continue to scale-persistence only if:

- joint within-video ROC >= 0.60 overall and HateClipSeg >= 0.55;
- joint exceeds text-only and unmasked by >= 0.02 on the dataset macro;
- joint produces non-zero F1@0.5 on at least two datasets;
- no single modality explains all gains.

Otherwise kill or narrow the multimodal/isolation claim before adding a more
complex decoder.
