# Pilot Results: Label-Free Hateful Event Localization

All results use eight duration-balanced videos per dataset and the frozen 4 FPS
ground truth. They are mechanism-selection pilots, not final benchmark claims.

| Method | HateMM ROC / within / F1@.5 | MHC | MHC-ZH | HateClipSeg | Four-set average ROC / within / F1 |
|---|---|---|---|---|---|
| A01--A05 prompt transplants | .500 / .500 / .000 (typical) | .500 / .500 / .000 | mostly .500 / .500 / .000 | mostly .500 / .500 / .000 | failed intrinsic localization gate |
| A06/A07 sparse mechanisms | .500 / .500 / .000 | .500 / .500 / .000 | .928 / .500 / .667 | <=.500 / .500 / .000 | pooled artifact; failed gate |
| A01v4 binwise timed multimodal | .746 / .087 / .000 | .900 / .633 / .000 | .863 / .496 / .286 | .737 / .493 / .057 | .812 / .427 / .086 |
| A12 official TimeLens-8B | .359 / .500 / .000 | .624 / .696 / .182 | .814 / .593 / .250 | .501 / .473 / .090 | .574 / .565 / .130 |
| Symmetric rank fusion (post-hoc) | .260 / .087 / .000 | .708 / .659 / .095 | .708 / .204 / .000 | .440 / .530 / .000 | .529 / .370 / .024 |

## Decision

- Reject whole-video transcript prompting: it produces empty or constant curves.
- Retain independent timestamped multimodal windows for semantic scoring.
- Retain official TimeLens as the strongest boundary/localization prior.
- Reject direct symmetric rank fusion: incompatible curve semantics degrade all averages.
- Preserve the 611 videos outside this pilot as the untouched confirmation set.

## Integrity note

The pilot test labels have now been opened. These 32 videos must not be included in
the untouched confirmation set used for final claims.
# A08 VideoTGB Pilot (official LSTP-7B checkpoint)

| Dataset | pooled ROC-AUC | PR-AUC | within-video ROC-AUC | interval F1@0.5 |
|---|---:|---:|---:|---:|
| HateMM | 0.755 | 0.063 | 0.775 | 0.667 |
| MHC | 0.845 | 0.277 | 0.826 | 0.000 |
| MHC-ZH | 0.422 | 0.137 | 0.500 | 0.000 |
| HateClipSeg | 0.574 | 0.451 | 0.544 | 0.067 |
| Macro mean | 0.649 | 0.232 | 0.661 | 0.183 |

The 32-video pilot used the released 8.05B-parameter LSTP-7B checkpoint with
no missing or unexpected state-dict keys. Results are promising but highly
heterogeneous: strong on HateMM/MHC, weak on MHC-ZH, and lower mean PR-AUC than
the current T3AL reference. One HateClipSeg prediction ID is not defined in the
119-video GT archive and is reported as skipped by the evaluator.

# A09 GroundingGPT Adaptation Audit

Both the initial explicit `[start,end]/none` prompt and a second prompt matched
to the released Charades-STA training template produced `nobody` on the pilot's
HateMM hateful example. The failed attempts are preserved as append-only v1/v2
prediction files. A09 is therefore not advanced to a full pilot in its current
released-checkpoint form.

# A11 Seq2Time Mechanism Transplant

The relative-position-token (`0000..9999`) Qwen3-VL transplant completed all 32
records, with five video-decode errors and only one non-empty prediction. Its
four-dataset pooled ROC-AUCs are approximately 0.5 and all interval F1@0.5
scores are zero. This adaptation fails the pilot gate. It is not an official
Seq2Time reproduction because the CVPR work has no released code/checkpoint.

# Untouched Confirmation Results

The 611-video confirmation split excludes the 32-video pilot. All three
advanced methods produced 611 unique predictions with zero recorded errors.

| Method | Mean pooled ROC | Mean PR | Mean within-video ROC | Mean F1@0.5 | Decision |
|---|---:|---:|---:|---:|---|
| A08 VideoTGB | 0.521 | 0.300 | 0.539 | 0.115 | reject |
| A10 Vid-Group | 0.520 | 0.318 | **0.615** | **0.163** | retain as boundary baseline |
| A12 TimeLens-8B | **0.573** | **0.329** | 0.580 | 0.063 | retain as semantic grounding baseline |

## A12 per-dataset confirmation

| Dataset | pooled ROC | PR | within-video ROC | F1@0.5 |
|---|---:|---:|---:|---:|
| HateMM | 0.515 | 0.262 | 0.569 | 0.061 |
| MHC | 0.594 | 0.282 | 0.638 | 0.075 |
| MHC-ZH | 0.632 | 0.262 | 0.583 | 0.037 |
| HateClipSeg | 0.551 | 0.508 | 0.529 | 0.081 |

No adapted method exceeds the preregistered T3AL reference on the overall
pooled metrics. A10 is the strongest interval/boundary baseline, while A12 is
the strongest pooled semantic-localization baseline; they expose complementary
failure modes suitable for follow-up work.
