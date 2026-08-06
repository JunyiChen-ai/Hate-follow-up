# Held-out test note: the channel-restoration method on four benchmarks

Measurement, not a kill-test. No pre-registration governs these runs and none is claimed. Each benchmark is measured on its own `test_clean` split under the configuration frozen during the train-split work; nothing was refitted for the test data, and train and test scores are never pooled into one distribution.

## What was run

One condition per dataset per model size: source media pulled from B2 and byte-verified, audio re-transcribed with Whisper large-v3 on cuda, the frozen degeneracy gate applied, the gated fresh transcript fed uncapped to a single judge call, and a label-free KDE-valley threshold computed on that run's own raw-z distribution. One MLLM call per video per arm. The 2B arm is the boundary-condition contrast, not a second method.

Every component is inherited rather than chosen here:

- **audio** — scripts/duplex/crossbench_audio.py, which imports ffprobe, extract_wav and the 1800 s cap from scripts/duplex/channel_restoration_audio.py unmodified; ffmpeg 16 kHz mono, silero-VAD speech fraction
- **asr** — scripts/duplex/crossbench_asr.py, which imports the model id and the text statistics from scripts/duplex/channel_restoration_asr.py unmodified: openai/whisper-large-v3, transformers ASR pipeline, fp16 on cuda (asserted), language auto (so MHClip_ZH resolves to zh), chunk_length_s=30 long-form, batch_size=8
- **gate** — scripts/duplex/crossbench_gate.py, which imports collapse_repeats and both bounds from scripts/duplex/channel_restoration_gate.py unmodified: clause-level repetition collapse (max_ngram 8), reject iff vad_speech_frac < 0.05 AND gzip ratio of raw fresh text > 7
- **judge** — src/duplex/extract_duplex_readout.py, unmodified: frozen prag reader, 16 frames from frames_16, max_pixels 100352, single forward pass, raw unclipped z = logsumexp(Yes ids) - logsumexp(No ids), transcript_limit 0 (uncapped)
- **threshold_recipe** — docs/duplex/reports/rawz_detector_train.json, estimator b_kde_valley, reimplemented here and verified against that document's C0 value

The threshold recipe was self-checked before use: applied to the ImpliHateVid C0 scores it returns -2.903500 against the frozen -2.903500 in `rawz_detector_train.json` (match: True).

## Restoration coverage

How much of the judge input the channel-restoration stage actually replaced. A low fraction here means the method degenerated to its fallback branch and the numbers below measure the judge and the threshold alone.

| dataset | n | with source media | usable audio | fresh transcripts | gate accepted | gate rejected | no fresh pass | restoration fraction |
|---|---|---|---|---|---|---|---|---|
| ImpliHateVid | 400 | 401 | 401 | 401 | 393 | 8 | 0 | 0.9800 |
| HateMM | 215 | 215 | 215 | 215 | 201 | 14 | 0 | 0.9349 |
| MHClip-EN | 161 | 161 | 161 | 161 | 155 | 6 | 0 | 0.9627 |
| MHClip-ZH | 149 | 149 | 149 | 149 | 147 | 2 | 0 | 0.9866 |

## Headline

AUC is hateful vs normal on the raw z. macro-F1 and accuracy are at the label-free valley computed on that arm's own test distribution; the oracle column is the F1-maximizing threshold chosen against gold labels and is a diagnostic ceiling, never part of the method.

| dataset | model | n | prev. | AUC | AUC 95% CI | modes | valley | macro-F1 (label-free) | acc | FN | FP | oracle macro-F1 | gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ImpliHateVid | Qwen3-VL-8B | 400 | 0.497 | 0.9473 | [0.925, 0.966] | 2 | -4.455 | 0.8823 | 0.8825 | 15 | 32 | 0.8925 | 0.0101 |
| ImpliHateVid | Qwen3-VL-2B | 400 | 0.497 | 0.9199 | [0.892, 0.945] | 2 | 0.483 | 0.8023 | 0.8075 | 5 | 72 | 0.8230 | 0.0208 |
| HateMM | Qwen3-VL-8B | 215 | 0.400 | 0.9232 | [0.881, 0.959] | 2 | -2.346 | 0.6562 | 0.6605 | 3 | 70 | 0.8879 | 0.2317 |
| HateMM | Qwen3-VL-2B | 215 | 0.400 | 0.8912 | [0.844, 0.933] | 1 | -- | -- | -- | -- | -- | -- | -- |
| MHClip-EN | Qwen3-VL-8B | 161 | 0.304 | 0.7847 | [0.714, 0.853] | 2 | -0.503 | 0.6962 | 0.7267 | 16 | 28 | 0.6888 | -0.0074 |
| MHClip-EN | Qwen3-VL-2B | 161 | 0.304 | 0.7442 | [0.658, 0.827] | 1 | -- | -- | -- | -- | -- | -- | -- |
| MHClip-ZH | Qwen3-VL-8B | 149 | 0.302 | 0.8547 | [0.785, 0.912] | 2 | 0.607 | 0.7256 | 0.7383 | 6 | 33 | 0.7932 | 0.0676 |
| MHClip-ZH | Qwen3-VL-2B | 149 | 0.302 | 0.7917 | [0.716, 0.861] | 1 | -- | -- | -- | -- | -- | -- | -- |

### Valley stability

1000-resample bootstrap of the valley location, plus the relative trough depth: the fraction by which the valley's density falls below the lower of the two modes bracketing it. 1.0 is a clean split, 0 is a valley sitting exactly at mode height, and a dash means the recipe found fewer than two modes and returned no threshold at all.

| dataset | model | trough depth | valley 95% CI | sd | resamples with no valley |
|---|---|---|---|---|---|
| ImpliHateVid | Qwen3-VL-8B | 0.604 | [-6.51, -1.78] | 1.20 | 0/1000 |
| ImpliHateVid | Qwen3-VL-2B | 0.045 | [0.12, 1.11] | 0.23 | 91/1000 |
| HateMM | Qwen3-VL-8B | 0.283 | [-5.14, 0.40] | 1.43 | 1/1000 |
| HateMM | Qwen3-VL-2B | -- | [-1.50, 1.65] | 0.85 | 595/1000 |
| MHClip-EN | Qwen3-VL-8B | 0.082 | [-5.80, 3.35] | 2.29 | 92/1000 |
| MHClip-EN | Qwen3-VL-2B | -- | [-1.24, 2.08] | 1.06 | 489/1000 |
| MHClip-ZH | Qwen3-VL-8B | 0.162 | [-2.80, 2.39] | 1.46 | 20/1000 |
| MHClip-ZH | Qwen3-VL-2B | -- | [-1.62, 0.84] | 0.81 | 454/1000 |

### ImpliHateVid: where the misses sit

The gold EX/IM/NH prefixes are diagnostic ground truth for this table only; no scoring component reads them. The channel-restoration story predicts the residual misses concentrate in the implicit stratum.

| model | AUC EX vs NH | AUC IM vs NH | FN EX | FN IM | miss rate EX | miss rate IM | share of FN that is IM |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | 0.9731 | 0.9255 | 0 | 15 | 0.000 | 0.139 | 1.000 |
| Qwen3-VL-2B | 0.9436 | 0.8999 | 2 | 3 | 0.022 | 0.028 | 0.600 |

## MHClip-ZH anomaly diagnostic (2026-08-07)

The first MHClip-ZH test run reported 0.675 for the 8B against 0.8125 for the 2B, the only scale inversion across the four benchmarks, with restoration coverage at 1.0. Two explanations were on the table: the fresh Whisper Chinese transcripts had hurt the larger model, or the test split was simply harder. Neither is what happened.

**Root cause.** frames_16/<one MHClip_ZH test video>/frame_012.jpg was a partially written JPEG: 131,059 bytes against roughly 260,000 for every intact neighbour in the same directory. extract_duplex_readout.py opens all 16 frames with PIL.Image.open(p).convert('RGB') outside any try block. The truncated file raises OSError, which propagates out of main() and kills the process. The run script retries three times, and each retry resumes onto the same file and dies at the same video. The loop died at video 14 of 149. Both the 8B and the 2B arm therefore scored 13 videos, and the published AUCs rested on 5 positives and 8 negatives. A scan of every frame of every `test_clean` video across all four benchmarks found exactly one unreadable JPEG, which is why only this dataset was affected. The frame was re-decoded from the local source mp4 at the index the original extractor used; the same code path reproduces the intact neighbour frame at 37.61 dB PSNR, so the frame numbering agrees. The run was then repeated to full coverage and the reports above are the corrected ones.

**The four cells.** Every test cell is the same 149 videos, 45 hateful and 104 normal. The train row is 579 videos. Train under fresh Whisper was never measured: the train-split source media was not pulled to this machine, so the ASR route was never exercised there.

| split | transcript the judge read | 8B AUC | 8B 95% CI | 2B AUC |
|---|---|---|---|---|
| train | dataset | 0.8555 | [0.824, 0.886] | 0.7615 |
| train | fresh Whisper | not measured | -- | -- |
| test | dataset | 0.8665 | [0.805, 0.919] | 0.7870 |
| test | fresh Whisper (auto language) | 0.8547 | [0.787, 0.909] | 0.7917 |
| test | fresh Whisper (language forced zh) | 0.8577 | [0.791, 0.912] | -- |

**Did restoration hurt?** No, not measurably. On the same 149 videos the fresh transcript moves the 8B AUC by -0.0118, paired bootstrap [-0.0334, 0.0083], straddling zero; the 2B moves 0.0047. The 8B raw z ranks the two arms at Spearman 0.975 even though the median normalized edit distance between the two transcripts is 0.43. The text changes substantially; what the judge does with it does not.

**Is the test split different?** No. Holding the transcript fixed at the dataset one, the 8B scores 0.8555 on train and 0.8665 on test, a difference of 0.0110 with heavily overlapping intervals.

**Language mismatch.** Whisper's own per-chunk language vote came back empty for all 149 videos, so language was read off the Unicode script profile instead: 24 of 149 auto-detected transcripts (0.161) are not Han-dominant, which fired the pre-registered trigger for a forced-zh arm. Forcing zh rescues 20 of those 24 and lifts the median Han fraction to 1.0, and moves the 8B AUC by 0.0030, [-0.0078, 0.0140]. The mechanism is real and the consequence is nil, so automatic detection stays: forcing a language would buy nothing and would oblige the method to make a correct per-corpus language decision on every new corpus.

**What this says about the method on non-English corpora.** Channel restoration is neutral on MHClip-ZH, not harmful and not helpful. The starvation premise it runs on is weak here: the dataset transcript already has a median of 76 characters and only 1.3 percent of videos exceed the 300-character window the old clipped instrument could see, so there is little starvation left to relieve. Report it as a neutral result on this corpus rather than as evidence against the mechanism, and do not tune the ASR to chase it.

**Infrastructure.** One unreadable frame silently cost 91 percent of a held-out measurement, and the run still reported `DONE` with a plausible-looking AUC on the surviving 13 videos. The judge should record and skip an unreadable frame rather than abort, and the analysis should refuse to write a report when coverage falls far below the split size instead of quietly reporting the subset.

Full statistics, including the transcript forensics and the score-movement anatomy: `docs/duplex/reports/test_zh_anomaly_diag.json`.

## Context

Two reference points, neither of them a like-for-like comparison.

**Supervised ceiling on ImpliHateVid.** IARE (SIGIR 2026) reports F1 91.75 on this same test split. It is trained on the split's labels; the method here sees none, at train time or at threshold time. The gap between the two is the price of dropping supervision, not a defect to be closed by tuning.

**The ImpliHateVid train row this was frozen on.** The train numbers come from the full-corpus C2 run over 1283 `train_clean` videos. They are quoted side by side, never pooled: each split's threshold is computed on its own distribution.

| model | train AUC | train macro-F1 | train oracle macro-F1 | test AUC | test macro-F1 |
|---|---|---|---|---|---|
| Qwen3-VL-8B | 0.9505 | 0.8796 | 0.8796 | 0.9473 | 0.8823 |
| Qwen3-VL-2B | 0.9230 | 0.8330 | 0.8501 | 0.9199 | 0.8023 |

## What these numbers do and do not settle

- The threshold is transductive: it reads the unlabeled test scores as a batch. That is legitimate for a label-free method and is how the train runs computed it too, but it is not an online per-video decision rule, and a deployment that scores one video at a time would need the threshold carried over rather than recomputed.
- A single arm per dataset per model. No repeated sampling, no ensembling; the run is one forward pass per video.
- The oracle column bounds how much of the remaining error is a threshold problem rather than a ranking problem. Where the gap is near zero, the label-free recipe has already found what the score supports and further threshold work is wasted effort.
- MHClip is 3-class collapsed to binary with `Offensive` mapping to 1. The `Hateful` and `Offensive` subgroup AUCs in the JSON say how much of the binary number rests on each.
- The surface-cue regex family used in the error anatomy is English. On MHClip-ZH it under-fires and those rates are not interpretable.

Per-arm JSON, with the full distributions, gate statistics and starvation diagnostics: `docs/duplex/reports/test_c2_<slug>_<arm>.json`.
