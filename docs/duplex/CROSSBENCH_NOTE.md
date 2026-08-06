# Cross-benchmark note: the channel-restoration method on HateMM, MHClip-EN and MHClip-ZH

Measurement, not a kill-test. No pre-registration governs this run and none is claimed. Each benchmark is measured on its own `train_clean` split; no test split of any dataset was read.

## What was run

One condition per dataset per model size, the method as it stands after the ImpliHateVid full-corpus run: the gated fresh Whisper large-v3 transcript fed uncapped to a single judge call, with the dataset transcript as the gate's fallback, and a label-free KDE-valley threshold computed on that run's own raw-z distribution. One MLLM call per video. The 2B arm is the boundary-condition contrast, not a second method.

Every component is inherited rather than chosen here:

- **audio** — scripts/duplex/crossbench_audio.py, which imports ffprobe, extract_wav and the 1800 s cap from scripts/duplex/channel_restoration_audio.py unmodified; ffmpeg 16 kHz mono, silero-VAD speech fraction
- **asr** — scripts/duplex/crossbench_asr.py, which imports the model id and the text statistics from scripts/duplex/channel_restoration_asr.py unmodified: openai/whisper-large-v3, transformers ASR pipeline, fp16 on cuda (asserted), language auto (so MHClip_ZH resolves to zh), chunk_length_s=30 long-form, batch_size=8
- **gate** — scripts/duplex/crossbench_gate.py, which imports collapse_repeats and both bounds from scripts/duplex/channel_restoration_gate.py unmodified: clause-level repetition collapse (max_ngram 8), reject iff vad_speech_frac < 0.05 AND gzip ratio of raw fresh text > 7
- **judge** — src/duplex/extract_duplex_readout.py, unmodified: frozen prag reader, 16 frames from frames_16, max_pixels 100352, single forward pass, raw unclipped z = logsumexp(Yes ids) - logsumexp(No ids), transcript_limit 0 (uncapped)
- **threshold_recipe** — docs/duplex/reports/rawz_detector_train.json, estimator b_kde_valley, reimplemented here and verified against that document's C0 value

The threshold recipe was self-checked before use: applied to the ImpliHateVid C0 scores it returns -2.903500 against the frozen -2.903500 in `rawz_detector_train.json` (match: True).

## The restoration stage did not run on any of the three benchmarks

This is the first thing to read, and it bounds everything below. The channel-restoration component needs the source video to re-transcribe. The three benchmarks are present on this machine only as the frames-plus-annotation payloads; an exhaustive listing of the project B2 bucket (211,566 objects, single bucket, all prefixes) found source media for ImpliHateVid alone. What exists for HateMM and MHClip is pooled Whisper *encoder embeddings* from a different project, not text, and not usable as judge input. The source cluster that holds the mp4s is not reachable from this machine by key-based ssh.

So on all three benchmarks the gate took its `no_fresh_pass` branch for every video and the judge read the dataset transcript, uncapped. The numbers below therefore measure the judge and the label-free threshold travelling across benchmarks, at zero restoration coverage. They do not test whether channel starvation is an ImpliHateVid quirk or a general property — that question stays open until the source videos are available.

Everything is staged for that run. `crossbench_audio.py` and `crossbench_asr.py` take an `--mp4-dir`, the driver is idempotent, and pointing it at the videos re-runs stages A-C and re-scores into a fresh judge directory keyed by an input fingerprint.

## Headline

AUC is hateful vs normal on the raw z. macro-F1 is at the label-free valley computed on that arm's own distribution; the oracle column is the F1-maximizing threshold chosen against gold labels and is a diagnostic ceiling, never part of the method.

| dataset | model | n | prev. | AUC | AUC 95% CI | modes | valley | macro-F1 (label-free) | FN | FP | oracle macro-F1 | gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HateMM | Qwen3-VL-8B | 744 | 0.401 | 0.8760 | [0.849, 0.903] | 2 | -2.291 | 0.6429 | 20 | 243 | 0.8255 | 0.1826 |
| HateMM | Qwen3-VL-2B | 744 | 0.401 | 0.8293 | [0.799, 0.859] | 2 | 0.946 | 0.6565 | 36 | 219 | 0.7556 | 0.0991 |
| MHClip-EN | Qwen3-VL-8B | 550 | 0.307 | 0.8259 | [0.791, 0.859] | 2 | 2.597 | 0.7064 | 76 | 57 | 0.7158 | 0.0094 |
| MHClip-EN | Qwen3-VL-2B | 550 | 0.307 | 0.7313 | [0.687, 0.775] | 1 | -- | -- | -- | -- | -- | -- |
| MHClip-ZH | Qwen3-VL-8B | 579 | 0.311 | 0.8555 | [0.824, 0.886] | 2 | -1.657 | 0.7038 | 21 | 146 | 0.7386 | 0.0348 |
| MHClip-ZH | Qwen3-VL-2B | 579 | 0.311 | 0.7615 | [0.722, 0.801] | 1 | -- | -- | -- | -- | -- | -- |

### Valley stability

1000-resample bootstrap of the valley location, plus the relative trough depth: the fraction by which the valley's density falls below the lower of the two modes bracketing it. 1.0 is a clean split, 0 is a valley sitting exactly at mode height, and a dash means the recipe found fewer than two modes and so returned no threshold at all.

| dataset | model | trough depth | valley 95% CI | sd | resamples with no valley |
|---|---|---|---|---|---|
| HateMM | Qwen3-VL-8B | 0.145 | [-8.06, 2.49] | 3.02 | 3/1000 |
| HateMM | Qwen3-VL-2B | 0.017 | [0.82, 1.94] | 0.32 | 195/1000 |
| MHClip-EN | Qwen3-VL-8B | 0.001 | [-0.68, 7.72] | 2.12 | 254/1000 |
| MHClip-EN | Qwen3-VL-2B | -- | [-1.32, 1.96] | 0.63 | 521/1000 |
| MHClip-ZH | Qwen3-VL-8B | 0.104 | [-5.09, 0.55] | 1.49 | 21/1000 |
| MHClip-ZH | Qwen3-VL-2B | -- | [0.42, 0.77] | 0.13 | 863/1000 |

## Starvation diagnostics

How much text the judge had on each benchmark, before any restoration. The 300-character column is the old clipped instrument's visible window, so it bounds how much of each benchmark's transcript that instrument could never see.

| dataset | median dataset-transcript chars | hateful | normal | empty | > 300 chars | restoration coverage |
|---|---|---|---|---|---|---|
| HateMM | 694 | 1144 | 378 | 39 | 467 (0.628) | 0.000 |
| MHClip-EN | 318 | 287 | 322 | 27 | 280 (0.509) | 0.000 |
| MHClip-ZH | 78 | 48 | 92 | 24 | 26 (0.045) | 0.000 |

- **HateMM** — no fresh transcript exists, so the fresh-vs-dataset edit distance and the gate rejection rates are undefined for this run.
- **MHClip-EN** — no fresh transcript exists, so the fresh-vs-dataset edit distance and the gate rejection rates are undefined for this run.
- **MHClip-ZH** — no fresh transcript exists, so the fresh-vs-dataset edit distance and the gate rejection rates are undefined for this run.

## Prior-project reference points

NOT directly comparable to the numbers in this report. Those rows are TEST-split results from the prior project, produced by a different judge (2B/7B/8B depending on the row) under the old clipped-transcript instrument and a generative hateful/not decode, not the raw-z readout used here. This report measures TRAIN_CLEAN with a raw-z threshold. Read them as an order-of-magnitude orientation only.

| dataset | prior method | variant | accuracy / macro-F1 (test split) |
|---|---|---|---|
| HateMM | alarm_backup_7b_20260416 | test_alarm | 0.7953 / 0.7937 |
| HateMM | boundary_rescue | test_v2_alpha0.10_v1_G7 | 0.8465 / 0.8370 |
| HateMM | mars_2b | test_mars | 0.6977 / 0.6964 |
| HateMM | naive_2b | test_naive | 0.6744 / 0.5868 |
| MHClip-EN | alarm_backup_7b_20260416 | test_alarm | 0.6957 / 0.6023 |
| MHClip-EN | boundary_rescue | test_v2_alpha0.15_v1_G6 | 0.7702 / 0.6997 |
| MHClip-EN | mars_2b | test_mars | 0.6584 / 0.6280 |
| MHClip-EN | naive_2b | test_naive | 0.7143 / 0.5289 |
| MHClip-ZH | alarm_backup_7b_20260416 | test_alarm | 0.7114 / 0.4749 |
| MHClip-ZH | boundary_rescue | test_v2_bayes_band_rate_v1_G7 | 0.8255 / 0.8043 |
| MHClip-ZH | mars_2b | test_mars | 0.7450 / 0.7047 |
| MHClip-ZH | naive_2b | test_naive | 0.7450 / 0.5822 |

No published SOTA numbers are quoted: the repo's own baseline briefs carry reproductions rather than reported figures, and no web search was performed.

## Caveats

- Measurement on `train_clean` only. No test split of any dataset was read, so nothing here is a benchmark result and nothing here is comparable to a published test number.
- Labels are used for evaluation alone. The threshold is computed from the score distribution with no labels; the oracle threshold is reported as a gap and never used.
- MHClip is 3-class and `Offensive` maps to 1, so the binary task is Hateful+Offensive vs Normal. That is a coarser positive class than HateMM's, and the prevalences differ across the three benchmarks, so the macro-F1 column is not comparable row to row without that in mind.
- The surface-cue regex family is English. On MHClip-ZH it under-fires and the cue rates in that report are not interpretable.
- **The restoration stage never ran.** Zero restoration coverage on all three benchmarks, for want of source media. The cross-benchmark generalization question the run was launched to answer is not answered by it.

