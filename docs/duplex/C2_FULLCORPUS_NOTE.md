# Full-Corpus C2 Measurement

**Date**: 2026-08-06. **Dataset**: ImpliHateVid `train_clean`, 1283 videos. The test split is not touched.
**Model**: Qwen3-VL-8B-Instruct, bf16, one forward pass per video.

This is method-development measurement, not a kill-test. No pre-registration governs it and none is claimed. The channel-restoration kill-test scored the C2 condition on the 662 dismissed videos only; the flip statistics it reported were conditioned on a stratum that was itself selected by the C0 scores. Extending C2 to all 1283 videos gives the first measurement of the methodized pipeline that is not conditioned on the baseline it is being compared against.

## What the pipeline is

Three stages, one MLLM call per video.

1. The original mp4 audio is re-transcribed with Whisper large-v3.
2. The fresh transcript passes the frozen degeneracy gate and is fed to the judge uncapped.
3. The judge emits a raw unclipped `z = logsumexp(Yes ids) - logsumexp(No ids)`, and the decision threshold is the valley of a kernel density estimate over the 1283 scores. No labels enter any of the three.

## Configuration provenance

Nothing here was chosen for this run. Every component is inherited unmodified from an earlier frozen artifact.

| Component | Source | Status |
|---|---|---|
| Audio and VAD | `scripts/duplex/channel_restoration_audio.py` | unmodified |
| Whisper ASR | `scripts/duplex/channel_restoration_asr.py` | unmodified |
| Degeneracy gate | `scripts/duplex/channel_restoration_gate.py` | unmodified |
| Judge and readout | `src/duplex/extract_duplex_readout.py` | unmodified |
| Threshold recipe | `docs/duplex/reports/rawz_detector_train.json`, estimator `b_kde_valley` | reimplemented, verified |

The ASR configuration is the pre-registered one: `large-v3` through the `transformers` pipeline, fp16 on CUDA with the device asserted at load, language auto-detected, 30-second chunked long-form decoding, and a 30-minute audio cap applied when the wav is extracted. The gate is the pre-registered conjunction: clause-level repetition collapse applied to every fresh transcript, then rejection if and only if the silero-VAD speech fraction is below 0.05 **and** the gzip ratio of the raw fresh text exceeds 7.

The threshold recipe was validated before use. Applied to the C0 scores it returns -2.903500000000001, against the frozen -2.903500000000001 recorded in `rawz_detector_train.json` (match: True).

## Headline numbers

| | C0 baseline | C2 full corpus |
|---|---|---|
| AUC, hateful vs NH | 0.9334 | **0.9505** |
| AUC, IM vs NH | 0.9150 | 0.9298 |
| AUC, EX vs NH | 0.9518 | 0.9711 |
| Label-free macro-F1 | 0.8503 | **0.8796** |
| Label-free threshold | -2.903500000000001 | -4.854375000000001 |
| FN / FP at that threshold | 110 / 82 | 49 / 105 |

The C2 AUC carries a bootstrap 95% interval of [0.938713, 0.960813].

## The label-free threshold

The valley computed on the real full-corpus C2 distribution sits at **-4.8544**, with Scott bandwidth 3.2046 and modes at [-17.745, 12.658]. It lies -1.9509 from the frozen C0 valley of -2.9035.

Trough depth is 0.6004: the valley density is 0.3996 of the lower bracketing mode. Under 1000 bootstrap resamples of the score set the valley has standard deviation 1.1590 and a 95% interval of [-6.581, -2.127], against the C0 valley's own 1.1958 and [-5.335, -0.718]. 0 of the resamples produced no valley at all.

Two further thresholds are reported in the JSON as diagnostics and are not part of the method. Carrying the frozen C0 valley over unchanged gives macro-F1 0.8798. The F1-maximizing threshold chosen against gold labels gives 0.8796 at -5.0, so the label-free choice costs -0.0000 macro-F1 against an oracle that is not available to a label-free method.

## Error anatomy at the self-computed threshold

**False negatives: 49.** 6 explicit and 43 implicit, an implicit share of 0.8776. Within-group recall loss is 0.0185 for EX and 0.1327 for IM. Of the eight videos Step 0 established have no transcribable speech, and for which C2 therefore has nothing to restore, 7 remain false negatives. 3 of the false negatives fell back to the dataset transcript because the gate rejected their fresh pass.

**False positives: 105**, all NH by construction. 0.7905 of them carry a surface cue in the transcript the judge actually read, against 0.5331 among the true negatives (Fisher exact p = 3.02e-07). The cue family is the frozen regex set from the kill-test analysis, used as a text-locating device only and never as an input to any model or threshold.

## Transcription and gate statistics

Across all 1283 videos the gate accepted 1263 fresh transcripts and rejected 20 (rate 0.0156), which fall back to the dataset transcript. 4 clips reached the 30-minute audio cap and 0 raised an ASR error. Median transcript length goes from 952 characters in the dataset to 1008 after the fresh pass and the gate, at a median normalized edit distance of 0.1045 against the dataset text.

Of the 1283 judged videos, 1263 were judged on a gated fresh transcript and 20 on the dataset transcript.

## Determinism

662 videos were scored in both this run and the kill-test C2 run under a byte-identical judge input. 662 of them reproduce their score exactly; the largest absolute difference is 0.0.

## Scope

Every figure is on `train_clean`. The ImpliHateVid test split has not been scored and is reserved. Published supervised numbers on that dataset are test-split results under full supervision, so nothing here licenses a comparison against them. Gold EX/IM/NH prefixes were consumed by the evaluation only: the transcription, the gate, the judge, and the threshold are all label-free.

Full statistics: `docs/duplex/reports/c2_fullcorpus_8b.json`. Raw transcripts, wavs, and per-video scores stay in the gitignored `results/` tree and are not published.
