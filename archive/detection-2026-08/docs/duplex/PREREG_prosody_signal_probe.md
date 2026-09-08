# Pre-registration — Non-lexical delivery/prosody signal

**Frozen:** 2026-08-08 before loading or scoring the emotion encoder.  
**Cohort:** existing 90-video ImpliHateVid attribution cohort: 30 corrected-8B
FP, 30 cue-matched TP, 30 cue-carrying TN.  
**Purpose:** gate an information-adding paralinguistic restoration method.

## Mechanism

Whisper restores lexical content but discards delivery. The same hostile words
can be neutrally reported or aggressively advanced. A general-purpose emotion
encoder supplies non-lexical acoustic evidence absent from both transcript and
frames. If aggressive/angry delivery distinguishes cue-sharing TP from FP, the
method will attach timestamped delivery tags to ASR turns before one unchanged
holistic moderation call.

## Frozen feature

- Encoder: `superb/wav2vec2-base-superb-er`, frozen emotion recognition.
- Audio: existing 16-kHz mono WAV.
- Intervals: valid Whisper timestamped chunks, split to at most 10 seconds;
  clips shorter than 0.5 seconds are skipped.
- Per interval: model posterior for its `ang`/`angry` label; if label names are
  unavailable, abort rather than choose a class post hoc.
- Per video primary feature: speech-duration-weighted mean angry posterior.
- Secondary diagnostics: maximum and top-four mean; they cannot replace the
  primary feature.

No transcript words, identity terms, hate resources, or dataset labels enter
the encoder.

## Frozen signal rule

The channel **PASSES** only if all hold:

1. coverage at least 75/90 videos;
2. AUC(TP over FP) by mean angry posterior at least 0.65;
3. within the marker-positive half, AUC(TP over FP) at least 0.65;
4. AUC(TP over TN) at least 0.60;
5. after regressing the prosody score on raw z without labels, residual
   prosody still separates TP from FP at AUC at least 0.60;
6. a duration-only feature fails to reach 0.60 TP-vs-FP AUC.

**Pass:** generate timestamped `[DELIVERY=...]` tags plus shuffled-tag and
constant-tag controls, then run the one-call judge probe.  
**Fail:** emotion/prosody does not expose the missing specificity signal; close
the channel and return to method discovery with the measured error budget,
prioritizing a new information source rather than another threshold variant.

