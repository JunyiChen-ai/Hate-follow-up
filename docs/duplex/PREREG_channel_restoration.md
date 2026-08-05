# Pre-registration — Channel Restoration Kill-Test

**Date frozen**: 2026-08-06, after Step-0, before any bulk download,
transcription, or judge run.
**Date drafted**: 2026-08-06, before Step-0.
**Prior**: `docs/duplex/PREREG_duplex_killtest.md`, `docs/duplex/KILLTEST_POSTMORTEM.md`,
`docs/duplex/PREREG_duplex_readout.md`, `docs/duplex/READOUT_POSTMORTEM.md`.
**Data**: ImpliHateVid `train_clean`, 1283 videos, EX/IM/NH encoded in `Video_ID`.
The test split is not touched.
**Models**: Qwen3-VL-8B-Instruct (primary arm) and Qwen3-VL-2B-Instruct
(secondary arm), bf16, HuggingFace transformers, the frozen `prag` judge from
`src/duplex/score_duplex_probe.py::READER_BLOCKS`.
**Precheck**: every figure quoted below was recomputed from
`results/duplex_readout/ImpliHateVid/scores.jsonl`, its 2B counterpart, and
`annotation(new).json`. Every Step-0 figure was recomputed from that probe's
own per-clip audio, VAD, and transcription records.

## Hypothesis

Two mechanisms are already dead. Instructional reading-separation died in the
kill-test, because a thoroughness-matched placebo reproduced the divergence
statistic to within 0.006 AUC. Perception–report dissociation died in the
readout probe, because a label-free linear direction recovered the verbalized
ranking and never exceeded it, reaching 0.852 against the raw logit baseline of
0.875 on the dismissed stratum. Both deaths share a shape: the proposed
mechanism promised to extract something the model already had, and the model did
not have it hidden anywhere.

This probe therefore stops looking for a better readout and asks a different
question. **The judge may be failing because the evidence never reached it.**

The residual error budget is located and small. At the label-free threshold the
8B judge produces exactly 110 false negatives, all of them inside the frozen
dismissed stratum (`z < −2.944`), split 78 IM and 32 EX, with no interior misses
at all. Five facts about those 110 videos, all recomputed for this document,
point away from judgment failure and toward missing input.

First, they are consistent misses rather than borderline calls. Of the 110, 84
were scored in the earlier five-reader run and 26 were absent from it, every one
lost to the 128-video high-resolution context overflow. Among the 84 present, 82
(97.6%) score `P(Yes) < 0.5` under all five readers, including the two literal
readers and the effort placebo. The reference rates are 39 of 439 (8.9%) for
detected hateful videos and 574 of 632 (90.8%) for NH. These videos read as
normal to every prompt this project has written.

Second, their visible input window looks like NH rather than like hateful
content. Using a fixed regex family of surface cues purely as a text-locating
device, the rate of any cue inside the 300-character window the judge actually
sees is 48.2% (53/110) for the dismissed-hateful set against 81.1% (361/445) for
the asserted-hateful set, Fisher exact `p = 1.2e-11`. The dismissed group's
visible window sits closer to the NH rate of 28.9% than to the asserted rate.

Third, the missing cues are demonstrably present later in the same transcript.
Among truncated videos, the pattern "no cue in the visible window, cue in the
hidden tail" occurs at 34.0% (32/94) for dismissed-hateful against 11.9%
(46/385) for asserted-hateful, a factor of 2.85. The transcripts are long: the
dismissed set has a median of 1177 characters, 85.5% exceed the 300-character
budget, and the median truncated video loses 78.3% of its transcript.

Fourth, hand inspection found no judgment failure. Fifteen dismissed IM videos,
sampled evenly by `z` rank across the 78, were read with frames and full
transcript. Zero cases were visible hate that the judge dismissed. The recorded
patterns were empty or near-empty transcripts, non-English speech, song lyrics,
and cases where the hateful turn arrives after the visible window closes.

Fifth, a subgroup has no usable text at all. Under the repo's frozen degeneracy
rule, 9 of 110 dismissed-hateful videos (8.2%) have a degenerate visible window,
against 5 of 445 asserted-hateful (1.1%) and 3 of 634 NH (0.5%).

The mirror case supplies the control that makes this a claim about input rather
than about the judge. The 82 false positives are cue-*enriched* in exactly the
same window: 52.4% (43/82) carry a surface cue against 25.4% (140/552) among
dismissed NH, Fisher exact `p = 8.4e-7`. A judge that over-flags when cues are
visible and dismisses when they are absent is reading its input faithfully. What
it is not receiving is the evidence.

**Named mechanism — channel starvation.** In the dismissed subpopulation the
hate-bearing evidence is carried by speech, and that channel fails to reach the
judge. Two failure routes are hypothesized: the transcript never captured the
speech (ASR failure), or it captured it beyond the 300-character budget
(truncation). The falsifiable consequence is a causal one. Restore the speech
channel and the dismissed hateful videos should flip; the dismissed NH videos
must not.

## Step-0 outcome (2026-08-06, run before freeze)

Step 0 was a blocking gate on the C2 arm. The `Transcript` field in
`annotation(new).json` is not the ImpliHateVid authors' released text: the
project owner produced it with Whisper large-v3. C2 therefore could not be
premised on beating a weaker third-party ASR system, and the arm was held until
the degenerate cases had been measured directly.

**What was run.** 22 clips were pulled: the union of the two degenerate-subgroup
definitions, 17 videos, plus 5 controls drawn from the remaining
dismissed-hateful set under a fixed seed. The definitions are recorded here
because they differ and the difference matters downstream. *Fully visible* —
`len(Transcript) ≤ 300`, so the whole transcript reached the judge and
truncation cannot be the cause — n = 16. *Degenerate window* — the repo rule
frozen in `analyze_duplex_readout.py`, first 300 characters shorter than 40
characters or less than 50% Latin letters — n = 9. The two sets overlap in 8
videos, giving a union of 17, with 8 fully-visible-only and 1 degenerate-only
(IM_189, a 486-character transcript whose visible window is non-Latin). Each
clip received `ffprobe` stream inspection, a loudness and silero-VAD pass, one
Whisper large-v3 transcription through the `transformers` pipeline in 30-second
chunked mode with language auto-detection, and a per-window language
detection sweep.

**The dataset transcripts are large-v3-grade where the audio is clean.** On four
of the five controls the fresh pass reproduces the dataset transcript almost
exactly: normalized edit distance 0.03 (IM_235), 0.08 (IM_270), 0.09 (IM_359),
0.11 (IM_154), against speech fractions of 0.69 to 0.98. The fifth control,
EX_140, diverges at 0.69 — and its audio is a shouting match with overlapping
speakers, which places it with the hard-audio cases rather than with the clean
ones. Two independent large-v3 passes agree on clean speech and disagree on
hard audio. Whatever C2 recovers, it will not be recovered by using a better
model; it will be recovered where the first pass hit audio it could not handle.

**The 17 split three ways.** Every clip had an audio stream and none reached the
30-minute cap.

- *Recoverable speech*, n = 6 — EX_222, EX_408, EX_494, IM_139, IM_218, IM_86.
  The fresh pass returns substantive content the dataset transcript does not
  contain. Two are flagship cases. IM_86 goes from an empty transcript to 714
  characters of coherent dialogue at a speech fraction of 0.88, and the
  recovered text is a street confrontation carrying explicit hostility toward a
  religious group [verbatim content withheld from this public document; full
  transcripts in the private analysis artifacts on this machine]. EX_408 goes
  from a 3-character dataset transcript to a 1506-character repeated profane
  chant against an unnamed group [verbatim content withheld], at a speech
  fraction of 0.92; the chant is genuinely repetitive in the audio, so its
  length is an artifact of the chant and not of the model. EX_494 goes from 238
  characters of mistranscribed traffic dispute to 2431 characters of sustained
  gendered abuse. The remaining three are
  smaller but directional: EX_222 79 → 204, IM_139 163 → 618, IM_218 153 → 393
  characters.
- *No meaningful speech*, n = 8 — EX_231, EX_439, EX_52, IM_227, IM_276,
  IM_320, IM_328, IM_436. Four have a silero-VAD speech fraction at or below
  3.4% (EX_231, IM_227 and IM_320 at 0.0%, IM_276 at 3.4%), and their fresh
  transcripts are either near-empty politeness tokens (IM_227: a two-sentence
  broadcast-filler phrase, verbatim content withheld) or hallucination loops
  over music. The other four, EX_439, EX_52, IM_328 and IM_436, return garbled
  or degenerate output under both passes: EX_52 collapses into one repeated
  Chinese token at compression ratio 17.3, IM_328 into a single short nonsense
  interjection repeated five times, and IM_436 into 197 characters of
  overlapping shouting no better than the 176 it started with. These videos are
  not audio-starved in the sense C2 addresses, and the pre-registered
  consequence is that C2 cannot help them.
- *Non-English or sung, and fragile*, n = 3 — IM_122 (Russian, all detection
  windows `ru`), IM_189 (windows split `la`/`el`/`en` over liturgical chant),
  IM_367 (English-detected but sung throughout, and the two passes return
  different lyrics at edit distance 0.70). For these the transcript is unstable
  across passes for reasons unrelated to the mechanism under test.

EX_439 is worth recording separately because it occupies a fourth state that
none of the three pre-registered Step-0 outcomes anticipated: it has clear vocal
activity, speech fraction 0.84, yet **both** passes are degenerate — the dataset
transcript is a 22-character abusive fragment and the fresh pass is 844
characters of a single short abusive clause repeated to the end of the clip
[verbatim content withheld from this public document]. Vocal activity is
present, transcribable content is not. EX_439 is assigned to
the no-meaningful-speech group, which is the conservative assignment: it enters
P2 as a predicted non-flip.

**Consequence.** Outcome 1 holds for 6 videos, outcome 2 for 8, outcome 3 for 3.
C2 is retained, because the recoveries are real and one of them is an explicit
targeted confrontation the judge never saw. The gate parameters below and the
P2 subgroup structure are frozen from these measurements.

## Interventions

Three conditions on the 662 dismissed videos, 110 hateful and 552 NH. The NH
half is not a convenience sample: it is the collateral-damage control, and it is
scored under every condition the hateful half is scored under.

**C0, baseline.** The frozen judge input as-is, with the 300-character dataset
transcript. These scores are already on disk and are not recomputed.

**C1, full budget.** The same dataset transcript with no character cap. This
isolates the budget and tail-position factor: the text is unchanged, only its
visibility changes.

**C2, restored audio.** The original mp4 audio re-transcribed with Whisper
large-v3 through the `transformers` pipeline in chunked long-form mode with
automatic language detection, the full new transcript fed with no cap. This
isolates the ASR-failure factor. Whisper settings are frozen here: `large-v3`,
`transformers` ASR pipeline, `language` auto, 30-second chunked long-form
decoding, 30-minute audio cap per clip. Nothing about this configuration is
tuned after the bulk run starts.

**The C2 degeneracy gate.** Step 0 showed that a fresh large-v3 pass fails in a
characteristic way — it emits one clause repeated to the end of the clip, most
severely where there is no speech to transcribe. Feeding such output to the
judge would test Whisper's failure mode rather than the channel-starvation
mechanism, so every fresh transcript passes a two-part gate before it reaches
the judge. Both parts are deterministic and both are fixed now, from Step-0
measurements, not tuned later.

- *Repetition collapse.* Consecutive repeated clauses and n-grams are collapsed
  to a single instance, where a clause boundary is sentence-terminal punctuation
  or a comma. Nothing else about the text changes; the collapse is
  order-preserving and applies to every fresh transcript, rejected or not. A
  sentence-level prototype run during Step 0 already cut EX_439 from 844 to 125
  characters and EX_494 from 2431 to 1971 while leaving the control transcripts
  intact. The frozen rule also collapses comma-chained repeats, which the
  prototype missed and which is the form EX_408's chant takes.
- *Rejection.* If the silero-VAD speech fraction is below 5% **and** the gzip
  compression ratio of the raw fresh transcript — uncompressed bytes over gzip
  bytes — exceeds 7, the fresh transcript is discarded and that video falls back
  to its dataset transcript under C2. Both bounds come from Step 0. EX_231 sits
  at ratio 18.0 with a speech fraction of 0.0%, IM_189 at 10.2 with 0.0%; the
  eight Step-0 transcripts that read as ordinary connected English fall between
  1.5 and 2.2, and no clip in the sample lies between 4.4 and 7.6.

The conjunction is the load-bearing part of the rejection rule. A ratio test
alone would discard EX_408, whose recovered chant compresses at 19.6 because the
audio genuinely repeats a slogan at a speech fraction of 0.92 — the single most
informative recovery in the sample. A VAD test alone would discard IM_367 and
IM_227, whose fresh transcripts are harmless. Requiring both conditions means
the rule fires only when there is no speech *and* the text is pathologically
repetitive, which is the signature of hallucination over music or silence.
Applied to the 22 Step-0 clips the rule discards 2 fresh transcripts, EX_231 and
IM_189, and keeps every recovery.

Everything else is byte-identical to the frozen judge: the `prag` block, 16
frames at the capped processor resolution, greedy decoding, and the raw
unclipped `z = logit(Yes) − logit(No)` readout. No prompt token changes. Whisper
is input-pipeline preprocessing and not an MLLM call, so the method still spends
one MLLM call per video and the two-call cap in `CLAUDE.md` is untouched.

**Flip.** The primary definition is `z > 0` under the intervention, the model's
own decision boundary, for a video that started below `−2.944`. Full `z`
distributions and flip rates at the frozen density valley `−2.9035` are reported
as descriptive context, not as substitute pass criteria.

## Predictions and failure lines

Primary arm, 8B, on the 662 dismissed videos.

| # | Prediction | Pass | Fail |
|---|---|---|---|
| P1 | Flip asymmetry. Restoring the channel moves hateful videos and leaves normal ones alone | under C2, flip-rate(dismissed-hateful) − flip-rate(dismissed-NH) ≥ 0.15, Fisher exact `p < 0.01` | asymmetry < 0.05, or NH flips at a comparable rate → the intervention shifted sensitivity rather than restoring evidence → **kill** |
| P2 | Three-way dissociation over Step-0 subgroups. Each starvation route moves its own videos and no others | the directional pattern below holds across all three cells, with exact flip counts reported per cell per condition | the pattern does not hold directionally → the routes are not distinguished; interpretation weakened, not killed |
| P3 | No collateral damage | C2 flip-rate among dismissed-NH ≤ 0.08, and among a 150-video seed-0 subsample of dismissed NH the median `z` change under C2 lies within ±1.0 | above either bound → the intervention is a global sensitivity shift |
| P4 | The transcription actually changed | descriptive: new-versus-old transcript character length and normalized edit distance, the fraction of dismissed-hateful videos whose Whisper transcript carries surface cues absent from the old visible window, and the count of the 662 fresh transcripts rejected by the C2 degeneracy gate, broken down by label half | no descriptive bar; a null change would make P1 uninterpretable and is reported as such |

**P2, stated in full.** The subgroups are named before the run, and two of the
three come from Step 0 rather than from the outcome being tested.

| Cell | Subgroup | Predicted |
|---|---|---|
| (i) | Truncation route: the 32 truncated dismissed-hateful videos with no surface cue in the visible 300-character window and a cue in the hidden tail | flip under C1 |
| (ii) | ASR route: the 6 Step-0 recoverable-speech videos, EX_222, EX_408, EX_494, IM_139, IM_218, IM_86 | flip under C2 and not under C1 |
| (iii) | Residual: the 8 Step-0 no-meaningful-speech videos, EX_231, EX_439, EX_52, IM_227, IM_276, IM_320, IM_328, IM_436 | flip under neither |

Cell (iii) is what makes P2 a dissociation rather than a pair of one-sided
predictions. If these eight flip under C2, the intervention is adding
suggestibility rather than evidence, since Step 0 established there is no speech
in them to restore. Pass requires the directional pattern across all three
cells; exact flip counts are reported per cell per condition whatever the
outcome. The three fragile-language videos, IM_122, IM_189 and IM_367, are
reported alongside the three cells but excluded from P2's pass/fail decision.
The exclusion is pre-registered here, and its reason is the Whisper instability
documented in Step 0: two large-v3 passes over the same audio return different
content for these clips, so a flip or a non-flip in this group measures decoding
variance rather than channel restoration.

**P3, scope of the transcripts evaluated.** The NH-stability bounds are
evaluated on gated C2 transcripts, that is, after repetition collapse and after
the rejection rule has fallen back to the dataset transcript where it fires.
This is stated so the bound is not read as a claim about raw Whisper output.
Ungated degeneracy was identified as a threat in Step 0 and is handled by design
rather than by the pass criterion: a judge fed 6794 characters of a repeated
clause would be a test of the decoder, not of the channel. The rejected count
enters P4 as a descriptive figure.

Secondary arm, 2B, full 1283 videos, C1 only. The 2B is included because the
length effect that is merely suggestive on the 8B is significant on it: its 79
false negatives have longer transcripts than its 570 detections, AUC 0.608,
`p = 0.0018`, and lose more of them to truncation, AUC 0.609, `p = 0.0017`. The
corresponding 8B contrast does not reach significance, AUC 0.547, `p = 0.128`.
If the budget matters anywhere it should matter there first.

| # | Prediction | Pass | Fail |
|---|---|---|---|
| P5 | Lifting the cap improves the weaker model | AUC(IM vs NH via raw `z`) improves by ≥ 0.02 under C1, **and** the false-negative set below the frozen valley 0.374625 shrinks by ≥ 15% while the false-positive set grows by less than half the false-negative reduction | no AUC change, or false-positive growth swamps false-negative reduction → **kill for the 2B arm** |

## Kill rule

The two arms are independent and both are reported.

- P1 fails → the channel-starvation mechanism is dead for the 8B and the method
  is not to be built on it.
- P5 fails → the mechanism is dead for the 2B.
- P2, P3 and P4 qualify the interpretation of a P1 pass and do not by themselves
  kill the mechanism.

Two consequences follow from the design and are stated in advance so that they
are not mistaken for results. For the 16 fully-visible videos the C1 input is
byte-identical to C0, so their C1 flip rate must be zero; any nonzero value is
decoder nondeterminism and is reported as a determinism check, not as evidence.
And the fully-visible subgroup does not leave P2: Step 0 split it rather than
retiring it, so 6 of its 16 members are P2 cell (ii), 8 are cell (iii), where a
non-flip is the prediction rather than a missing result, and 2 are in the
fragile-language exclusion.

## Confound-control map

| Confound | Control |
|---|---|
| Sensitivity shift: longer input makes the judge say Yes more often, regardless of content | P1 asymmetry against the dismissed-NH half, plus the P3 median-shift bound on the 150-video subsample |
| Whisper hallucination inventing hate cues in silence or music | the C2 degeneracy gate, frozen from Step-0 measurements, plus P3 NH stability, P4 edit-distance and rejection-count reporting, and a manual listen-through of 10 flipped transcripts against their audio (analysis-only spot check) |
| Tail position versus channel: which of the two starvation routes is doing the work | P2 three-way dissociation, C1 against C2, with cell (iii) as the residual that no route should move |
| The existing transcript is already a competent ASR pass, making C2 a null intervention | settled by Step 0: the dataset pass is large-v3-grade on clean audio and fails only where the audio is hard, and 6 of the 17 degenerate cases yield substantive recoveries; P4 is the running check that C2 changed the input at all |
| Label leakage | every intervention and every score is label-free; EX/IM/NH prefixes enter the analysis only; the test split is untouched |
| Prompt-wording sensitivity | one frozen prompt, no reader variation anywhere in this design |
| Readout censoring, the artifact that killed the first probe's P1 | raw unclipped logit difference throughout |

## Label-use statement

Gold EX/IM/NH prefixes and labels are consumed by the analysis alone, as ground
truth for evaluating the predictions and for defining the diagnostic subgroups.
Transcription, judging, and the flip criterion are label-free. The regex cue
family used in the precheck and in P4 is a text-locating device for analysis and
is never an input to any model or threshold. The test split is reserved for the
eventual paper.

## Cost and execution note

The mp4s live on B2 under the original dataset folders. The 662 dismissed clips
are a subset of a 51 GB, 2009-video directory, so the transfer is roughly 12 GiB
on the owner's estimate and at most about 17 GiB under a proportional bound;
the exact figure is measured during the pull and recorded here.

Whisper large-v3 runs over about 662 clips with a heavy-tailed duration
distribution. Per-clip transcription is capped at 30 minutes of audio and any
clip reaching the cap is flagged in the output and excluded from the P4 edit
distance statistics rather than silently truncated. Re-judging costs 662 videos
across two conditions on the 8B, roughly 7 minutes, and 1283 videos on the 2B,
roughly 3 minutes. The machine is the single-GPU RTX 5090 host, so execution is
direct rather than through Slurm.

## Post-run discipline

Results are reported for all pre-registered arms regardless of outcome. Any
deviation from this document, whether a new arm, a changed threshold, an edited
prompt, or a different transcription configuration, is recorded here as a dated
amendment before the deviating run is submitted. The Step-0 outcome recorded
above, together with the gate parameters and the P2 subgroups it fixes, is the
last change made before the freeze.
