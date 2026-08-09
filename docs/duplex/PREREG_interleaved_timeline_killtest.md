# Pre-registration — Interleaved-timeline kill test (speech–visual binding)

**Frozen:** 2026-08-09, before any judge call under either new arm and before
any timestamped ASR chunk was written to disk. **Compute:** one RTX 5090,
direct execution. Whisper large-v3 re-transcription for chunk timestamps, then
one Qwen3-VL-8B-Instruct forward pass per video per new arm. **Status:** kill
test on the input-format family. Nothing in the judge, the prompt rules, the
reader block, or the readout moves; only the arrangement of evidence already
present in the baseline call changes.

## Phenomenon

Hate in video is often carried by the *binding* of what is said to what is
shown at that moment. A skit delivers a slur while the camera holds on the
mocked group; a meme cut pairs benign narration with a targeted image; fiction
frames an assertion as a quotation only because of what is on screen while it
is spoken. The offending content is the pair, not either member.

The frozen judge cannot see pairs. Its input is sixteen frames delivered as one
undifferentiated block, followed by the whole transcript as one blob. Every
timestamp is destroyed at the input boundary: frame 3 and the words spoken over
frame 3 are separated by fifteen images and an arbitrary amount of text. If the
model must reconstruct co-occurrence to judge these videos, the current format
gives it nothing to reconstruct from.

## Claim under test

Restructuring the *same* evidence as a time-interleaved sequence — frame 1, the
words spoken while frame 1 was on screen, frame 2, the words spoken then, and
so on, with the title last — restores the binding inside a single call, with
zero new information added. If the claim is true, videos whose hate lives in
the pairing should rank better; videos whose hate is carried by monologue
should not move.

If the claim is false, the format change either does nothing or moves
everything indiscriminately, and a control that keeps the format but destroys
the pairing should capture the whole effect.

## Corpora

| Corpus | n | Positives / negatives | Role |
|---|---|---|---|
| MHClip-ZH test_clean | 149 | 45 / 104 | binding-heavy stratum host |
| MHClip-EN test_clean | 161 | 49 / 112 | descriptive |
| HateClipSeg test_clean | 394 | 344 / 50 | binding-heavy stratum host |
| ImpliHateVid test_clean | 400 | 199 / 201 | negative control (monologue-carried, binding-irrelevant) |

Label mapping is the frozen one: MHClip `Hateful` and `Offensive` both map to
1, `Normal` to 0. HateClipSeg binary is the shipped `Offensive` vs `Normal`.
ImpliHateVid is already binary. One ImpliHateVid id has no usable frames and
was absent from the baseline scores; the corpus is 400, not 401, in all arms.

## Arms

Three arms per corpus. The judge (Qwen3-VL-8B-Instruct, bf16), the system
message, the platform rules block, the `prag` reader block, the question, the
Yes/No token-id sets, the logit-difference readout, the per-frame pixel budget
(min 65536, max 100352), and the sixteen-frame count are identical across all
three, and identical to the baseline runs already on disk.

**BASELINE.** The scores already on disk, unchanged and not recomputed:
`results/testruns/{mhclip_zh,mhclip_en,implihatevid}/judge_8b/scores.jsonl`
and `results/hateclipseg/judge_8b/scores.jsonl`. Content order is sixteen
images, then one text block carrying title, whole transcript, rules, reader
block, question.

**INTERLEAVED.** Content order is frame 1, the transcript segment assigned to
frame 1, frame 2, its segment, and so on through frame 16, then the same text
block with the title, rules, reader block and question. The transcript content
is byte-identical in total to the baseline's; only its position changes. The
`Transcript:` field of the prompt carries the fixed pointer string
`(given above, interleaved with the frames)` in place of the blob.

**MISALIGNED.** Byte-identical to INTERLEAVED in every respect except that
segment *j* is placed after frame *(j + 8) mod 16* instead of frame *j*. Same
format, same number of content items, same total text, wrong binding. Rotation
by 8 of 16 is the maximal displacement on a cyclic sixteen-slot timeline.

No numeric timestamps appear in any prompt. Alignment is carried purely by
adjacency, so INTERLEAVED and MISALIGNED differ in nothing a token counter
could detect, and no metadata absent from the baseline enters the input.

Empty segments emit no text item. Rotation permutes segments, so the count of
non-empty text items is identical between INTERLEAVED and MISALIGNED for every
video.

## Segmentation

Frame *i* of the sixteen (0-indexed) is taken to cover
`[i·D/16, (i+1)·D/16)` where *D* is the container duration from
`audio_meta.jsonl`; `frames_16` was extracted evenly over that span.

The transcript the judge sees is the frozen one: the C2-gated fresh Whisper
transcript where the gate accepted it, the dataset transcript otherwise. It is
split into clause units by the frozen `split_units` routine of
`scripts/duplex/channel_restoration_gate.py`, and each unit is assigned to the
frame slot containing its own midpoint in time. Three routes to that time, in
priority order:

1. **Timestamped route.** Whisper large-v3 is re-run on the same wav files with
   the frozen ASR configuration, storing the chunk timestamps the original run
   computed and discarded. A character position maps to a time by linear
   interpolation inside its chunk. This route is used only when the re-run
   reproduces the stored `fresh_text` exactly and the video's transcript came
   from the accepted fresh route.
2. **Proportional route (documented fallback).** Where route 1 is unavailable —
   the gate fell back to the dataset transcript, the video had no usable audio,
   or the re-run did not reproduce the stored text byte-for-byte — a unit at
   character midpoint *p* of a transcript of length *L* is assigned to slot
   `floor(16·p/L)`. Per-corpus counts of videos on this route are reported.
3. **Empty.** A video with no transcript text contributes no segments; its
   INTERLEAVED and MISALIGNED inputs are the frames plus the trailing block.

The unit-to-slot map is monotone in character position under both routes, so
reading the segments in frame order recovers the transcript in its original
order under INTERLEAVED.

Repetition collapse is handled by tracking which clause units survive the
frozen `collapse_repeats`; surviving units keep their own character offsets in
the pre-collapse text, so the timestamped route stays valid for videos whose
transcript was collapsed.

## Frozen baselines

Recomputed from the scores already on disk before this file was written.

| Measurement | AUC |
|---|---|
| MHClip-ZH full corpus | 0.8547 |
| MHClip-EN full corpus | 0.7847 |
| HateClipSeg full corpus | 0.7538 |
| ImpliHateVid full corpus | 0.9473 |
| MHClip-ZH binding stratum, codes G ∪ F ∪ I (13) vs shipped Normals (104) | 0.7260 |
| HateClipSeg insulting-only (59) vs clean normals (50) | 0.6373 |

The MHClip-ZH stratum is the false-negative side of the ranking autopsy coded
G (group stereotype), F (fiction or skit framing) or I (irony), read from
`results/ranking_autopsy/zh/coding_zh.tsv` with the alias-to-id map in
`results/ranking_autopsy/zh/packet.json`. These are the codes whose failure
description is a binding failure. The HateClipSeg stratum and its negatives are
the frozen pair already used by the readout-bottleneck kill test, from the
shipped video-level multi-labels.

## Frozen decision rule

The design **SURVIVES** only if all three clauses hold.

1. **Binding-heavy strata move.** On at least one of the two strata
   (MHClip-ZH G ∪ F ∪ I, HateClipSeg insulting-only), INTERLEAVED improves
   stratum AUC over BASELINE by at least +0.04, and on the other stratum
   INTERLEAVED does not degrade AUC by more than 0.01.
2. **The negative control does not move.** ImpliHateVid full-corpus AUC changes
   by less than 0.01 in absolute value between BASELINE and INTERLEAVED.
   Movement there means the format change is doing something generic, not
   restoring binding.
3. **The control does not explain it.** On the stratum that carried clause 1,
   MISALIGNED achieves no more than half of INTERLEAVED's gain over BASELINE.
   More than half means the effect is the format, not the pairing.

Any clause failing kills the design. No clause may be renegotiated after a
number is seen.

## Reported regardless of verdict

Full-corpus AUC for all three arms on all four corpora; both stratum AUCs for
all three arms; per-video sign flips between arms; the count of videos on the
proportional segmentation route per corpus; the count of videos whose re-run
ASR text did not reproduce the stored one.

## Falsification

The story dies if any of these is observed. The strata do not move under
INTERLEAVED (the binding was never the obstacle). ImpliHateVid moves as much as
the strata (the format change is a generic perturbation). MISALIGNED matches
INTERLEAVED (the model responds to interleaved *format*, not to *which* words
sit beside *which* frame — meaning it is not reading the pairing at all).

## Integrity constraints

- No raw video id and no transcript text appears in any committed file.
- The baseline is read, never recomputed; a replication check on 20 videos per
  corpus asserts that the new scorer reproduces the on-disk baseline `z`
  exactly when handed the baseline arrangement, before either new arm runs.
- Coverage is asserted at full corpus size per arm; a short arm is a failed
  run, not a result.
