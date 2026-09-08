# Interleaved-timeline kill test — DEAD

**Protocol:** `docs/duplex/PREREG_interleaved_timeline_killtest.md`, frozen
2026-08-09 before any judge call. **Ran:** 2026-08-09/10, one RTX 5090, direct
execution. **Compute:** 45.0 audio-hours of Whisper large-v3 re-transcription to
recover chunk timestamps (1.99 h wall) plus 2,208 Qwen3-VL-8B forward passes
across two new arms (7.9 min wall at 0.30–0.36 s/video). **Verdict:** DEAD.
Clause 1 fails, clause 2 passes, clause 3 fails — and clause 3 fails in the
direction that settles the question rather than leaving it open.

## What was tested

The hypothesis was that the frozen judge misses hate carried by the binding of
speech to the co-occurring visual because the input format destroys temporal
alignment: sixteen frames arrive as one block and the transcript as one blob, so
frame 3 and the words spoken over frame 3 are separated by fifteen images and an
arbitrary run of text. Restructuring the identical evidence as a time-interleaved
sequence — frame 1, the words spoken then, frame 2, the words spoken then, title
last — should restore the binding in a single call at zero information cost.

Three arms, identical in judge, dtype, pixel budget, frame count, system message,
platform rules, `prag` reader block, question, token-id sets and readout.
BASELINE is the frozen scores already on disk. INTERLEAVED pairs frame *i* with
segment *i*. MISALIGNED pairs frame *i* with segment *(i + 8) mod 16*: the same
format, the same text, the same number of content items, the wrong pairing.

## Verdicts

| Clause | Rule | Observed | Verdict |
|---|---|---|---|
| 1 — binding strata move | one stratum ≥ +0.04, the other ≥ −0.01 | MHClip-ZH G∪F∪I **+0.033**; HateClipSeg insulting-only **−0.001** | **FAIL** |
| 2 — control does not move | ImpliHateVid full-corpus \|Δ\| < 0.01 | **−0.0085** | PASS |
| 3 — control does not explain it | MISALIGNED gain ≤ 0.5 × INTERLEAVED gain | **+0.038 vs +0.033**, a ratio of **1.16** | **FAIL** |

Clause 3 is the one that matters. The rotated pairing did not capture half the
gain; it captured all of it and a little more. Displacing every transcript
segment by half the timeline — putting the words spoken at the end of the video
next to the opening frame — left the ranking marginally better than the correct
pairing did. Whatever the interleaved format buys, it is not the pairing.

## Full-corpus ROC-AUC, all arms, all corpora

Coverage is complete in every cell: 149, 161, 400 and 394 videos, three arms
each, no video dropped.

| Corpus | n (pos/neg) | BASELINE | INTERLEAVED | Δ | MISALIGNED | Δ |
|---|---|---|---|---|---|---|
| MHClip-ZH | 149 (45/104) | 0.8547 | 0.8823 | +0.0276 | 0.8819 | +0.0272 |
| MHClip-EN | 161 (49/112) | 0.7847 | 0.7861 | +0.0014 | 0.7843 | −0.0004 |
| HateClipSeg | 394 (344/50) | 0.7538 | 0.7558 | +0.0020 | 0.7544 | +0.0006 |
| ImpliHateVid | 400 (199/201) | 0.9473 | 0.9387 | −0.0085 | 0.9373 | −0.0100 |

MHClip-ZH is the only corpus that moves at all, and it moves the same amount
under both arrangements: +0.0276 correct, +0.0272 rotated. The two differ by
0.0004, which is four ten-thousandths of an AUC point on 4,680 pairs.

## Stratum ROC-AUC

| Stratum | n (pos/neg) | BASELINE | INTERLEAVED | Δ [95% CI] | MISALIGNED | Δ [95% CI] |
|---|---|---|---|---|---|---|
| MHClip-ZH binding codes G∪F∪I | 13/104 | 0.7260 | 0.7589 | +0.0329 [−0.012, +0.085] | 0.7641 | +0.0381 [−0.010, +0.092] |
| HateClipSeg insulting-only | 59/50 | 0.6373 | 0.6366 | −0.0007 [−0.040, +0.039] | 0.6346 | −0.0027 [−0.047, +0.039] |

Both interleaved intervals straddle zero. The MHClip-ZH stratum holds 13
positives, so even the point estimate that fell short of the +0.04 bar is worth
roughly one rank position; it is not a measurement the design could have been
saved by.

## Why clause 3 is decisive: the two new arms agree with each other

If the pairing carried the effect, the correct and rotated arrangements would be
far apart, and each would sit closer to the baseline than to the other. The
opposite holds on every corpus.

| Corpus | rank corr. INTERLEAVED vs MISALIGNED | mean \|Δz\| between the two arms | mean \|Δz\| vs BASELINE | decision disagreements between arms |
|---|---|---|---|---|
| MHClip-ZH | 0.981 | 1.17 | 1.59 | 3 / 149 |
| MHClip-EN | 0.966 | 1.49 | 1.90 | 7 / 161 |
| HateClipSeg | 0.958 | 1.24 | 1.63 | 8 / 394 |
| ImpliHateVid | 0.981 | 1.65 | 1.80 | 20 / 400 |

Rotating every segment by half the video moves the judge's score less than
moving the transcript out of the blob does, on all four corpora. The judge
registers that the transcript now arrives in pieces between images. It does not
register which piece sits beside which image.

## Decision flips

Sign of *z* (Yes above zero, No below), against BASELINE.

| Corpus | arm | changed | → Yes | → No | flips toward the label | flips away |
|---|---|---|---|---|---|---|
| MHClip-ZH | INTERLEAVED | 10 | 9 | 1 | 3 | 7 |
| MHClip-ZH | MISALIGNED | 9 | 7 | 2 | 3 | 6 |
| MHClip-EN | INTERLEAVED | 14 | 5 | 9 | 6 | 8 |
| MHClip-EN | MISALIGNED | 15 | 7 | 8 | 7 | 8 |
| HateClipSeg | INTERLEAVED | 12 | 10 | 2 | 9 | 3 |
| HateClipSeg | MISALIGNED | 16 | 14 | 2 | 11 | 5 |
| ImpliHateVid | INTERLEAVED | 17 | 9 | 8 | 5 | 12 |
| ImpliHateVid | MISALIGNED | 23 | 12 | 11 | 6 | 17 |

Flip counts are small (2.5–5.8% of each corpus) and net-negative on three of the
four corpora. MHClip-ZH gains AUC while losing decisions, which is the signature
of a ranking shift rather than a comprehension gain: interleaving pushes borderline
positives up without moving them past the threshold, and pushes a few normals
over it.

## Evidence quality

The test is not weak on inputs, which is what makes the negative worth
recording.

- **Segmentation used real timestamps almost everywhere.** 1,067 of 1,105 videos
  took the timestamped route. 14 took the documented proportional fallback, all
  of them because the C2 gate had already fallen back to the dataset transcript
  and no ASR timeline exists for it. 24 have no speech at all.

  | Corpus | timestamped | proportional | no speech | mean non-empty slots |
  |---|---|---|---|---|
  | MHClip-ZH | 147 | 2 | 0 | 7.3 |
  | MHClip-EN | 155 | 4 | 2 | 12.6 |
  | ImpliHateVid | 393 | 8 | 0 | 14.4 |
  | HateClipSeg | 372 | 0 | 22 | 13.8 |

- **The re-transcription reproduced the frozen text exactly.** 1,105 of 1,105
  videos: Whisper large-v3 under the frozen configuration returned the stored
  `fresh_text` byte-for-byte, so the chunk timestamps apply to the very
  characters the judge reads. Zero videos were disqualified for text drift.
- **The transcript is unchanged.** The concatenation of the sixteen slots
  rebuilds the judged transcript character for character; this is asserted per
  video and no video failed it.
- **The interleaving is dense.** Between 7 and 14 of the sixteen slots carry
  text on average, so the arrangement is a genuine timeline, not a transcript
  parked beside one frame.
- **The scorer is the frozen judge.** 80 of 80 replication videos (20 per
  corpus) reproduced the on-disk baseline *z* bit-exactly under the baseline
  arrangement. Any difference in the new arms is the arrangement and nothing
  else.

## What this rules out

The premise, not just the implementation. The claim was that co-occurrence
information is present in the evidence and destroyed by the input format; supply
it and the judge will use it. The evidence was supplied at genuine ASR-timestamp
resolution, and a control that supplies the same evidence with the pairing
deliberately falsified performs identically. A model reading the pairing could
not produce that result.

Two readings survive and both close the direction. Either the 8B judge does not
bind text to adjacent images across a long interleaved context at all — in which
case no rearrangement of the same evidence will help, and the fix would have to
be a different model, not a different format. Or it does bind, but the binding
contributes nothing to this judgment, because the judgment is being made from
the transcript and the title with the frames as weak context — which is what the
five-corpus error attribution already concluded from the other direction.

The one real effect the test found is not about hate video. Moving the
transcript out of one blob and into pieces between images is worth about +0.027
AUC on MHClip-ZH and nothing anywhere else, with the same value whether the
pieces are placed correctly or scrambled. That is a formatting sensitivity of a
Chinese-language corpus with short transcripts, and by this project's standards
it is an engineering artifact with no story: it has no mechanism that references
hateful video, and its own control reproduces it.

## Deviations from the pre-registration

One, amended and committed before any judge call, with no score in existence at
the time. Segmentation granularity was raised from the clause to the atomic
piece — clause, Whisper chunk, or whitespace-delimited token, whichever cuts
finer. The clause is the wrong unit here for a reason unrelated to the
hypothesis: Whisper emits Mandarin without punctuation, so an entire MHClip-ZH
transcript is a single clause, and clause-granular interleaving would have
placed the whole transcript beside one frame across most of that corpus, testing
nothing. The finer cut changes no character of the transcript and applies
identically to both new arms.

Everything else ran as frozen. The clause thresholds were not touched after any
number was seen.

## Artifacts

- `results/interleaved_timeline/results.json` — machine-readable, all arms, all
  corpora, both strata, flips, arm agreement, clause verdicts.
- `results/interleaved_timeline/segmentation_census.json` — route census.
- `scripts/duplex/interleaved_timeline_{asr,build,score,analyze}.py`,
  `scripts/duplex/run_interleaved_{asr,timeline}.sh`.
- Scores and segmentations stay under `results/`, which is not committed; no
  video id and no transcript text appears in any tracked file.
