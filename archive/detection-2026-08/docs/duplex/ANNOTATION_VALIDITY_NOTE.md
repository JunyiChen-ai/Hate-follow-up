# Blind annotation-validity audit: HateClipSeg normals, MHClip-EN positives

**Date:** 2026-08-08. **Type:** preregistered measurement study, CPU only, no
judge call. **Protocol:** `docs/duplex/PREREG_annotation_validity_audit.md`,
frozen before any coding output existed.

## Verdicts

**Claim 1, HateClipSeg negative-class contamination: CONFIRMED.** Blind coders
placed 16 of the 44 decidable clean-normal videos in the hateful or
extremist-glorifying categories, a rate of 0.364 with an exact 95 percent
interval of 0.224 to 0.522. The frozen floor was 0.20 and the whole interval
sits above it.

**Claim 2, MHClip-EN positive-class construct mismatch: CONFIRMED.** Blind
coders found no protected-group target in 34 of the 49 union positives, a rate
of 0.694 with an exact interval of 0.546 to 0.817. The frozen floor was 0.40 and
the whole interval sits above it.

Both claims were originally derived score-aware, which left them open to the
objection that the analysis relabelled whatever the model got wrong. That
objection does not survive. Coders who never saw a score, a prior code, or the
autopsy reproduced both rates to within three percentage points of the
score-aware estimates.

## What was done

Fresh agent instances coded 164 videos. Each coder received the benchmark's own
written label definitions, one video's automatic transcript, and four frames
sampled uniformly across that video. Coders received nothing else. They did not
see judge scores, prior taxonomy codes, the autopsy note, this study's
motivation, or any file under `results/` other than the packet written for them.
Items carried neutral identifiers, and the identifiers were assigned over the
pooled and shuffled item list under seed 20260808, so an identifier carries no
information about which corpus or which stratum an item came from. Each item was
coded once. There was no adjudication round and no majority vote, which mirrors
the single-call discipline imposed on the judge.

The sample was frozen in advance. On HateClipSeg it is all 50 clean-normal
videos, meaning the videos in the scored split whose shipped video-level label
list contains none of the five offensive categories, plus 25 randomly sampled
shipped positives as blinding fillers. On MHClip-EN it is all 49 union
positives, meaning the Hateful and Offensive videos in the scored test split,
plus 40 randomly sampled shipped Normals as fillers. Fillers were coded
identically and are reported below, and no frozen claim depends on them.

The orchestrating agent opened no score file until every coding output was on
disk. The blinding held end to end.

## Claim 1 in detail

The HateClipSeg coding scheme allowed several codes per video. Rates below are
over the 44 clean normals that received a decidable code; 6 of the 50 were coded
"cannot determine" and leave both the numerator and the denominator.

| Blind code on the clean-normal stratum | k | n | Rate | Exact 95% CI |
|---|---:|---:|---:|---|
| Hateful toward a protected group | 9 | 44 | 0.205 | 0.098 to 0.353 |
| Extremist or hate-movement glorification | 12 | 44 | 0.273 | 0.150 to 0.428 |
| **Either of the two, the frozen claim** | **16** | **44** | **0.364** | **0.224 to 0.522** |
| Any offensive code, including other-offensive | 21 | 44 | 0.477 | 0.325 to 0.633 |

Five videos carry both hate codes, which is why the two rows above the claim sum
past it. The verdict does not depend on the exclusion rule for undecidable
items. Counting all 6 undecidable videos as clean gives 16 of 50, a rate of
0.320, still above the floor.

The score-aware autopsy put this figure at 17 of 50, or 0.340. The blind figure
is 0.364. The two estimates agree far more closely than the width of either
interval.

## Claim 2 in detail

No MHClip-EN item was coded "cannot determine", so all 49 union positives count.

| Blind code on the union-positive stratum | k | n | Rate | Exact 95% CI |
|---|---:|---:|---:|---|
| **No protected group targeted, the frozen claim** | **34** | **49** | **0.694** | **0.546 to 0.817** |
| Coded Hateful or Offensive, agreeing with the shipped positive | 41 | 49 | 0.837 | 0.703 to 0.927 |
| Coded Hateful | 13 | 49 | 0.265 | 0.149 to 0.411 |
| Coded Normal, disagreeing with the shipped positive | 8 | 49 | 0.163 | 0.073 to 0.297 |

The second row matters for interpretation. Coders agreed with the shipped
positive label on 84 percent of the positive class. The problem the claim
identifies is not that these videos are benign. The problem is that most of them
are offensive on an axis the benchmark's own Hateful definition excludes, namely
profanity, sexual content, and cruelty aimed at named individuals rather than at
groups defined by a protected attribute.

## Filler results

Fillers test whether the coders are simply generous. They are not.

| Filler stratum | k | n | Rate | Exact 95% CI |
|---|---:|---:|---:|---|
| HateClipSeg shipped positives coded as carrying any offensive content | 20 | 25 | 0.800 | 0.593 to 0.932 |
| HateClipSeg shipped positives coded hateful or extremist | 15 | 25 | 0.600 | 0.387 to 0.789 |
| MHClip-EN shipped Normals coded Hateful or Offensive | 8 | 40 | 0.200 | 0.091 to 0.356 |

No filler item was coded "cannot determine". Coders endorsed 80 percent of
HateClipSeg's shipped positives and only 20 percent of MHClip-EN's shipped
Normals, so they are not applying a blanket positive reading. The 20 percent
figure on MHClip-EN Normals is a genuine finding in its own right and is
consistent with the autopsy's false-positive taxonomy on that corpus, but no
frozen claim rests on it.

## Cannot-determine counts

Six items in total, all six in the HateClipSeg clean-normal stratum, and none in
any other stratum of either corpus. Coders described the same failure mode in
every case: four near-identical static frames combined with an automatic
transcript that had degenerated into recognition noise, leaving no interpretable
evidence in either channel. This concentration is itself informative. The
undecidable items cluster in exactly the stratum the audit was testing, and they
are undecidable because of evidence delivery rather than because the content is
ambiguous.

## Secondary: judge ranking against blind-coded labels

These numbers were computed only after all coding was collected. They use the
frozen single-call Qwen3-VL-8B raw scores already committed for both corpora.
The first table is restricted to coded items with a decidable code.

| Corpus and boundary | n | AUC against shipped labels | AUC against blind codes |
|---|---:|---:|---:|
| HateClipSeg, offensive union | 69 | 0.697 | 0.892 |
| HateClipSeg, hateful strict | 69 | 0.787 | 0.919 |
| MHClip-EN, union | 89 | 0.814 | 0.893 |
| MHClip-EN, hateful strict | 89 | 0.621 | 0.807 |
| MHClip-EN, protected-group-target axis | 89 | 0.814 | 0.791 |

The judge ranks the blind codes better than it ranks the shipped labels on every
boundary except the last, where the comparison is against a different target
variable rather than a corrected version of the same one.

The second table redoes the autopsy's corrected-corpus arithmetic on the full
scored split, substituting the blind codes for the score-aware ones. Positives
and negatives that were never coded keep their shipped labels.

| Corrected corpus figure | n | AUC | Delta |
|---|---:|---:|---:|
| HateClipSeg, offensive union, as shipped | 394 | 0.754 | — |
| HateClipSeg, union, 16 blind-flagged negatives removed | 378 | 0.895 | +0.141 |
| HateClipSeg, union, those 16 and the 6 undecidable removed | 372 | 0.875 | +0.121 |
| MHClip-EN, union, as shipped | 161 | 0.785 | — |
| MHClip-EN, positives restricted to the blind protected-target set | 127 | 0.866 | +0.082 |

The autopsy's headline correction of +0.147 on HateClipSeg reproduces at +0.141
under blind codes, and at +0.121 under the conservative variant that also
discards every undecidable video. The arithmetic survives. It is not withdrawn.

## Duplicate-transcript audit

This part of the study involves no model of any kind. Two videos whose audio
transcribes to the identical byte string are either the same upload twice or the
same speech recut, so a shipped label that differs between them is an annotation
inconsistency that neither the judge nor the coders can be blamed for.

| Quantity | Value |
|---|---:|
| Scored HateClipSeg videos | 394 |
| Videos with an empty transcript | 0 |
| Exact-duplicate transcript groups | 8 |
| Videos inside those groups | 16 |
| Groups disagreeing on the offensive-union binary label | 1 |
| Groups disagreeing on the hateful-strict binary label | 2 |
| Groups disagreeing on the exact multilabel set | 4 |
| Groups containing a clean-normal stratum video | 2 |
| Groups with both members blind-coded, and agreeing | 1 of 1 |

Every group is a pair. Half of the eight pairs carry different shipped label
sets, and one pair is split across the binary offensive-union boundary itself,
meaning one copy is a shipped positive and the other is a shipped negative. That
single pair is enough to establish that at least one clean-normal video is a
labelling mistake rather than a policy choice, and it is established without
reference to any model output.

One correction to the autopsy is due here. The autopsy wrote that two of the
eight duplicate groups disagree about the binary collapse. Under the primary
offensive-union collapse the correct count is one. Two groups disagree under the
hateful-strict collapse. The autopsy's substantive point stands, but the number
attached to it was for the secondary collapse rather than the primary one.

MHClip-EN carries two exact-duplicate transcript groups covering five videos,
but both are degenerate recognition outputs of three and twenty characters, so
they carry no label-consistency information. Separately, two coders in different
batches flagged MHClip-EN items whose narration does not match the picture. A
check confirmed the two transcripts are distinct from each other, so this is a
property of the source clips rather than a misalignment in the transcription
pipeline.

## Interpretation

The reframing the autopsy proposed is licensed, on both corpora, by evidence
that does not depend on the judge's scores.

HateClipSeg's offensive-union number is not a measurement of the model. A third
of its negative class is content that the benchmark's own written definitions
place in an offensive category, and the model is right to score it high. Future
comparisons on this corpus must report the hateful-strict collapse, must state
that even the strict collapse leaves 164 offensive-only videos on the negative
side, and must carry the label-corrected range alongside the shipped-label
number.

MHClip-EN's positive class is not a hate class. It is a discomfort class. Two
thirds of it carries no protected-group target under the corpus's own
definition of Hateful, which requires exactly such a target. Coders agreed that
these videos are offensive; they disagreed that the offence is group-directed.
Any evaluation on this corpus is therefore measuring agreement with a
platform-harm boundary, not with a hate boundary, and the follow-up must say so
rather than reporting the union figure as a hate-detection score.

Neither result adds a method component, and neither is a licence to tune against
corrected labels. The correction is a reporting obligation, not a new
supervision signal. Nothing in this study changes the two open problems the
autopsy ended on, which remain the hateful-versus-offensive-only boundary on
HateClipSeg and implicit protected-group hostility on MHClip-EN.

The known residual weakness was stated in advance and still holds. The coders
are language models, not trained human annotators. They come from a different
model family than the judge under study, and they were blind to its scores, but
they share general pretraining biases with it. The duplicate-transcript audit is
the one piece of evidence here that is free of that objection, and it is
consistent with the coded result. A human-subjects audit remains out of scope.

## Decision

Both claims are hardened rather than retracted. The closeout note needs no
amendment on either corpus. The evaluation protocol for the follow-up gains one
requirement: every reported number on HateClipSeg and MHClip-EN must appear
alongside its label-corrected counterpart, with the correction sourced from this
blind coding rather than from the score-aware autopsy.

## Reproduction

```bash
python3 scripts/duplex/annotation_validity_manifest.py
python3 scripts/duplex/annotation_validity_duplicates.py
python3 scripts/duplex/annotation_validity_analyze.py
```

The first script builds the blinded packets and the identifier mapping. The
second is the model-independent duplicate audit. The third is the only step that
opens a score file, and it refuses to run until every item has a code. The
manifest, the raw coder outputs, the coder-facing packets and the machine-readable
summary are under `results/annotation_validity/`, which is gitignored because
those files carry video identifiers and transcript text. `lexicons.json` was
never opened. No model call was made against any GPU at any point.
