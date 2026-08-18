# Frame-level evaluation protocol (corpus-general)

**Frozen:** 2026-08-18, Phase 1 of the baseline reproduction plan, before
any baseline is trained or scored. This document supersedes nothing: it
generalizes the HateMM-only protocol frozen in
`PREREG_frame_level_evaluation_hatemm.md` to every corpus in the
reproduction study, and adds the per-corpus gold rules that HateMM did
not need. The HateMM numbers produced under the earlier prereg are
reproduced bit for bit under this protocol (see "Regression" below), so
the generalization costs no comparability.

## Why this document exists

`LOCALIZATION_PROTOCOL_SURVEY.md` records the state of the field across
all six hateful-video temporal localization works. Not one of them
publishes a frame-level ground-truth array, and not one states the rule
that converts an annotated span into frame labels. The two works that
report frame-level ROC-AUC and AP, LELA (2602.09637) and MultiHateLoc
(2512.10408), omit both the frame grid and the conversion rule, so their
numbers are neither reproducible nor comparable to each other.

Every frame-level number in this study is therefore produced under the
protocol below, and the ground-truth arrays it defines are released
alongside. Protocol plus arrays are the artifact the field visibly
lacks, and they are what makes the comparison table in this study
checkable by a third party rather than merely asserted.

## The grid

One frame per second. Frame timestamps are the integers
`t = 0, 1, 2, ...` taken while `t < duration`, so a 30.0 second video
yields frames 0 through 29 and a 101.22 second video yields frames 0
through 101. Duration is the wav duration of the extracted 16 kHz mono
audio, read from the corpus's timestamped-chunk manifest. The container
duration reported by the video file is not used. The choice is not
cosmetic: the two disagree by up to 1.13 s on HateMM and 3.22 s on
MultiHateClip ZH, which is several frames, so the source is frozen here
rather than left to each method. The wav is authoritative because it is
the signal every audio and transcript method actually consumed.

The rate is 1 fps because that is the resolution at which the upstream
annotations are given: HateMM's spans are whole seconds parsed from
`HH:MM:SS` strings, and MultiHateClip's `Duration` column holds integer
second pairs. A finer grid would manufacture precision the gold does not
carry; a coarser grid would discard spans as short as one second, of
which HateMM has several.

## Span to frame conversion

A frame is positive if and only if its timestamp lies inside some hate
span under half-open containment, `start <= t < end`. Overlapping spans
union. A span reaching past the end of the audio is truncated by the
grid rather than by the span list, so it contributes only the frames
that exist. Spans with `end <= start` are degenerate: they are dropped
before conversion, and the drop is counted in the sidecar rather than
repaired by guessing an intended end.

Half-open containment is the choice that makes adjacent spans tile
without double-counting the boundary second, and it matches the
convention already frozen for the chunk spans on the method side.

## What the gold contains, and what it does not

The gold arrays carry frame labels only. Nothing about how a method
assigns a score to a frame belongs in them. In particular, the
**uncovered-frame floor rule is a method-side score assignment, not part
of the gold**: a transcript-chunk method that scores Whisper segments
has no evidence on frames no segment covers (silence, music, a dropped
chunk), and the pre-registered honest assignment is
`(corpus-wide minimum chunk score) - 1`, computed per score column. That
rule applies to transcript-chunk methods and to no others. A method that
emits a score for every frame, such as a CLIP-feature detector at 1 fps,
has no uncovered frames and the rule never fires for it. A method whose
output is a set of predicted intervals rasterizes those intervals onto
the same grid and reports how it fills the gaps, in its own
documentation, not here.

## Per-corpus gold rules

### HateMM (test_clean, 215 videos with local media)

Spans come from the upstream `HateMM_annotation.csv` as parsed by
`scripts/duplex/hatemm_span_gold.py`. The video-level label is the id
prefix. Non-hate videos contribute all-negative frames, which is the
pooling convention LELA reports under and the reason the pooled metric
is meaningful at all.

One video, `hate_video_427`, carries a single degenerate span
(`00:00:01` to `00:00:00`). After the degenerate span is dropped it is a
hate video with no localizable gold, and it is excluded from the
localization cohort. Included: 214 videos, 85 hate and 129 non-hate.

### MultiHateClip EN and ZH (test split, owner-frozen 2026-08-18)

The video-level label is the upstream `Majority_Voting` field;
`Hateful` and `Offensive` are the positive classes. Two annotation
irregularities in the upstream `Duration` column needed an owner
decision, and these are the decisions:

**Rule (a): a Normal-majority video that carries leftover spans is
all-negative for its whole length.** Some annotators flagged segments in
videos the majority vote then called Normal. The video-level majority
vote governs, the leftover spans are ignored, and the video contributes
only negative frames. Affected in the current cohort: 8 EN, 5 ZH.

**Rule (b): a Hateful or Offensive video with no usable span is excluded
from localization evaluation.** Such a video is known to contain hate
but carries no information about where, so scoring it either way would
be a fabrication: treating it as all-negative would penalize a correct
detection, and treating it as all-positive would reward an indiscriminate
one. It is dropped, and the count is reported. Affected in the current
cohort: 4 EN, 4 ZH.

ZH zero-length `(0, 0)` spans are degenerate and are dropped under the
general rule above. One such span appears in the current ZH cohort, in
`BV1da411c76p`, which retains its second span `(5, 13)` and stays in the
cohort. A video whose only span is degenerate falls to rule (b).

The EN video `k9OtaMbK0Ac` is listed upstream in both train and test
with identical label and identical spans. It is counted as a test video
here and must be removed from any training split.

## Evaluation cohort

The cohort is the test-split videos whose media is present locally at
build time. Media for the remaining test videos is still being fetched;
when it lands, the arrays are rebuilt and their SHA256 changes. **A
number computed against one SHA256 is not comparable with a number
computed against another**, so every reported result names the array
hash it was scored against.

| Corpus | Test videos with local media | Excluded, rule (b) | Included | All-negative (rule a plus Normal) | Frames | Positive frames |
|---|---|---|---|---|---|---|
| HateMM test_clean | 215 | 1 (degenerate span) | 214 | 129 | 29266 | 7080 (24.2%) |
| MultiHateClip EN test | 161 | 4 | 157 | 112 | 5578 | 1383 (24.8%) |
| MultiHateClip ZH test | 149 | 4 | 145 | 104 | 4547 | 1074 (23.6%) |

Released arrays, built by `scripts/duplex/build_gt_arrays.py`:

| Array | SHA256 |
|---|---|
| `results/reproduction/gt/hatemm_test.npz` | `f4af758acbddd301c4898b1ce1a2436e6b260670ff3fcaedb99025d8a433ba65` |
| `results/reproduction/gt/mhclip_en_test.npz` | `1c7f4d08c9915d7d15b0cef9245e9b98db8ca745b7270a4e2181661968f59052` |
| `results/reproduction/gt/mhclip_zh_test.npz` | `b3d5239d45ba265418de3f41e1f21cded097be0fb761d10d3b21201cb99b3655` |

Each npz holds one `uint8` array per video keyed by video id. Each has a
JSON sidecar carrying the cohort counts, the exclusion lists with
reasons, the per-video frame and positive counts, and the hash above.
The npz is written with a fixed zip layout and fixed member timestamps,
so rebuilding from the same inputs reproduces the same bytes.

## Statistics

Pooled frame ROC-AUC is primary and pooled average precision is
secondary, both over the frames of all cohort videos concatenated. This
is the convention the frame-prediction literature reports against.
Alongside them, and never averaged away, goes the per-video macro ROC-AUC
over positive videos carrying both frame classes: the pooled number is
partly driven by separating positive videos from negative ones, and the
macro number is the one that says whether a method finds the right
seconds inside a video that has hate in it.

ROC-AUC is the Mann-Whitney rank statistic with midranks for ties.
Average precision is the step-wise form with tied scores collapsed into
one group, so it does not depend on input order within a tie. Both live
in `scripts/duplex/frame_eval_common.py`, and
`python scripts/duplex/frame_eval_common.py --selftest` checks the
ROC-AUC implementation against `scipy.stats.mannwhitneyu` to 16
significant digits along with the grid and conversion rules above.

## Regression

`scripts/duplex/frame_eval_regression_hatemm.py` recomputes the frozen
HateMM endpoint entirely through the shared module and the released
array: the locator's per-chunk `z_masked` spread onto the grid, uncovered
frames floored, scored against `hatemm_test.npz`. It reproduces the
original numbers exactly, 0.7450936536032907 pooled ROC-AUC and
0.5600748476714787 pooled AP over the same 28751 frames (6965 positive,
21786 negative), which is the evidence that the generalized base did not
move the endpoint.

The method scored 212 of the 214 gold videos. The two it skipped,
`hate_video_321` and `non_hate_video_512`, have Whisper chunks with
missing timestamps and therefore no usable spans. That is a method-side
gap and is recorded as such: the gold array keeps both videos, and any
method that can score them will be evaluated on them.

## Boundaries

No parameter of this protocol may be tuned after seeing a method's
numbers. The frame rate, the containment convention, the degenerate-span
rule, and the two MultiHateClip gold rules are frozen as of this
document. Cohort membership changes only when media arrives, never in
response to a result, and any change is visible in the array hash.
