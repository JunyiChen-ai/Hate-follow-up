# Frame-level evaluation protocol (corpus-general)

**Amended 2026-08-23 (model-selection split only).** Trained baselines now
preserve HateMM's released train/validation/test split and MultiHateClip's
released train/valid/test split. The earlier baseline runs merged train and
validation and then carved 10%; they are retained as `legacy-resplit-val`.
HateClipSeg's frozen test IDs are unchanged; 20% of its former training cohort
is now a seeded, stratified validation manifest. Frame-level test GT and its
cohort remain unchanged by this amendment.

**Frozen:** 2026-08-18, Phase 1 of the baseline reproduction plan, before
any baseline is trained or scored. This document supersedes nothing: it
generalizes the HateMM-only protocol frozen in
`PREREG_frame_level_evaluation_hatemm.md` to every corpus in the
reproduction study, and adds the per-corpus gold rules that HateMM did
not need. The HateMM numbers produced under the earlier prereg are
reproduced bit for bit under this protocol (see "Regression" below), so
the generalization costs no comparability.

**Amended 2026-08-19:** HateClipSeg joins as the fourth corpus. The
amendment is additive — grid, containment convention, degenerate-span
rule and statistics are unchanged, the three existing arrays keep their
hashes, and everything specific to the new corpus lives in its own
section and its own build script.

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

### HateMM (test_clean, 215 videos, all with local media)

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

### HateClipSeg (our test split, added 2026-08-19)

HateClipSeg is the fourth corpus and the only one whose timeline is
annotated *exhaustively*. The other three draw hate spans on an
otherwise unlabelled timeline; HateClipSeg partitions each video into
segments and gives every segment a six-dimensional multi-hot label
`[normal, hateful, insulting, sexual, violence, harm]`. The gold rules
below follow from that difference, and because they do, the arrays are
built by their own script, `scripts/duplex/build_gt_arrays_hateclipseg.py`.
`build_gt_arrays.py` is untouched and its three arrays keep their hashes.

**Split provenance: the split is ours.** The HateClipSeg paper reports an
80/20 division and publishes no video ids, so there is nothing upstream
to intersect with and nothing to reproduce. `reproduction_splits.py`
draws its own: eligible videos are the annotated ones whose media is
present locally, a video is positive if at least one of its segments is
offensive under the union rule, ids are sorted then shuffled inside each
stratum by `random.Random(234)` — one generator, strata visited in the
fixed order (negative, positive) — and the first `round(0.2 n)` of each
stratum go to test. Sorting before shuffling makes the draw independent
of filesystem order; re-running reproduces both manifests byte for byte.
Every number reported on this corpus must name the split as ours.

Of the 435 annotated videos, 394 have local media. The 41 without it are
listed by id in `results/reproduction/splits/manifest_report.json`. One
of the 41, `yt_NzvfkIYS5Yg`, is held on the B2 mirror but the object
there is a 135 KiB truncated file — `ffprobe` reports `partial file` and
cannot read its audio stream — so it stays out. The corpus is heavily
positive at the video level: 344 of the 394 carry at least one offensive
segment. The draw gives 315 train and 79 test.

| Stratum | Eligible | Train | Test |
|---|---|---|---|
| Video positive (≥1 offensive-union segment) | 344 | 275 | 69 |
| Video negative | 50 | 40 | 10 |
| Total | 394 | 315 | 79 |

**Frame rule (primary).** A frame is positive if and only if the segment
covering it, half-open, is offensive under the **union rule**: any of the
five non-normal dimensions set. This is the rule
`sentinel_localization_pilot.is_offensive_union` already applies, so the
frame gold and that pilot's cohort agree by construction rather than by
coincidence.

**Frame rule (sensitivity).** A second array over the same videos and the
same grid marks a frame positive if and only if its covering segment sets
dimension 1, `hateful`, alone. It exists because the union rule is broad
— `violence` and `harm` are in it — and a reader is entitled to ask
whether a method tracks hate or tracks the union. Same cohort, same frame
counts, so the two are directly comparable. The primary stays primary.

**Segment tiling, measured before the rules were fixed.** Across all 435
annotated videos and 11,714 segments: every video's first segment starts
at 0.00; there are **0 gaps and 0 overlaps** between adjacent segments,
to a 1 µs tolerance. The annotation genuinely tiles. Two consequences.
First, the overlap-resolution rule — a frame covered by both a positive
and a negative segment is positive, the same union the other corpora use
— is frozen for completeness but never fires here. Second, an uncovered
frame is negative, and after the tiling measurement that rule can only
reach the sub-second tail past the last segment: 0 frames in the test
cohort, and 1.2 s of audio corpus-wide.

**Degenerate segments.** 23 segments have `end <= start`. Every one is
the **final** segment of its video, and in every case its `end` equals
the media duration to within 0.12 s, while its `start` is the previous
boundary. They are dropped and counted, as elsewhere in this protocol.
Dropping them removes no covered interval, since the interval each names
is empty. One falls in the test cohort.

**Annotation-clock rule.** For 18 of the 394 videos with media, the last
usable segment ends 1.1 to 20.8 s past *both* the wav and the container
duration — the segmentation was produced against a longer version of the
video. Where a frame in that region lands is not recoverable, and neither
answer is honest, so those videos are excluded: **a video whose last
usable segment ends more than 1.0 s past the media is dropped.** The
tolerance is one frame on the 1 fps grid, which is the largest overshoot
that cannot move any label; it is fixed by the grid, not chosen against a
cohort size. All 18 happen to fall in train, so the rule excludes nothing
from the current test cohort — but it is frozen now, before any method is
scored, and it matters to any baseline that builds train-side targets.
The audit runs corpus-wide and is written into the sidecar.

Six further videos have an audio stream shorter than the container by 1.2
to 7.3 s while the annotation matches the container exactly. There the
annotation clock is sound and only the audio stops early, so they are
kept and the grid, which is built from the wav, simply truncates the
annotated tail. One of them, `bit_7EOOUGa9y9h4`, is in the test cohort
and loses 7.25 s of annotated timeline this way.

| Array | Videos | Frames | Positive | Positive rate | Both classes within video | All-negative | All-positive |
|---|---|---|---|---|---|---|---|
| Primary, offensive union | 79 | 18839 | 9900 | 52.6% | 67 | 10 | 2 |
| Sensitivity, hateful strict | 79 | 18839 | 4039 | 21.4% | 37 | 41 | 1 |

The both-classes column is the one to watch, because the per-video macro
ROC-AUC is computable only on videos that carry both frame classes. Under
the primary rule 67 of 79 HateClipSeg test videos do, or 85%, against 85
of 214 on HateMM (40%), 44 of 158 on MultiHateClip EN (28%), and 7 of 153
on ZH (5%). The macro statistic, which the other three corpora compute
over a thin slice, rests on most of the cohort here, and the pooled number
is correspondingly less driven by separating positive videos from negative
ones. That is what the finest annotation in the study buys, and it is the
reason HateClipSeg is worth carrying.

Timestamped ASR covers the split completely: all 394 locally held videos
have a usable record in
`results/interleaved_timeline/hateclipseg/timestamped_chunks.jsonl` — no
errors, no empty chunk lists, every one reproducing its frozen text — so
315 of 315 train and 79 of 79 test videos are covered and no video needed
re-transcribing. `scripts/duplex/hateclipseg_asr_coverage.py` measures
this and writes the gap list; the list is currently empty, and the
`hateclipseg_missing` corpus entry in `interleaved_timeline_asr.py`
consumes it if media lands later.

## Evaluation cohort

The cohort is the test-split videos whose media is present locally at
build time. Media for the remaining test videos is still being fetched;
when it lands, the arrays are rebuilt and their SHA256 changes. **A
number computed against one SHA256 is not comparable with a number
computed against another**, so every reported result names the array
hash it was scored against.

One thing thins the MultiHateClip cohorts before any gold rule applies:
media. 162 of the 200 upstream EN test videos and 157 of the 200 ZH have
been fetched; the rest are unavailable and are listed by video id in the
sidecars. HateMM loses nothing at this stage: all 215 test_clean videos
have local media. HateClipSeg loses 41 of its 435 annotated videos to
media, but that loss happens *before* the split is drawn rather than
after, so it thins the pool the draw runs over instead of thinning a
fixed test list; the 41 ids are in the split report.

The local annotation mirror `annotation(new).json` no longer thins
anything. It kept 890 of the 1000 upstream EN videos, which is why
`span_gold_{en,zh}.json` covers only 182 EN and 176 ZH test videos, but
the gold itself — the majority vote and the `Duration` spans — comes from
the upstream TSVs, and the mirror supplies only a cross-check label. A
test video absent from the mirror therefore still has complete gold, and
`build_gt_arrays.py` reads it from the upstream TSV rather than dropping
the video. One EN video (`hXv7bR9i5Q4`, Offensive, span (1, 21)) reaches
the cohort this way; the sidecar lists it under
`videos_with_gold_from_upstream_tsv_only`.

Duration comes from the timestamped-chunk manifest where that manifest
has an entry, and from the wav header otherwise — the same quantity
either way, since the manifest's `wav_duration` is itself the wav length.
The nine test videos whose media arrived after the frozen ASR runs (1 EN,
8 ZH) take the wav-header route; the sidecar records the source per
video.

| Corpus | Upstream test | With local media | Excluded, rule (b) | Included | All-negative | Frames | Positive frames |
|---|---|---|---|---|---|---|---|
| HateMM test_clean | 215 | 215 | 1 (degenerate span) | 214 | 129 | 29266 | 7080 (24.2%) |
| MultiHateClip EN test | 200 | 162 | 4 | 158 | 112 | 5600 | 1403 (25.1%) |
| MultiHateClip ZH test | 200 | 157 | 4 | 153 | 110 | 4817 | 1121 (23.3%) |
| HateClipSeg test (ours) | 79 | 79 | 0 (clock rule) | 79 | 10 | 18839 | 9900 (52.6%) |

The "all-negative" column counts included videos with no positive frame:
every Normal-majority video, including the 8 EN and 5 ZH whose leftover
spans rule (a) discards. The HateClipSeg row reads differently from the
other three: its "upstream test" is our own 79-video draw, not a published
list, and its exclusion column is the annotation-clock rule rather than
rule (b).

Released arrays, built by `scripts/duplex/build_gt_arrays.py` (first
three) and `scripts/duplex/build_gt_arrays_hateclipseg.py` (last two):

| Array | SHA256 |
|---|---|
| `results/reproduction/gt/hatemm_test.npz` | `f4af758acbddd301c4898b1ce1a2436e6b260670ff3fcaedb99025d8a433ba65` |
| `results/reproduction/gt/mhclip_en_test.npz` | `7099195e0a2bbcfb3e9be6e4117d709393beea06b78adf6bcefeba736d8c12c8` |
| `results/reproduction/gt/mhclip_zh_test.npz` | `1abd4ae620add7e9d45b3895357ece7446a14ee69f2e509b724d3e63b5069cf6` |
| `results/reproduction/gt/hateclipseg_test.npz` | `e7d164c04d77262f4cb77ad14592751b2cfdccc53f76a8a2d55162b8e0196b31` |
| `results/reproduction/gt/hateclipseg_test_hateful_strict.npz` | `4e8e705c915d7197b6f3e5be580ad52bd63d47217da8c3d6f69c865814ae05d7` |

Frozen split manifests for HateClipSeg, one video id per line:

| Manifest | SHA256 |
|---|---|
| `results/reproduction/splits/hateclipseg_train.txt` | `5eb86a2cfdf070c7024e284925819a980a0c25d5906f341c80ef3b4f00b83319` |
| `results/reproduction/splits/hateclipseg_test.txt` | `0d6486438a27493322ffdc862cbcc079448a9b7530fd53b0203564992f800a2b` |

The HateMM hash is unchanged from the first build: its cohort did not
move, and the two MultiHateClip rebuilds are purely additive — no array
that existed in the first build changed a single byte. The frame-level
regression (`scripts/duplex/frame_eval_regression_hatemm.py`) still
recovers the frozen endpoint, 0.7450936536 ROC-AUC and 0.5600748477
PR-AUC, against the rebuilt HateMM array.

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
document. So are the four HateClipSeg rules added on 2026-08-19: the
union positive rule, the hateful-strict sensitivity rule, the
uncovered-frame and overlap conventions, and the 1.0 s annotation-clock
tolerance. So is the HateClipSeg split — seed 234, 80/20, stratified by
video-level label — which was drawn and hashed before any method saw the
corpus and may not be redrawn to improve a number. Cohort membership
changes only when media arrives, never in response to a result, and any
change is visible in the array hash.
