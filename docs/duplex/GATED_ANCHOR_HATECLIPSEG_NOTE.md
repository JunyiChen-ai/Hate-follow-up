# Occupancy-gated anchored operating point on HateClipSeg: result note

**Date:** 2026-08-08. **Verdict:** **FAIL** under the frozen rule. The gate
fired correctly and clause 1 held, but clause 2 failed: the anchored operating
point scores 0.547 macro-F1 against the incumbent KDE valley's 0.652 and the
free two-component mixture's 0.563.
**Compute:** one GPU pass on a single RTX 5090, about 1.1 hours; all fitting on
CPU.
**Preregistration:** `docs/duplex/PREREG_gated_anchor_hateclipseg.md`
**Code:** `scripts/duplex/gated_anchor_hateclipseg.py`,
`scripts/duplex/hateclipseg_prep.py`, `scripts/duplex/run_hateclipseg.sh`
**Machine-readable result:** `results/hateclipseg/gated_anchor_results.json`

The gate did its job. HateClipSeg puts 46.4 percent of its corpus in the
model's positive saturation band, the highest occupancy of any corpus measured
so far, so the gate fired and selected the anchored arm exactly as the
preregistration predicted it would. What failed is the arm the gate selected.
Under the primary label collapse the anchored fit loses to a threshold rule
that this project has been trying to replace, and it loses for a reason the
occupancy story does not address: HateClipSeg's offensive union is 87.3 percent
positive, and a two-component mixture has no way to place a boundary that
extreme.

## What was run

The 435 annotated HateClipSeg videos were scored with the frozen c2 test
pipeline, unchanged. Audio was extracted and measured with silero voice
activity, re-transcribed with Whisper large-v3 in fp16 on cuda under automatic
language detection, passed through the frozen degeneracy gate, and fed uncapped
into one Qwen3-VL-8B-Instruct forward pass per video over a uniform-16 frame
grid at the 100,352-pixel per-frame budget. The readout is the raw logit
difference between the Yes and No token sets at the final prompt position. No
video received more than one judge call, and the 2B contrast arm was not run
because the anchored operating point is defined on 8B scores only.

The threshold analysis imports its EM, its posterior-0.5 decision, its
KDE-valley recipe and its prevalence resampling from
`scripts/duplex/anchored_operating_point.py`, the predecessor experiment, so
nothing about the fitting was re-implemented for this corpus.

Two label collapses were computed from the shipped video-level annotation,
using the rules frozen in the B1 pilot code. The primary collapse is the
offensive union: a video is positive if any of hateful, insulting, sexual,
violence or harm appears at video level. The secondary collapse is
hateful-strict: only the hateful category counts. `lexicons.json` was never
downloaded and never read.

## Attrition

| Stage | Videos |
|---|---:|
| Annotated in the shipped csv | 435 |
| Media file available (local or in the bucket) | 395 |
| Media decodable to 16 frames | 394 |
| In the scored split | 394 |
| With usable audio | 394 |
| Scored by the judge | 394 |

Forty annotated videos have no media anywhere. A fresh recursive listing of the
whole bucket, unfiltered by modification time, finds no object whose basename
matches any of those forty ids. One further video is a truncated download: the
container advertises a 274-second h264 stream but carries 138 kB and no
decodable video packet, and the copy in the bucket is byte-identical, so the
source file itself is broken rather than the transfer. The preparation script
prunes any media that yields no decodable frame, so that file leaves the split
through the ordinary missing-media path.

Twenty-six of the 395 available files could not be opened by the bundled decord
build, which lacks AV1 and VP9 support; 25 of those were re-extracted with the
system ffmpeg under the identical frame-index rule and the 26th is the truncated
file. Every one of the 394 frame sets was then decode-checked with a full PIL
open-and-load before any GPU time was spent, and the judge stage was asserted to
have scored exactly 394 videos before the analysis ran.

## Occupancy gate

| Quantity | Value |
|---|---:|
| Videos with raw z at or above +13 | 183 of 394 |
| Fraction in band | 0.4645 |
| Frozen firing threshold | 0.10 |
| Gate fires | yes |
| Arm selected | anchored, positive mean pinned at +15.0 |

For comparison, the same statistic on the four corpora that motivated the gate
is 37.7 percent on HateMM, 17.5 percent on ImpliHateVid, 3.1 percent on
MHClip-EN and 1.3 percent on MHClip-ZH. HateClipSeg is the most heavily
occupied corpus measured, which is what the preregistration expected from a
corpus whose normal class carries the same violence-heavy contamination
structure as HateMM.

## Comparator table

Macro-F1 on all 394 scored videos. The gate selected the anchored column.

| Method | Offensive union (primary) | Hateful strict (secondary) |
|---|---:|---:|
| Anchored +15.0 | 0.5474 | 0.6767 |
| Placebo anchor +10.0 | 0.2961 | 0.3244 |
| Placebo anchor +20.0 | 0.1419 | 0.3201 |
| Free 2-GMM | 0.5627 | 0.6412 |
| KDE valley (incumbent) | 0.6522 | 0.4479 |
| Labeled oracle (macro-F1 max) | 0.6697 | 0.7200 |
| Labeled oracle (hateful-F1 max) | 0.6180 | 0.7004 |

Decision boundaries in raw-z units:

| Method | Offensive union | Hateful strict |
|---|---:|---:|
| Anchored +15.0 | 10.89 | 10.89 |
| Placebo anchor +10.0 | 10.29 | 10.29 |
| Placebo anchor +20.0 | 23.91 | 23.91 |
| Free 2-GMM | 9.05 | 9.05 |
| KDE valley | -5.44 | -5.44 |
| Labeled oracle | 3.00 | 13.25 |

The label-free thresholds are identical across the two collapses, as they must
be: no fitted component reads a label, so the same score distribution produces
the same fit. Only the labels against which the predictions are scored change.
That makes the difference between the two columns a statement about the label
definition, not about the method.

## Prevalence stress test

Positives were subsampled to 50 and to 25 percent of their original count,
negatives kept whole, 200 resamples per rate under seed 20260808, with all
three label-free rules refitted on every resample. Drift is the median absolute
distance in raw-z units between a resample's threshold and the full-corpus
threshold.

| Collapse | Rate | Valley drift | Free 2-GMM drift | Anchored drift | Valley F1 | Free 2-GMM F1 | Anchored F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Offensive union | 0.50 | 0.372 | 0.929 | 0.207 | 0.6439 | 0.6269 | 0.6122 |
| Offensive union | 0.25 | 0.701 | 1.621 | 0.371 | 0.6138 | 0.6650 | 0.6575 |
| Hateful strict | 0.50 | 0.079 | 1.939 | 0.216 | 0.3634 | 0.5295 | 0.6332 |
| Hateful strict | 0.25 | 0.079 | 2.904 | 0.424 | 0.2830 | 0.4291 | 0.5806 |

The anchored threshold drifts 0.22 and 0.23 times as far as the free mixture's
under the primary collapse and 0.11 and 0.15 times as far under the secondary.
This replicates, on a fifth corpus, the one robustness effect the predecessor
experiment found real on HateMM. Every mixture fit produced a two-sided
decision region; the reported threshold is the crossing nearest the median score
of the set being fitted. The valley recipe failed to find two modes in 1 of 200
resamples at rate 0.50 and 5 of 200 at rate 0.25 under the primary collapse, and
those resamples are excluded from its rows.

## Clause-by-clause verdict

Primary collapse, offensive union. The gate fired, so clauses 1 and 2 apply and
clause 3 does not.

| Clause | Requirement | Result |
|---|---|---|
| 1. Correct selection | Selected arm within 0.02 macro-F1 of the better of {anchored, free} | PASS: anchored 0.5474 against free 0.5627, a shortfall of 0.0152 |
| 2a. Beats the incumbent | Anchored at least valley plus 0.03 | **FAIL**: 0.5474 against 0.6522, a shortfall of 0.105 |
| 2b. Not an ordinary mixture | Anchored at least free minus 0.005 | **FAIL**: 0.5474 against 0.5627, a shortfall of 0.0152 |
| 2c. Anchor location is load-bearing | Anchored at least each placebo minus 0.01 | PASS: +0.251 against +10.0 and +0.406 against +20.0 |
| 2d. Prevalence robustness | Anchored drift at most half the free drift at both rates | PASS: ratio 0.22 at rate 0.50 and 0.23 at rate 0.25 |
| 3. Gate declined a bad regime | Not applicable | The gate fired |
| **Overall** | All applicable clauses hold | **FAIL** |

Secondary collapse, hateful strict, reported for robustness and outside the
verdict: clause 1 passes with the gate selecting the better arm outright, and
all four parts of clause 2 pass, giving an overall PASS. The preregistration
made the primary collapse decisive, so this does not change the verdict.

## Cross-benchmark measurement of the c2 pipeline

Reported descriptively, outside the decision rule. This is the first time the
channel-restoration pipeline has been run on a fifth corpus.

| Corpus | Judge ROC-AUC (8B) | Positives | Negatives |
|---|---:|---:|---:|
| ImpliHateVid | 0.9473 | 199 | 201 |
| HateMM | 0.9232 | 86 | 129 |
| MHClip-ZH | 0.8547 | 45 | 104 |
| MHClip-EN | 0.7847 | 49 | 112 |
| HateClipSeg, hateful strict | 0.7709 | 180 | 214 |
| HateClipSeg, offensive union | 0.7538 | 344 | 50 |

HateClipSeg is the hardest corpus the judge has faced under either collapse.

Transcript restoration: all 394 videos carried a decodable audio track and all
394 produced a fresh transcript with no ASR error. The degeneracy gate accepted
372 and rejected 22, so 94.4 percent of the corpus was judged on a fresh
transcript. Median voice-activity fraction was 0.618. The repetition collapse
removed more than half the characters in 6.1 percent of transcripts.

Score distribution: minimum -19.0, median 12.5, maximum 19.5, mean 9.73,
standard deviation 7.95. The distribution is strongly right-shifted, with 65
percent of the corpus above z = +10 and only 1.8 percent at or below -13.

## Interpretation

The gate is not the broken part. It fired on the corpus the preregistration said
it should fire on, at an occupancy higher than HateMM's, and clause 1 held: the
arm it selected sits inside the frozen tolerance of the better arm. A gate that
picks the wrong regime on its first unseen corpus would have been fitted noise,
and this one did not.

The anchored arm is the broken part, and prevalence broke it, not occupancy.
Under the offensive union HateClipSeg is 87.3 percent positive. The anchored
mixture calls 59.6 percent of the corpus positive and the free mixture 67.3
percent, while the KDE valley places its threshold at -5.44 and calls 92.4
percent positive. The valley wins by accident of the corpus rather than by
insight: a rule that labels almost everything positive scores well when almost
everything is positive. Neither mixture can reach that operating point, because
a two-component decomposition of this score distribution puts its crossing near
z = +10 regardless of where the positive component is pinned.

The secondary collapse makes the same point from the other side. The fitted
model, the threshold and the predictions are byte-identical between the two
columns; only the labels move. At 45.7 percent prevalence the anchored arm beats
both the valley and the free mixture and every clause passes. What decides pass
from fail here is which annotation categories count as positive, not anything
the method does. The anchored operating point tracks the hateful category
specifically; the insulting, sexual, violence and harm categories that the
offensive union adds are scored lower by the judge and fall below the mixture's
crossing.

The prevalence-robustness effect replicated. On HateMM the predecessor found
anchored drift far below free-mixture drift and dismissed it as one corpus. It
now holds on a second occupied corpus under both label collapses, with ratios
between 0.11 and 0.23. Pinning the positive component does remove a degree of
freedom that prevalence otherwise exploits. That is a real and now twice-observed
property. It is also not enough: the preregistration made clause 2 conjunctive
precisely so that robustness alone could not carry a rule that loses on accuracy.

The placebo comparison is cleaner here than it was on the four earlier corpora.
Both placebos collapse, at 0.296 and 0.142 against the anchor's 0.547 under the
primary collapse, so on a corpus that genuinely occupies the saturation band the
anchor location is load-bearing in the way the saturation pilot claimed. That is
the mechanism working. The mechanism working and the rule still losing is the
honest summary of this experiment.

## Decision

- Do not promote the occupancy-gated anchored operating point. It failed its own
  confirmatory test on the only corpus that carried confirmatory weight.
- Do not rescue it with the secondary collapse. The collapse was declared
  secondary before the run, and reversing that now would be exactly the post-hoc
  move the preregistration existed to prevent.
- Keep the occupancy gate as a diagnostic. It correctly identified the regime on
  an unseen corpus and it costs nothing to compute. Any successor threshold rule
  should report it.
- The open problem has changed shape. It is no longer "where should the positive
  component sit" but "how does a label-free rule discover that a corpus is 87
  percent positive". Neither the valley nor either mixture estimates prevalence
  from anything except the shape of the score distribution, and on this corpus
  that shape is misleading. A successor needs a label-free prevalence signal, and
  this project does not have one.
- Record that HateClipSeg is the hardest corpus for the judge so far, at ROC-AUC
  0.754 under the primary collapse. Threshold selection is not the binding
  constraint there: the labeled oracle reaches only 0.670 macro-F1, so even a
  perfect threshold leaves most of the gap open.

## Implementation choices not fixed by the preregistration

- The dataset was placed in the standard benchmark layout so that every c2 stage
  ran unmodified. Title and dataset transcript are empty strings, because
  HateClipSeg ships neither; HateMM and ImpliHateVid also carry an empty title,
  so this is not new. It does mean the 22 gate-rejected videos were judged with
  no transcript at all rather than falling back to a dataset transcript.
- The platform rules block is the YouTube one, which is what the frozen judge
  selects for every dataset except MHClip-ZH. HateClipSeg mixes BitChute and
  YouTube sources, so no unmodified branch fits it exactly.
- Prevalence resample streams are keyed by collapse index 4 and 5, continuing the
  predecessor's numbering of corpora 0 to 3. The choice is arbitrary and the run
  is deterministic given it.
- EM stopping, the standard-deviation floor of 0.1, the two-sided decision region
  handling and both oracle conventions are inherited unchanged from the
  predecessor script.
