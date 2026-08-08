# Self-trained readout kill test: result note

**Date:** 2026-08-09. **Verdict: DEAD.**
**Preregistration:** `docs/duplex/PREREG_selftrained_readout_killtest.md`,
frozen before any head was fitted on any corpus.
**Machine-readable result:** `docs/duplex/reports/selftrained_readout_killtest.json`
(also written to the gitignored `results/selftrained_readout/results.json`).
**Compute:** CPU only, 3 minutes 17 seconds wall clock for all five corpora. No
model call, no GPU.

## Headline

Training a linear head on the judge's own committed extremes does not resolve
the congested middle. On HateMM the head reaches macro-F1 **0.6960** against
the preregistered bar of 0.75, and its posterior ranks the corpus at AUC
**0.8893** against the bar of 0.9132. Two of four clauses fail, and the
ImpliHateVid regression guard fails as well, so the family is dead under the
frozen rule.

The two clauses that pass are the diagnostic ones, and they say something worth
keeping. The hidden-state head beats the z-only control by 0.0398, just over
the 0.03 margin, so the hidden space does carry something the scalar cannot
express. The shuffled-pseudo-label placebo lands at 0.4248, fully 0.2712 below
the head, so the head learned structure and not band arithmetic. Both of the
"is there a mechanism" clauses hold. Both of the "does it work" clauses fail.

The movement analysis explains the shortfall exactly. Of the 70 false positives
the incumbent KDE valley makes on HateMM, the head reclassifies **8** as
normal. Of the valley's 3 false negatives it recovers **0**. It introduces no
new error of either kind: the head's positive set is a strict subset of the
valley's, 145 videos out of the valley's 153, and the 8 removed videos are all
genuine false positives. The head and the valley agree on 96.3 percent of the
corpus. So the self-trained readout is a small, one-directional cleanup of the
incumbent threshold, not a new reading of the congested middle. Eight of
seventy is not a resolution of the middle; it is a trim at the edge of it.

## Frozen clauses

| # | Clause | Bar | Observed | Result |
|---|---|---:|---:|---|
| 1 | Performance | HateMM head macro-F1 >= 0.75 | 0.6960 | **FAIL** |
| 2 | Ranking preserved | head posterior AUC >= 0.9132 | 0.8893 | **FAIL** |
| 3 | Hidden space is load-bearing | head >= z-only + 0.03 | 0.6960 vs 0.6562, delta 0.0398 | PASS |
| 4 | Signal, not band arithmetic | placebo mean <= head - 0.10 | 0.4248 vs 0.6960, delta 0.2712 | PASS |
| G | Regression guard | ImpliHateVid head macro-F1 >= 0.8523 | 0.8522 | **FAIL** |

The guard fails by 0.0001. The exact preregistered floor is the ImpliHateVid
valley 0.8823345 minus 0.03, which is 0.8523345, and the head sits at 0.8522331,
short by 0.0001014. This is a hairline miss and should be read as "the head
matched the valley to within a rounding error and did not beat it", not as a
collapse. It does not change the verdict, which clauses 1 and 2 already
determine.

## What ran

Everything was read from disk. The inputs are the layer-27 final-token hidden
states and the raw z scores written by the frozen single-call Qwen3-VL-8B judge
runs, in `results/testruns/{hatemm,implihatevid,mhclip_en,mhclip_zh}/judge_8b/`
and `results/hateclipseg/judge_8b/`. Each stored array is 37 rows by 4096
dimensions in float16; row 27 was taken and cast to float32.

The pipeline per corpus is the preregistered one and nothing else. Videos with
z >= +13 became pseudo-positives and videos with z <= -13 became
pseudo-negatives. A corpus abstains if either band holds fewer than 10 videos.
The feature matrix is the layer-27 row standardized per dimension with the mean
and standard deviation of that corpus's own hidden states over every scored
video, with no label read; no dimension had zero variance on any corpus. The
head is L2 logistic regression with lambda fixed at 1.0, fitted on the
pseudo-labeled extremes only, and the decision is posterior at or above 0.5 on
every video including the extremes. No threshold was searched anywhere in this
experiment.

The z-only control repeats that fit with the standardized scalar z as the sole
feature. The placebo permutes the pseudo-labels inside the extreme set, 20
permutations at seed 20260808, and reports the mean and standard deviation of
macro-F1 over those permutations.

Label loading is imported from `scripts/duplex/anchored_operating_point.py` and
the HateClipSeg preparation module, and the script aborts unless every corpus's
KDE-valley macro-F1 reproduces its committed value to within 0.002. All five
reproduced, so the labels here are the committed labels by construction.
MHClip-EN and MHClip-ZH use the Offensive-maps-to-1 union collapse; HateClipSeg
uses the strict collapse in which only the hateful category maps to 1.

## Five corpora

| corpus | n | prev. | band + | band - | applicable | head macro-F1 | head AUC | valley macro-F1 |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| HateMM | 215 | 0.400 | 81 | 25 | yes | 0.6960 | 0.8893 | 0.6562 |
| ImpliHateVid | 400 | 0.498 | 70 | 144 | yes | 0.8522 | 0.9383 | 0.8823 |
| HateClipSeg (strict) | 394 | 0.457 | 183 | 7 | no | — | — | 0.4479 |
| MHClip-EN | 161 | 0.304 | 5 | 43 | no | — | — | 0.6962 |
| MHClip-ZH | 149 | 0.302 | 2 | 17 | no | — | — | 0.7256 |

Three of five corpora abstain under the frozen occupancy rule, and they abstain
for opposite reasons. MHClip-EN and MHClip-ZH have almost no positive-band
occupancy, 5 and 2 videos, which the occupancy facts already predicted.
HateClipSeg fails on the other side: 183 of its 394 videos sit in the positive
band but only 7 sit in the negative band, so the method has no pseudo-negatives
to learn from. That direction was not anticipated in the preregistration, which
listed HateClipSeg as reportable rather than at risk of abstaining.

One preregistration figure did not match. The protocol described HateMM's
positive band as holding 57 videos; the frozen scores put 81 videos at z >= +13,
which is the count the commitment-precision note already committed. The band
edge was never changed, so this is a transcription error in the preregistration
text rather than a protocol deviation, and it favored the method by giving the
head more pseudo-positives than the protocol expected.

The method is therefore applicable on two of five corpora, and on those two it
is at best a small improvement and at worst a small regression. The pseudo-label
bands themselves are clean where they exist: on HateMM the positive band is
87.7 percent gold-hateful and the negative band 96.0 percent gold-normal, for
89.6 percent pseudo-label accuracy; on ImpliHateVid the corresponding numbers
are 97.1, 97.9 and 97.7 percent. The pseudo-labels were not the weak link.

## Controls

| corpus | head | z-only | placebo mean | placebo sd | placebo range |
|---|---:|---:|---:|---:|---|
| HateMM | 0.6960 | 0.6562 | 0.4248 | 0.0749 | 0.3019 to 0.5488 |
| ImpliHateVid | 0.8522 | 0.8675 | 0.4874 | 0.0824 | 0.4009 to 0.6075 |

The z-only control on HateMM reproduces the KDE valley's confusion matrix
exactly: 83 true positives, 70 false positives, 3 false negatives, 59 true
negatives. Its implied z threshold is -2.4390 against the valley's -2.3455, and
no video sits between the two, so the two rules make identical decisions. That
coincidence is worth recording. It means the extremes, read through the scalar
alone, rediscover the incumbent operating point and add nothing; the entire
0.0398 that clause 3 measures comes from the hidden space.

Ranking tells the same story in reverse. The z-only control's AUC on HateMM is
0.9232, the judge's own ranking, while the head's is 0.8893. The head buys a
better operating point at the cost of a worse ordering, which is why clause 2
fails while clause 3 passes. On ImpliHateVid the head is worse on both axes
than the scalar it was trained from: 0.8522 against 0.8675 in macro-F1 and
0.9383 against 0.9473 in AUC.

## Where the errors moved

| corpus | valley FP | of those, head calls normal | valley FN | of those, head recovers | new FP | new FN | agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| HateMM | 70 | 8 | 3 | 0 | 0 | 0 | 0.963 |
| ImpliHateVid | 32 | 10 | 15 | 0 | 0 | 22 | 0.920 |

On HateMM the head is a pure contraction of the valley's positive set and every
video it removes is a real false positive, which is why macro-F1 rises without
any compensating loss. On ImpliHateVid the same contraction is harmful: the
head removes 10 real false positives but also drops 22 true positives that the
valley had caught, and it recovers none of the valley's 15 misses. The head
never converts a negative decision into a positive one on either corpus. It
only ever shrinks the positive set. That is a structural property of what the
extremes teach, and it caps what this design can do: a rule trained on the
committed extremes inherits the judge's positive class and can only prune it.

## Interpretation

The preregistration named the consequences in advance. Clause 1 or 2 failing
means extreme-supervision does not resolve the middle, and the family is retired
honestly. That is what happened. The middle is where all 70 of HateMM's
threshold errors live, and the head moved 8 of them.

The two passing clauses keep a narrower fact alive, and it is the same fact the
readout-bottleneck test established: layer-27 hidden states hold judgment-relevant
structure that the scalar readout discards. The 0.0398 margin over an exactly
valley-equivalent z-only control is a clean demonstration of that, obtained
without a single human label. What this test adds is the boundary: that
structure is real but thin, and self-supervision from the model's own committed
extremes is not enough to extract it in usable quantity. The extremes are clean
(90 to 98 percent pure) and they still only teach the head to prune the positive
set slightly.

There is also an applicability finding that stands on its own. The occupancy
requirement of any extreme-supervised method is two-sided, and no corpus in this
program satisfies both sides comfortably. Two corpora starve on the positive
side, one starves on the negative side, and the two that qualify do so with 25
and 70 videos in their smaller band. A method that needs both saturation bands
populated is a method that applies to a minority of corpora, which is a
deployment fact independent of how well the head performs where it does apply.

## Decision

The self-trained readout family is closed. No follow-up fit, no lambda sweep, no
alternative band edge: the preregistration forbids fallback tuning and the
failure is not marginal on clause 1, which misses by 0.054, or on clause 2,
which misses by 0.024.

The result that carries forward is the one clause 3 isolates. Hidden-state
access is worth 0.0398 macro-F1 over the best scalar rule on HateMM when the
supervision is free, and that number is now measured rather than assumed. Any
future use of layer-27 structure has to beat it with a source of supervision
that is not the model's own extremes.
