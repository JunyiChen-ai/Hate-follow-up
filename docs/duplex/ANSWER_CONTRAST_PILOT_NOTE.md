# Answer-contrast readout pilot: result note

**Date:** 2026-08-10. **Verdict: DEAD**, on the preregistered z-renaming abort.
**Preregistration:** `docs/duplex/PREREG_answer_contrast_pilot.md`, frozen
before any GPU call and amended only to record two measurements, never a bar.
**Machine-readable results:** `docs/duplex/reports/answer_contrast_pilot.json`
(Stage B) and `docs/duplex/reports/answer_contrast_stage_a.json` (Stage A).
**Compute:** 31 minutes of GPU on one RTX 5090 for 5,650 video extractions,
plus 27 minutes of CPU for Stage A and 3 seconds for the Stage-B analysis.

## Headline

The contrast was manufactured on the answer side instead of the question side,
and it produced the same failure in a stronger form. The leading direction of
the Yes-minus-No difference correlates with the scalar readout z at Spearman
0.983, 0.979, 0.949 and 0.946 on the four evaluation corpora. The preregistered
abort fires at 0.90 on any one of them, so the design is dead before its
clauses are read.

The comparison with the earlier failure is the point. Reading the state at the
last prompt position, the corpus-wide first principal component tracked z at
0.992 and carried 46 percent of the variance. Forcing the model through two
opposed answers and reading the difference gives a first component that tracks
z at up to 0.983 and carries **64 percent** of the variance. Manufacturing the
contrast did not free the representation from the scalar; it concentrated more
of the variance onto it.

## Stage A: the free control, and two premise corrections

Stage A asked whether the earlier PCA failure has a cheap rescue. Inside the
frozen saturation band z was believed to be flat, so a PCA fitted on band
members only could not rediscover it. The control ran on 183 HateClipSeg, 81
HateMM and 70 ImpliHateVid in-band videos, standardized within band, three
components at each of layers 18, 27 and 36, scored sign-free on the
HateClipSeg in-band arena of 126 strict-hate against 57 non-strict videos.

| Fit | Layer | PC | Variance share | Spearman with in-band z | Sign-free AUC |
|---|---:|---:|---:|---:|---:|
| self | 18 | 1 | 0.178 | 0.041 | 0.550 |
| self | 18 | 2 | 0.125 | 0.096 | 0.566 |
| self | 27 | 1 | 0.199 | 0.801 | 0.632 |
| self | 27 | 2 | 0.123 | 0.393 | 0.527 |
| self | 36 | 1 | 0.188 | 0.856 | **0.635** |
| self | 36 | 2 | 0.111 | 0.273 | 0.505 |
| pooled | 27 | 1 | 0.202 | −0.789 | 0.629 |
| pooled | 36 | 1 | 0.187 | −0.842 | 0.632 |

The best of the nine primary candidates reaches 0.635 against the 0.70 floor,
so the control fails and Stage B was entitled to run. Pooling the three
corpora's bands before the fit changes nothing, which is why it is reported as
descriptive.

Two things measured here contradict the premise the idea was written on, and
both were recorded in the preregistration before any GPU call.

**z is not flat inside the band.** The in-band range is +13.000 to +19.500 with
a standard deviation of 1.50, and z reaches AUC **0.6472** on the C1 arena. The
claim that the incumbent scalar "cannot order at all" there is false. The C1
bar of 0.70 was fixed before this was known and was not changed; it simply
acquired the stronger meaning of having to beat 0.6472 rather than 0.5.

**The C1 arena is close to unseparable at layer 27 even with labels.** An L2
logistic probe over the 183 in-band videos, leave-one-out, reaches **0.6253**,
below the scalar it was meant to beat. The same probe on the MHClip-EN
construct stratum reaches 0.837, so the instrument works; this arena is simply
hard. Band-conditioning removes the variance that carried the earlier
supervised signal along with the variance that carried z.

## What Stage B ran

The prompt is the frozen judge's prompt: the same system message, the same
`prag` reader block, the same platform rules, the same title, the same 16
frames at the same pixel budget, and the same fresh Whisper transcript read
through the same override map with no character cap. One prefill produces the
frozen z and the key-value cache. The cache is then cropped back to the prefill
length between two single-token forwards, one appending the first token of
"Yes" and one appending the first token of "No", both corpus-constant. The
hidden state at the appended position is stored for all 37 rows in both arms.

Two checks establish that this is the frozen judge and not a near-relative.
The cached two-arm path was compared against a full uncached forward with the
answer token concatenated to the input: relative Frobenius error 0.0098 to
0.0110 and cosine 0.9994 to 0.9996 on the layer-27 difference, which is bf16
arithmetic noise. And the recomputed z reproduces the frozen `scores.jsonl` z
exactly on all five runs, Spearman 1.000 and maximum absolute difference 0.000
on 2,453 videos.

That second check is worth dwelling on, because the first extraction pass
failed it. The extractor's defaults read the dataset transcript capped at 300
characters, while the frozen judge reads a fresh Whisper transcript with no
cap. Pre-flight caught the difference at Spearman 0.827 and a median absolute z
difference of 1.75, the run aborted before a single clause was computed, and
the 3,197 videos of that pass were discarded and re-extracted. The gate did the
job it was written for.

The readout is the preregistered one. Δh(v) is the Yes arm minus the No arm at
layer 27; each corpus is centred by its own unlabeled mean Δh; the per-dimension
scale and the single principal direction come from the unlabeled fit corpus;
the sign is fixed once on the fit corpus by the sign of its Spearman
correlation with z.

## Aborts

| Abort | Rule | Value | Fires |
|---|---|---:|---|
| A1, degeneracy | median cos(Δh, mean Δh) ≥ 0.99 on the fit set | 0.902 | no |
| A2, z-renaming | \|Spearman(score, z)\| ≥ 0.90 on any eval corpus | **0.983** | **yes** |

A1 not firing matters: the manufactured contrast is genuinely
evidence-conditioned rather than one constant vector. Videos differ in Δh, at a
median cosine of 0.90 to the corpus mean, and the residual has real structure —
the pairing placebo confirms it below. The mechanism half of the story is
therefore intact. It is the direction of that structure that kills the design.

A2 fires on every corpus, not one:

| Corpus | Spearman(score, z) |
|---|---:|
| ImpliHateVid test | 0.983 |
| HateMM test | 0.979 |
| HateClipSeg | 0.949 |
| MHClip-EN test | 0.946 |

## Clauses

Reported in full, as the preregistration requires, even though A2 already
settles the verdict.

| # | Clause | Bar | Observed | Reference | Result |
|---|---|---:|---:|---|---|
| C1 | HateClipSeg band, strict 126 vs non-strict 57 | 0.70 | 0.6256 | in-band z 0.6472; band PCA 0.6348; supervised ceiling 0.6253 | **FAIL** |
| C2 | MHClip-EN, no-target 34 vs protected-target 15, sign-free | 0.72 | 0.6765 | z 0.679; probe 0.837 | **FAIL** |
| C3 | HateMM, 70 valley false positives vs 83 true positives | 0.65 | 0.9232 | z 0.9036 | PASS |
| C4 | Pairing placebo margin on the C1 arena | 0.05 | 0.0920 | placebo 0.5336 | PASS |
| C5 | ImpliHateVid test, score under the frozen valley recipe | 0.852 | 0.8665 | incumbent valley 0.8823 | PASS |

The three passes are the three clauses that a good copy of z would also pass,
and the two failures are the two clauses that required something other than z.
C3 asks whether the score separates the judge's own mid-band errors from its
own mid-band hits; z itself scores 0.9036 there, so a readout correlated with z
at 0.979 passing at 0.9232 is a restatement of the correlation. C5 asks the
score to survive substitution into the valley recipe; it lands at 0.8665
against the incumbent's 0.8823, that is, slightly worse than the scalar it
copies. C1 and C2 are the two arenas where z is known to be weak, and the score
is weak in exactly the same places: 0.6256 against z's 0.6472 on the first,
0.6765 against z's sign-free 0.679 on the second.

C2 deserves one more line. Read directionally the score reaches 0.324, which
means it ranks protected-target hate above target-free abuse. That is the same
severity ordering the scalar readout makes, and the opposite of the ordering
the supervised probe finds at 0.837. The answer-side contrast inherits not just
z's magnitude but z's confusion.

## Controls

**Pairing placebo.** With the No arms permuted across videos by a fixed
derangement and the whole pipeline re-run, the C1 arena AUC falls from 0.6256
to 0.5336. The margin of 0.092 clears the 0.05 bar, so the pairing of the two
arms carries information and the score is not an artefact of either arm's
marginal distribution. Given A2, what the pairing carries is z.

**Norm control.** The residual norm ‖r(v)‖ alone reaches 0.6536 on the C1
arena, which is *higher* than the projection's 0.6256. The preregistered
condition is met. It is recorded as not decisive because C1 failed and there is
no passing score for it to explain away, but the direction is informative on
its own: how far the commitment state moves is a slightly better in-band
ordering than where it moves to.

## Layers

| Layer | PC1 variance share | median cos(Δh, mean) | max \|Spearman with z\| | C1 | C2 sign-free | C3 |
|---:|---:|---:|---:|---:|---:|---:|
| 18 | 0.215 | 0.996 | 0.860 | 0.6125 | 0.6922 | 0.8725 |
| 27 | 0.639 | 0.902 | 0.983 | 0.6256 | 0.6765 | 0.9232 |
| 36 | 0.628 | 0.803 | 0.987 | 0.6342 | 0.6824 | 0.9191 |

Layer 18 is the only depth that escapes the A2 abort, at 0.860, and it escapes
it by being degenerate instead: the median cosine of Δh to the corpus mean is
0.996, so at that depth the model's answer-side difference is very nearly the
same vector for every video. The two aborts trade off along depth. Early, the
contrast is evidence-independent; late, it is z. There is no layer at which it
is both evidence-conditioned and something other than the readout, and layer 18
fails C1 and C2 anyway at 0.6125 and 0.6922.

## Interpretation

The pilot's story was that the softmax saturates but the residual stream need
not, so the state of committing to Yes against this evidence should carry
construct information that the scalar discards. The first half of that story
survived: Δh is evidence-conditioned at layers 27 and 36, and the placebo shows
the pairing is load-bearing. The second half did not. The direction along which
Δh varies most is the direction the logit contrast already reports.

That result is not a surprise once stated in the right order, and it sharpens
the boundary the readout-bottleneck test drew. The difference between the two
answer states is, by construction, the direction in which the model's next-token
distribution moves between Yes and No. The logit contrast z is a linear
functional of very nearly that same difference, read through the unembedding.
Asking for the leading direction of Δh is therefore close to asking for the
direction that most changes z, and unsupervised variance answers the question
it is asked. Forcing the answer does not create a new degree of freedom; it
re-expresses the one that was already being read, and at layer 27 it does so
with 64 percent of the variance where the prompt-side component managed 46.

Two facts from Stage A stand on their own, independently of this design.
Inside the frozen saturation band the scalar still orders videos at 0.6472, so
"saturated" describes the posterior and not the logit; and the strict versus
non-strict distinction on that band is not linearly present in the layer-27
state even with labels, at 0.6253. Any future work choosing the in-band arena
as its target should know that its supervised ceiling there is 0.625, and
should suspect the arena as much as the method — HateClipSeg's negative class
is already known to contain protected-group hostility at a rate of 36 percent.

## Deviations

1. **Fit set narrowed to one corpus.** The preregistration fits on
   ImpliHateVid train and HateMM train. No fresh Whisper transcript exists for
   HateMM train anywhere in the repository, so the frozen judge's input for
   that split cannot be reconstructed and it was dropped. The fit ran on
   ImpliHateVid `train_clean`, 1,283 videos, under the byte-identical
   configuration, disjoint from every evaluation corpus. The narrowing was
   forced by data availability at pre-flight, before any clause was computed.
2. **First extraction pass discarded.** Described above; the preregistered
   pre-flight caught it and the corpora were re-extracted.
3. **Norm control scoping.** Clarified during a synthetic dry run, before any
   real arm existed, to be decisive only when C1 passes.
4. **GPU gate.** Pilot 1's driver ends by writing `SCORING_DONE` to its STATUS
   file and exits without the `DONE` marker this pilot was told to wait for.
   The gate was widened to accept either signal, still requiring an idle GPU,
   and nothing was ever killed or interrupted.

## Decision

The answer-contrast family is closed. No second component, no other layer, no
alternative token pair: A2 fires at every depth that is not degenerate, and the
two clauses that asked for non-z information fail below the scalar.

What carries forward is a sharper statement of the readout bottleneck. It is
not that nobody has yet found the right unsupervised summary of the judge's
states. It is that both places where the answer lives — the state that produces
the logits, and the difference between the states that follow the two answers —
have their dominant variance aligned with the logit contrast. Widening the
readout by any unsupervised summary of a single judge call is now falsified
from both sides, and the constraint that must be relaxed is the single call
itself, or the source of the direction.
