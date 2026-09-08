# Duplex readout probe: postmortem

Sources: `docs/duplex/PREREG_duplex_readout.md` (frozen 2026-08-05 23:47, before any
extraction), `docs/duplex/reports/readout_probe_8b_train.json`,
`docs/duplex/reports/rawz_detector_train.json`, and the 1283 scored rows in
`results/duplex_readout/ImpliHateVid/scores.jsonl`. Every figure below was recomputed
from those files.

## Verdict

**KILL.** Two clauses of the pre-registered kill rule fire. P1 fails at the primary
layer and at every one of the 37 layers, and P2 fails in the direction the
pre-registration named as disconfirming. The perception–report gap is dead for this
operationalization, and the method is not to be built.

| # | Prediction | Statistic | Bar | Verdict |
|---|---|---|---|---|
| P1 | probe separates dismissed IM from dismissed NH where `z` cannot | probe 0.8520 (layer 19), raw `z` 0.8751, Δ −0.0231; best layer 21 at 0.8938 | ≥ 0.65 and ≥ z + 0.10 | **fail** |
| P2 | gain asymmetry, implicit over explicit | Δ(IM) −0.0073, Δ(EX) +0.0030, difference −0.0102 | Δ(IM) ≥ Δ(EX) + 0.03 | **fail** |
| P3 | the direction is a hatefulness direction | probe AUC(EX vs NH) 0.9548 | ≥ 0.90 | pass |
| P4 | not anchor-set structure | real 0.8520 against permuted mean 0.5009, p95 0.8054, empirical p 0.01 | above the 95th percentile | pass |
| P5a | direction stability | mean cosine 0.9661, min 0.9247 over 100 resamples | ≥ 0.8 | pass |
| P5b | no resolution shortcut | inside ≤720p: probe 0.8554, raw `z` 0.8777, Δ −0.0222 (61 IM, 550 NH) | P1 passes there | fail |
| P6 | internals recognize over-flagged benign content | probe 0.8117 against raw `z` 0.7992, Δ +0.0126 | ≥ z + 0.10 | fail |

The diagnostic abort never triggered: anchor purity was 0.984 on the top tail and
1.000 on the bottom, against a 0.85 floor. The robustness points agree with the
primary arm. At k = 0.02 the P1 probe reaches 0.8287 and at k = 0.10 it reaches
0.8572, both below the same 0.8751 raw-`z` baseline, with direction cosines of 0.978
and 0.992 against the primary direction. P5b fails for the same reason P1 fails
rather than for a resolution reason, since the low-resolution stratum reproduces the
full-sample ordering almost exactly. The permutation placebo deserves one caveat:
when the placebo is granted the same freedom over layers that the sweep enjoys, its
95th percentile rises to 0.9049, above the observed 0.8938 peak. The pre-registered
arm fixes the primary layer and passes, so P4 stands, but no layer-shopped version of
this result would survive its own placebo.

## The fingerprint: neither pre-registered failure mode

The pre-registration named two diagnostic fingerprints, and the data matches neither.

The knowledge-absence fingerprint would have shown the dismissed-IM against
dismissed-NH curve flat at 0.5 across every layer. It is not flat. The probe reads
the contrast at 0.852 at the primary layer and 0.894 at layer 21, with a maximum
deviation from chance of 0.394. Internal representations do separate verbally
dismissed implicit hate from verbally dismissed normal content. The model is not
ignorant of the 78 videos it dismisses.

The dissociation fingerprint would have shown the verbalized channel unable to make
that same distinction. It makes it, and slightly better: the raw logit difference
reaches 0.8751 on the identical 78-versus-552 stratum, which is 0.0231 above the
probe at the primary layer. Two channels that rank the same stratum the same way are
not dissociated. P2 tells the same story from the other side. The probe buys
−0.0073 on IM against NH and +0.0030 on EX against NH, a difference of −0.0102
against a required +0.03. What the sweep shows is a readout that recovers the
verbalized signal, plateaus at it, and never exceeds it by a margin that would mean
anything.

## Where the phenomenon went

The hypothesis was built on an instrument that no longer exists. The kill-test
postmortem established that its own P1 pass came from censoring: 622 of 1155 videos
had a divergence of exactly zero because a renormalized P(Yes) was clipped at
`EPS = 1e-4` before the logit was taken, with 75.9% of NH pinned at a bound. Under
that readout, verbally dismissed videos are genuinely indistinguishable from one
another, because the instrument threw the distinction away. Dismissal looked like a
wall, and a wall invites the question of what lies behind it.

The pre-registration mandated the fix, retiring the clipped readout for the raw
unclipped logit difference. The fix dissolved the phenomenon. At raw precision the
dismissed stratum is not a wall but a range: `z` runs from −3.0 to −21.75 with an
interquartile range of [−17.69, −9.75], and the group medians inside it are −6.5 for
EX, −8.0 for IM, and −15.75 for NH. Verbal dismissal is graded, and the grade already
encodes the answer. Only 0.74% of the IM-versus-NH pairs in that stratum are tied at
bf16 logit resolution, so the baseline is doing real ranking rather than coasting on
the tie rule.

The lesson generalizes past this probe. A hypothesis about a hidden channel should be
stated against the best available reading of the visible one. Here the perception–report
gap was, in retrospect, a gap between perception and a lossy measurement of the report.
The pre-registration was correct to demand the readout fix first and correct to run
the raw-`z` baseline inside every arm; that discipline is what made the artifact
visible instead of letting it survive as a mechanism.

## What survives

**Single-call raw `z` is a strong ranker and a usable detector.** Task-A analysis in
`docs/duplex/reports/rawz_detector_train.json` scores one forward pass with one frozen
prompt against the 649 hateful and 634 normal videos of `train_clean`. Full-sample AUC
is 0.9334, with a bootstrap 95% interval of [0.9190, 0.9460]; the subgroup values are
0.9150 for IM against NH and 0.9518 for EX against NH. Thresholded label-free at the
density valley of the `z` histogram, at −2.90, the judge reaches F1 0.8488 on the
hateful class, accuracy 0.8504, macro-F1 0.8503, precision 0.8680 and recall 0.8305.
The model's own boundary at `z = 0` is worse, at F1 0.8215, because it trades 0.065 of
recall for 0.018 of precision. A labeled oracle threshold at −7.125, which is not
label-free and is quoted only as a ceiling, reaches F1 0.8661. The label-free rule
therefore sits 0.017 F1 below the best achievable threshold on the same scores. These
figures are on `train_clean`, the test split has not been scored, and the published
supervised result on the ImpliHateVid test split (IARE, F1 91.75, SIGIR 2026) is a
different split under full supervision, so nothing here licenses a comparison.

**The IM bimodality is a property of the input, not of the old instrument.** Across
the readout fix, which changed both the frame budget and the score scale and returned
the 128 previously lost HD videos to the sample, the IM commitment split barely moved:
0.611 asserted and 0.241 dismissed now, against 0.598 and 0.239 before. Roughly a
quarter of implicit-hate videos read as benign to this model, and that fraction is
stable under a change of instrument.

**The residual error budget is now located.** At the label-free threshold, all 110
false negatives fall inside the frozen dismissed stratum, and they are exactly its
occupants: 78 IM and 32 EX, with no interior misses at all. The density valley lands
at −2.90 and the pre-registered dismissal bound at −2.944, so the label-free
calibration rediscovers the commitment boundary without being told about it. The 24.1%
of IM that the mouth dismisses is therefore the whole implicit half of the error
budget, and both channels agree those videos look benign. No readout change reaches
them. A successor must add information or add processing.

The other half is over-flagging. The 82 false positives are all NH, 48 of them
verbally asserted and 34 in the interior, spanning `z` from −2.75 to 14.5 with a
median of 3.25. Of the 74 that fall inside the Phase-0 blind-tagged pool, 48 carry at
least one surface marker, a rate of 0.649 against 0.165 in the matched controls;
identity-group mention appears in 31, aggressive or violent language in 25, explicit
discussion of hate or racism in 13, and profanity in 11. That is topic-versus-stance
confusion, not a readout defect either.

**Linear probing buys nothing over the mouth at this position.** This is the real
negative result and it is worth stating plainly. At the judgment token, with this
prompt, a label-free difference-of-means direction estimated from high-purity anchors
recovers the verbalized signal and stops there: 0.9548 against 0.9518 on EX versus NH,
0.9077 against 0.9150 on IM versus NH, 0.8520 against 0.8751 on the dismissed stratum.
The direction is real, stable, and not an anchor artifact, which P3, P4 and P5a
establish. It is simply redundant. The claim is bounded by its conditions: one token
position, one prompt, one linear family, one model. A nonlinear probe, an earlier
position, or a pooled-over-frames representation could behave differently, and none of
those were tested.

## Two mechanisms dead, seventeen hours spent

Two pre-registered mechanisms are now closed. Instructional reading-separation died in
the kill-test: telling the model to read only surface meaning does not suspend the
pragmatic inference, and a thoroughness-matched placebo reproduced the divergence
statistic to within 0.006 AUC on both scales. Perception–report dissociation dies
here: the internals and the mouth carry the same ranking, so there is no suppressed
percept to recover.

Both were killed for the cost of one working day. The kill-test pre-registration was
frozen and committed at 06:31 on 2026-08-05 and the readout verdict landed at 00:07 on
2026-08-06, seventeen and a half hours later, covering two pre-registrations, two full
scoring runs over 1283 videos, a hidden-state extraction across 37 layers, and two
postmortems. That is the discipline paying for itself. A mechanism that cannot survive
a pre-registered kill-test costs a day to discard here, and would have cost a quarter
to discard after it had been built into a method, written up, and reviewed.

What a successor inherits is a well-characterized single-call baseline, a located
error budget, and one fewer place to look for a free lunch.
