# Dual-axis kill test: result note

**Date:** 2026-08-09. **Verdict:** **DEAD** under the frozen rule.
**Compute:** one GPU pass, 161 videos × 1 call, 55 s of scoring on an RTX 5090.
**Preregistration:** `docs/duplex/PREREG_dual_axis_killtest.md`
**Machine-readable result:** `results/dual_axis/results.json`

The offensiveness axis separates from the hate axis, but it does not measure
offensiveness better than the hate axis does. Clause 1 passes and clause 2
fails, so the factorized dual-axis method does not proceed.

## What ran

One new judge call per video on the 161 MHClip-EN test videos, against the
frozen joint judge already scored in
`results/testruns/mhclip_en/judge_8b/scores.jsonl`. The new call reuses the
joint judge's system message, prompt skeleton, full-reading judgment block, 16
pre-extracted frames, fresh-transcript override map, pixel budget, Yes/No
token-id sets, and the raw readout z = logsumexp(Yes) − logsumexp(No) at the
final prompt position. The only substitution is the construct: the hate-speech
policy and its nine protected-status rules are replaced by an offensive-content
policy of seven target-agnostic rules, plus one sentence stating that a
violation does not require a protected-group target. That prompt was written
into the preregistration's appendix and committed before the first call, and
was not touched afterwards.

Every frame of every video was decoded before the model loaded, and the run
asserted 161 finished scores before writing its DONE marker. No video was
dropped or retried.

The three strata were rebuilt from the committed blind-coding artifacts and
reproduced the expected sizes exactly: 34 no-protected-target positives, 15
protected-target positives, 112 shipped Normals.

## Frozen clauses

| Clause | Rule | Result |
|---|---|---|
| 1. Separability | Spearman(z_off, z_hate) over 161 videos < 0.95 | **PASS** at 0.935 |
| 2. Axis validity | AUC(z_off) ≥ AUC(z_hate) + 0.05 on no-protected-target positives vs Normals | **FAIL** at +0.041 |

Both clauses were required. Clause 2 fails, so the design is dead.

The Spearman correlation is 0.935, below the 0.95 ceiling but above the 0.90
that would make the two axes comfortably distinct. The Pearson correlation is
0.929. The two axes agree on the sign of the answer for 86% of videos; the
offensiveness axis says Yes for 81 videos and the hate axis for 58.

## Stratum AUCs

| Stratum | n positive | n negative | AUC(z_off) | AUC(z_hate) | Difference |
|---|---:|---:|---:|---:|---:|
| No protected target, vs Normals | 34 | 112 | 0.790 | 0.749 | +0.041 |
| Protected target, vs Normals | 15 | 112 | 0.866 | 0.866 | +0.000 |
| Full union task | 49 | 112 | 0.814 | 0.785 | +0.029 |

The clause-2 gain of +0.041 sits below the frozen floor of +0.05. It is also
not distinguishable from zero: a 5000-resample bootstrap over the stratum puts
the gain in [−0.003, +0.088], and only 34% of resamples reach the floor. The
preregistered threshold was not missed by measurement noise on an otherwise
solid effect; the effect itself is at the edge of nothing.

## Descriptive numbers reported outside the verdict

Composing the two axes by taking, per video, the higher of the two within-corpus
ranks gives AUC 0.801 on the full union task, against the joint judge's 0.785.
The composition is worse than the offensiveness axis used alone, which reaches
0.814. Two calls therefore buy less than one differently-worded call.

On the protected-target stratum the offensiveness axis scores 0.866 and the hate
axis scores 0.866 — a difference of 0.0003. The preregistration predicted the
offensiveness axis would rank protected-group hostility worse than the hate
axis, because an axis that is construct-specific should lose ground on the
construct it was not asked about. It loses nothing. This is the clearest single
result in the test.

One reference in the preregistration needs correcting. The body quotes 0.983 as
the joint judge's AUC on explicit protected-group hostility. That number comes
from the ranking autopsy's narrower seven-video explicit-hostility subset, not
from the fifteen blind-coded protected-target positives that the strata section
actually names. The like-for-like joint-judge number on the stratum used here is
0.866, which matches the blind audit's published figure. The comparison above
uses the matched stratum.

## Interpretation

The behavioral law survives its first cross-construct test. Three earlier
confirmations showed that rewording the same construct moves the judge almost
not at all. This test asked whether changing the construct itself — from
protected-group hate to target-agnostic offensiveness — moves it. The answer is
that it moves the scores a little and moves the discrimination almost not at
all. The judge reads the evidence, not the question.

Clause 1 passing is not a consolation. A correlation of 0.935 means the second
call reorders some videos, and the reordering is real; it is simply not
informative. If the reordering carried construct information, the no-protected-
target stratum would be where it shows up, since that stratum is defined by the
exact property the two questions disagree about. The gain there is +0.041 with a
bootstrap interval spanning zero, while the protected-target stratum shows no
loss at all. A second axis that gains nothing where it should gain and loses
nothing where it should lose is not a second axis.

## Decision

The factorized dual-axis method does not proceed. The follow-on preregistration
it was gating — HateClipSeg insulting-only stratum, per-axis label-free
thresholds, composition rules — is not written.

The wider consequence is the one the preregistration named in advance:
prompt-level construct factorization is dead for this judge. Asking a different
question does not obtain a different measurement. Any future attempt to separate
the offensiveness construct from the hate construct has to intervene somewhere
other than the prompt — on the evidence the judge sees, on the readout, or on
the model — because the prompt has now been shown not to be the place where the
construct lives.
