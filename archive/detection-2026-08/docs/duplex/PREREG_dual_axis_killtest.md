# Pre-registration — Dual-axis kill test (offensiveness axis vs joint axis, MHClip-EN test)

**Frozen:** 2026-08-08, before any offensiveness-axis judge call. The exact
offensiveness prompt wording must be committed before the first judge call
(engineering may shape it; scores may not).
**Compute:** one GPU pass, 161 videos × 1 new call. CPU analysis.
**Status:** kill test gating the factorized dual-axis method. It does not
itself claim a method result.

## Phenomenon

The benchmarks mix two constructs. Blind-audited: 69.4% of MHClip-EN union
positives contain no protected-group target. The frozen joint judge ranks
explicit protected-group hostility at AUC 0.983 but Hateful-vs-Offensive at
0.417 (random): one scalar axis conflates the two constructs, and the
ill-posedness result shows one scalar cannot serve two boundaries.

## Proposed mechanism (what this test gates)

Factorize the judgment into two single-call axes with distinct named roles:
a targeted-hate judge (the existing frozen call) and an offensiveness judge
(new call, same evidence, construct = insulting / degrading / vulgar /
aggressive content regardless of any target group). Per-axis label-free
thresholds; deployment composes axes per the task's stated boundary.

## The known risk this test targets

The behavioral law (three confirmations) says the judge is
question-insensitive: rewordings of the same construct correlate ≥ 0.97
with the baseline and never beat an effort placebo. If that law extends to
cross-construct questions, the offensiveness axis is the hate axis renamed,
and the method is dead. The law has never been tested across constructs;
this is the cheapest decisive test.

## Frozen protocol

- Corpus: MHClip-EN test, all 161 videos already scored by the joint judge
  (`results/testruns/mhclip_en/judge_8b/scores.jsonl` = z_hate).
- New call: identical pipeline (16 frames, fresh transcript override logic,
  Qwen3-VL-8B bf16, raw z = logsumexp(Yes)−logsumexp(No)), question swapped
  to the offensiveness construct. Prompt wording frozen in this file's
  appendix before scoring. One call per video; no other changes.
- Strata (fixed before this prereg, from committed audits): blind-coded
  no-protected-target union positives (n=34, `results/annotation_validity`)
  vs shipped Normals (n=112); blind-coded protected-target positives
  (n=15) vs shipped Normals.

## Frozen decision rule

The dual-axis design **SURVIVES** only if both clauses hold:

1. **Separability:** Spearman correlation between z_off and z_hate over the
   161 videos is < 0.95. (0.95–0.97 is a dead zone counted as failure: the
   axes would be too entangled to compose.)
2. **Axis validity:** on the no-protected-target-positive vs Normal
   stratum, AUC(z_off) ≥ AUC(z_hate) + 0.05.

Reported either way, outside the verdict: AUC of max(z_off_rank,
z_hate_rank) composition on the full union task vs the joint baseline
0.785; AUC(z_off) on the protected-target stratum (expected to be LOWER
than z_hate's 0.983 — if z_off matches z_hate there too, the "axis" is not
construct-specific even if clause 1 passes).

## Interpretation boundaries

- Both clauses pass: licenses the full dual-axis preregistration (second
  corpus HateClipSeg insulting-only stratum, per-axis thresholds,
  composition rules). Not itself a performance claim.
- Clause 1 fails: question-insensitivity extends across constructs; the
  readout is evidence-bound, not construct-bound; the dual-axis family dies
  and, with it, prompt-level construct factorization generally.
- Clause 2 fails with clause 1 passing: the axes decorrelate but the new
  axis does not measure offensiveness better than the old one — the extra
  call buys noise, not a construct; dead under the ≤2-call discipline.

## Appendix — frozen offensiveness prompt

Frozen 2026-08-09, committed before the first offensiveness judge call. The
implementation is `scripts/duplex/dual_axis_offensiveness_score.py`, which
imports the system message, the judgment-mode block, the media path, the
transcript-override logic, the pixel budget, and the Yes/No token-id sets from
the same frozen modules the joint judge uses. The only substitution is the
construct: the policy sentence and the rule list. The judgment-mode block, the
question sentence, and the answer instruction are byte-identical to the joint
judge's.

Rule list (replaces `YOUTUBE_RULES`):

```
1. Insulting or demeaning language aimed at any person, group, or thing
2. Degrading, humiliating, or shaming portrayal of anyone
3. Vulgar, obscene, or crude language, gestures, or imagery
4. Aggressive, hostile, threatening, or abusive speech or behaviour
5. Mockery, ridicule, or contemptuous framing of anyone
6. Profanity, name-calling, or personal attacks
7. Gratuitously shocking, disgusting, or distasteful material
```

Prompt skeleton (`{reader_block}` is the frozen `prag` block, unchanged):

```
You are a content moderation analyst. Observe this video's visual frames, title, and transcript.

Title: {title}
Transcript: {transcript}

Based on this platform's offensive-content policy, the following types of content are prohibited:
{rules}

These rules are about offensiveness alone and apply no matter who or what is targeted. Content violates them even when it targets no protected group (such as race, religion, ethnicity, national origin, gender, sexual orientation, or disability), and even when it targets one individual, an institution, or nobody at all.

{reader_block}

Does this video contain content that violates any of the above rules?

Answer "Yes" or "No".
```

The scope sentence after the rule list is part of the construct definition: the
offensiveness axis is defined to be target-agnostic, and the joint judge's rule
list carries "based on protected status" inside nearly every item, so a silent
substitution would leave the protected-group requirement implicit. Stating it
removes the ambiguity rather than adding a new instruction type.
