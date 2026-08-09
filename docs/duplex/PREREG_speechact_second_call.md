# Pre-registration — Speech-act second call (owner-proposed; attribution query on video evidence)

**Frozen:** 2026-08-10, before any new call is scored.
**Compute:** one RTX 5090; 215 + 149 = 364 single forwards (≈ minutes).
**Status:** kill test. Proposed by the owner: instead of probing hidden
states or annotating videos, ASK the frozen judge, in a second call with a
distinct role, whether the hostile content is asserted by the author or
only quoted/reported/criticized. This uses the unspent second slot of the
≤2-calls-per-video budget with a genuinely distinct named role
(hate judge / speech-act reader).

## Phenomenon (unchanged anchor)

The hate call scores surface hostile content regardless of speech act
(MHClip-ZH keyword-title benign videos +8 logits; 11/70 HateMM valley FPs
are blind-coded quoted/reported hate; MHClip-EN counter-speech cases).
The illocution transfer audit (ILLOCUTION_TRANSFER_NOTE.md) established
that the model's representation encodes speech act at ceiling on
text evidence (LOFO AUROC 1.0000, typography-invariant) while a
text-fitted linear readout does not transfer to video states. Untested:
whether the model can VERBALIZE the distinction on video evidence when
asked directly — its answer emerges from full-depth computation, not from
a text-fitted direction, so the transfer death does not cover it.

## Adjacency disclosures (confronted in advance)

1. **Dual-axis death** ("prompt-level construct factorization is dead;
   asking a different question does not obtain a different measurement").
   That test swapped hate for offensiveness — both severity constructs on
   the same content-badness axis; the swapped answer correlated 0.935 and
   kept the protected-target ranking bit-identical. The present question
   asks about ATTRIBUTION (who commits to the statement), not severity.
   The frozen risk that this distinction is illusory is exactly clause 4
   below: if the speech-act answer is z renamed, the design dies the same
   death and the note will say so.
2. **PP-v2 / MADP target×stance deaths.** "Stance" there = hostile stance
   toward a target (severity family), not attribution; and their binding
   lesson (no boolean AND, no product; any factored method must beat the
   joint single call) binds the LATER combination stage, not this
   measurement pilot. No combination is performed here.
3. **Observe-then-judge deaths** routed evidence through a text
   description into a second judgment of the SAME construct. Here both
   calls see identical full evidence and measure DIFFERENT constructs.

## Frozen protocol

- Corpora: HateMM test (215) and MHClip-ZH test (149), full splits.
- Inputs identical to the frozen c2 hate call: same 16 frames, same
  fresh-transcript overrides, same title, same pixel budget, same rules
  block and prompt skeleton per corpus convention. Implementation follows
  the dual-axis precedent (scripts/duplex/dual_axis_offensiveness_score.py):
  only the judgment question is replaced.
- Question (frozen verbatim, English for both corpora, matching the
  skeleton's language): "Consider the hostile, offensive, or negative
  statements that appear in this video (in its speech, its title, or its
  on-screen text). Does the video's author or narrator assert or endorse
  any of these statements themselves, rather than only quoting,
  reporting, criticizing, or mocking them as someone else's words?
  Answer Yes or No."
- Score: z_sa = logsumexp(Yes variants) − logsumexp(No variants), same
  token machinery as z. One forward per video, no generation, no
  sampling. High z_sa = asserted.
- Sealed strata (same as the illocution audit, labels used only in
  evaluation): ZH — 31 mentioned (Normal with `<em class="keyword">`
  title markup) vs 32 asserted (Hateful∪Offensive with markup); HateMM —
  11 mentioned (blind code Q FPs) vs 83 asserted (Hate-labeled, z >
  −2.3455).
- Residualization and bootstrap: OLS of z_sa on z within each stratum
  pool, residual AUROC; 10,000 resamples, seed 20260810 (same convention
  as the illocution audit).

## Frozen decision rule

The design **SURVIVES** only if all four clauses hold:

1. **ZH separation:** AUROC of (−z_sa) for mentioned-vs-asserted ≥ 0.75.
2. **HateMM separation:** same statistic ≥ 0.70 with bootstrap 95% lower
   bound > 0.55.
3. **Incremental over z:** residualized AUROC ≥ 0.65 on BOTH strata.
4. **Not z renamed:** |Spearman(z_sa, z)| ≤ 0.80 within each sealed
   stratum pool (the dual-axis death fingerprint was 0.935 corpus-wide).

Reported descriptively either way: full-corpus Spearman(z_sa, z) for both
corpora; z_sa distributions per stratum and per class; the three EN
incidental cases (EN-FN-02, EN-FP-07, EN-FP-16) scored with the same
call; answer-entropy occupancy of z_sa (saturation check).

## Interpretation boundaries

- Pass: licenses a separate method preregistration for combining z and
  z_sa (graded, monotone, no products/ANDs, must beat the joint single
  call at the same operating point per the binding lesson, ImpliHateVid
  regression guard included). This pilot makes no performance claim.
- Clauses 1–2 fail: the model cannot report speech act on video evidence
  even when asked directly. Combined with the illocution audit this
  closes the speech-act program under current constraints from BOTH
  sides (probe access dead, verbal access dead); the only remaining door
  reverts to human speech-act annotation, for owner ruling.
- Clause 3 or 4 fails: the answer is the hate scalar renamed — the
  dual-axis death repeats on the attribution construct; same closure as
  above, recorded as such.
- Two calls per video, distinct named roles, within budget; no input is
  modified; no other model; no labels touch anything except the sealed
  evaluation.
