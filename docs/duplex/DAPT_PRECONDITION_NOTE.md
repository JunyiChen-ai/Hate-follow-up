# Result note — DAPT precondition probe: FAIL (familiarity account dead; DAPT not run)

**Prereg:** `PREREG_dapt_precondition.md` (c5d36fd). **Run:** 2026-08-10,
script committed 13ca2bc; raw outputs in `results/dapt_precondition/`
(gitignored). **Verdict: Link 2 FAILS; the DAPT family is closed before
training, as the prereg required.**

## Numbers

- HateMM (181 analyzed after the <20-token exclusion, 34 excluded):
  per-video transcript NLL separates mid-band (|z|<13, n=88) from poles
  (n=93) at **AUC 0.5028, p 0.949** — indistinguishable. Bar was 0.60.
  Spearman(NLL, |z|) −0.03 (p 0.68).
- ImpliHateVid control (389 analyzed): **AUC 0.6631, p 3.0e-08** — the
  guard fired in the worst direction: the familiarity-congestion effect
  exists cleanly, but only on the corpus where the label-free operating
  point already works (0.8823). Generic property, not an explanation of
  HateMM's failure.
- Length confound checked: HateMM Spearman(NLL, length) −0.43 yet the
  band effect is exactly zero — nothing masked. ImpliHateVid's control
  effect is not a length artifact (−0.19 < band effect).
- The 11 blind-coded quoted-hate FPs sit at median NLL percentile 39.2 —
  NOT in the hard-to-parse tail. Their transcripts are parsed as easily
  as average; consistent with the speech-act findings: the conflation is
  a processing property, not a comprehension failure.

## Interpretation (bounded by the prereg)

1. HateMM's mid-band congestion is NOT a familiarity artifact. The
   unsaturated middle is exactly as predictable as the poles: the
   model's uncertainty there reflects construct-boundary content
   (generic hostility, mentions) — not unparsed evidence.
2. Corpus-conditioned continued pretraining (DAPT / next-token on the
   target corpus) therefore has no identified lever on the congestion
   and is not run — a prevented-in-advance death, recorded per protocol.
3. Substitute-signal family 5 (predict withheld input parts) joins the
   dead list for this problem. Of the seven label-free substitute
   signals in the literature map (LABELFREE_PHILOSOPHY_MAP.md), six are
   now dead, excluded, or vetoed here; the only untried survivor is
   within-video cross-modal agreement/disagreement — which must first
   survive a reducibility check against the dead families (its naive
   forms reduce to score fusion [PP-v4, dead], agreement pseudo-labels
   [confidence family, dead], or consistency-under-views [augmentation
   family, excluded]).
