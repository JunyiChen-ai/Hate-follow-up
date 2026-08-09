# Result note — Illocution instrument-to-natural-transfer audit: DEAD

**Prereg:** `PREREG_illocution_transfer_audit.md` (f91daf3, frozen before any
instrument sentence existed). **Run:** 2026-08-10, commit 7a11688 (banks +
scripts); raw outputs in `results/illocution/` (gitignored). **Verdict:
clauses 1–2 PASS, clauses 3–4 FAIL → the speech-act instrument family is
dead per the frozen rule.**

## What was tested

Whether a speech-act (author-committed vs attributed-to-other) direction,
fitted on 700 researcher-authored benign sentences (350 EN + 350 ZH,
person-swap minimal pairs, quote marks crossed independently, 30 carrier
families per language, zero hostile content — instrument z range −18.25 to
−8.25 confirms nothing hostile reached the judge), transfers from the
judge's text-evidence hidden states to its video-evidence hidden states.

## Numbers

- **Clause 1 (instrument validity): PASS, ceiling.** Leave-family-out AUROC
  = 1.0000 in all 25 grid cells (layers {20,24,27,30,33} × ranks
  {1,2,4,8,16}); selection degenerated to the preregistered tiebreak
  (layer 27, rank 1). EN-only, ZH-only, quoted-only, unquoted-only all
  1.0000. The reject cell behaves sensibly (attribute_reject sits between
  poles: assert −5.03 < reject +2.98 < attribute +5.09).
- **Clause 2 (format invariance): PASS.** Quoted-only 1.0000, unquoted-only
  1.0000, quote-reversal gap 0.0000.
- **Clause 3 (ZH natural transfer): FAIL, sign inverted.** 31 mentioned
  (Normal-with-keyword-markup) vs 32 asserted (Hateful∪Offensive-with-
  markup): residualized AUROC 0.3337, 95% CI [0.2329, 0.4587], raw 0.3679.
  Asserted videos score MORE attributed than mentioned ones.
- **Clause 4 (HateMM natural transfer): FAIL, chance.** 11 blind-coded
  quoted/reported FPs vs 83 above-threshold Hate videos: residualized AUROC
  0.4830, CI [0.3308, 0.6550], raw 0.5214.
- Exploratory full-grid sweep (permitted once): best ZH 0.6179 (L20), best
  HateMM 0.6035 (L33) — different layers, both below the 0.70 floor;
  unstable residue, not signal.
- Deviations from prereg, all declared in `results/illocution/report.json`
  and the run report: the cited normalization convention contains
  standardization only (no RMSNorm step exists in
  `readout_bottleneck_killtest.py` — the prereg phrase had no referent);
  per-corpus vs instrument standardization both computed (same conclusion:
  ZH 0.3185, HateMM 0.4973); rank-k whitening construction fixed before
  results; bootstrap re-runs OLS inside resamples.

## Interpretation (bounded by the prereg)

1. The representation **has** the concept: speech act is perfectly linearly
   decodable, in both languages, invariant to typography, from the judge's
   hidden states when the evidence is text. This is a clean positive fact.
2. The direction that carries it on text-evidence states carries nothing
   (HateMM) or the wrong sign (ZH) on video-evidence states. **Text-derived
   activation geometry does not transfer to video-evidence states in this
   judge.** This is the same fracture the spec-displacement pilot hit
   (construct-orthogonal, cos ≈ 0.03), now demonstrated at transfer-test
   strength with a ceiling-quality instrument.
3. Kill scope per the frozen interpretation boundary: round-3 candidates
   1 (illocution gate), 2 (illocution×hostility tensor), 4 (speech-act
   causal head editor — its localization pairs are the same instrument),
   10 (SAE leak editor — its factorial cells are the same instrument), and
   candidate 8's text-based feature selector. Candidate 3 (transport field
   trained on instrument pairs) is pre-doomed by the same bridge. Candidates
   5, 6, 7 (policy-text subspaces applied to video states) inherit the
   broken text→video bridge and are heavily discounted, matching the
   adversarial review's advance prediction for 5.
4. What survives conceptually: only mechanisms that manufacture their
   contrast **from video-evidence states themselves** (e.g., SAE trained on
   video activations with a video-side causal selection principle,
   candidate 9's fingerprint bridge) — plus the standing fact that no legal
   label-free selection principle currently exists on that side.
