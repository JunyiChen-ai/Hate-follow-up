# Pre-registration — Illocution instrument-to-natural-transfer audit (speech-act subspace, family-level kill test)

**Frozen:** 2026-08-10, before any instrument sentence is written and before
any natural-stratum file is opened for this analysis.
**Compute:** one RTX 5090 for a single short activation-collection run
(≈400–800 text-only forwards, well under 1 GPU-hour); all analysis CPU.
**Status:** kill test gating the entire speech-act mechanism family
(idea-round-3 candidates 1, 2, 4, 10 and the selector of 8). Successor
constraints: owner vetoes of 2026-08-09/10 (no preprocessing, no prompt
engineering, no cross-model supervision) are binding on the design below.

## Phenomenon

The judge scores surface hostile content regardless of speech act. Three
corpora, one disease: MHClip-ZH benign videos whose harvest titles embed the
offensive query keyword score median z +2.25 vs −6.50 for plain benign
videos; 11 of 70 HateMM valley false positives are quoted/reported/countered
protected hate (blind-coded, code Q); MHClip-EN counter-speech cases exist
but have no blind coding. Moderation targets ASSERTED hostility; web video is
saturated with MENTIONED hostility.

## Claim under test

The frozen judge's hidden states carry a speech-act (author-committed vs
attributed-to-other) direction that (a) is learnable from a researcher-
authored text instrument containing NO hostile content, and (b) transfers
zero-shot to natural videos, carrying information about the mention-type
false positives that the scalar z does not carry. If (a) holds but (b)
fails, researcher-authored text instruments cannot locate transferable video
speech-act geometry and the whole family dies — not just this variant.

## Instrument (frozen design; sentences to be authored AFTER this file commits, BEFORE any scoring)

- Two banks, English and Chinese, authored by hand, 200–400 items each.
- **No hostile content anywhere**: propositions are benign or mildly
  negative everyday claims (e.g., opinions about food, weather, products,
  fictional events). No slurs, no protected groups, no violence. Hostility
  detection stays entirely with the existing z; the instrument measures
  speech act only.
- Primary contrast: **person-swap minimal pairs** — identical clause
  structure and reporting verb, only the committing subject differs:
  "I say/claim/believe/maintain that P" (ASSERT) vs "She says/claims/
  believes/maintains that P" / "The report states that P" (ATTRIBUTE).
  This controls the reporting-verb/syntax confound by construction.
- Secondary cells (for invariance checks, same P inventory): direct quote
  format with and without quotation marks, crossed independently of speech
  act (asserted-with-quotes, attributed-without-quotes both present);
  attributed-and-rejected ("She claims P, which is wrong") as a
  descriptive cell.
- ≥8 carrier families per language (distinct proposition topic × syntactic
  frame combinations); family membership recorded per item; all model
  selection uses leave-family-out splits.
- Bank committed to the repo with SHA256 recorded in the results note
  before the collection run starts.

## Activation collection (frozen)

- Same prompt skeleton as the frozen judge (DUPLEX_PROMPT of
  src/duplex/score_duplex_probe.py path as used by
  src/duplex/extract_duplex_readout.py), instrument sentence in the
  transcript slot, empty title, and **16 identical mid-gray frames**
  (constant across ALL items — a constant cannot carry contrast
  information). Model Qwen3-VL-8B-Instruct bf16, HF path, final-prompt-token
  hidden states, all 37 layers, fp16, same dump format as testruns.
- One forward per item. No sampling, no generation.

## Readout fitting (frozen; instrument data only — natural data stays sealed)

- Normalization convention identical to the readout-bottleneck analysis
  (scripts/duplex/readout_bottleneck_killtest.py): same RMSNorm application
  and standardization.
- Grid, fixed in advance: layer rows {20, 24, 27, 30, 33} × subspace rank
  {1, 2, 4, 8, 16}. Rank-k readout = top-k LDA/mean-difference directions
  (rank 1) or class-covariance-whitened principal directions of the
  assert-vs-attribute contrast (k > 1); commitment score = signed distance
  along the fitted discriminant (logistic on the k-dim projection, fit on
  instrument only).
- Selection of (layer, rank): maximize leave-family-out AUROC on the
  primary contrast, pooling EN+ZH banks. Ties → smaller rank, then layer 27.
- The selected readout is then FROZEN before any natural-video file is
  touched.

## Sealed natural strata (evaluation only, one pass, defined from existing on-disk artifacts)

1. **ZH (label-free identifiable):** among the 63 MHClip-ZH test videos
   whose stored harvest title contains `<em class="keyword">` markup
   (results/ranking_autopsy/zh/items.json): Hateful∪Offensive (n=32,
   ASSERTED) vs Normal (n=31, MENTIONED). Both sides carry the same
   surface-keyword property; they differ in speech act.
2. **HateMM (blind-coded):** code-Q valley false positives (n=11,
   MENTIONED; results/hatemm_fp_audit/coding.tsv) vs test-split Hate
   videos with z above the frozen corrected valley threshold −2.3455
   (ASSERTED). Same high-z region; differ in speech act.
3. **EN (descriptive only, not load-bearing):** the three score-aware
   incidental cases (EN-FN-02, EN-FP-07, EN-FP-16) — reported qualitatively;
   no clause depends on them because no blind EN stratum exists.

Residualization, frozen: within each stratum's pooled videos, regress the
commitment score on z (ordinary least squares, intercept included), take
residuals, compute AUROC of residuals for MENTIONED vs ASSERTED (sign such
that MENTIONED-higher counts as the predicted direction: attributed pole of
the instrument). Bootstrap 95% CI, 10,000 resamples, seed 20260810.

## Frozen decision rule

The family **SURVIVES** only if all four clauses hold:

1. **Instrument validity:** leave-family-out AUROC ≥ 0.90 on the primary
   assert-vs-attribute contrast at the selected (layer, rank).
2. **Format invariance:** AUROC ≥ 0.85 separately within quoted-only and
   unquoted-only instrument subsets, and the quote-reversal gap ≤ 0.05.
3. **Natural transfer, ZH:** residualized AUROC ≥ 0.70 with bootstrap 95%
   lower bound > 0.55 and the predicted sign.
4. **Natural transfer, HateMM:** residualized AUROC ≥ 0.70 with bootstrap
   95% lower bound > 0.55 and the predicted sign.

**DIES** if any clause fails for every (layer, rank) cell of the frozen
grid (clauses 3–4 evaluated only at the selected cell; a failed selected
cell may be followed by AT MOST one full-grid descriptive sweep, reported
as exploratory, never promoted).

Reported descriptively either way: per-layer per-rank instrument AUROC
table; raw (non-residualized) natural AUROCs; commitment-score distributions
for all strata; correlation of commitment score with z per corpus; the EN
incidental cases; instrument-bank SHA256.

## Interpretation boundaries

- Survive: licenses the gate-design preregistration (separate document,
  separate decision rule with macro-F1 targets vs TRIAGE 0.808). This audit
  itself makes no performance claim.
- Clause 1 fails: the representation does not linearly encode speech act
  even on clean text — the family dies at the root.
- Clause 2 fails: the readout reads typography, not illocution — dead.
- Clause 3 or 4 fails: text-instrument geometry does not transfer to video
  states — the instrument approach dies for ALL round-3 speech-act
  candidates (1, 2, 4, 10, and 8's text-selected selector), and any revival
  must manufacture the contrast from video inputs, which currently has no
  legal construction.
- No corpus input is modified; no prompt is changed; no other model is
  queried; labels and blind codes are used only inside the sealed
  evaluation, as everywhere in this project.
