# Pre-registration — Answer-contrast readout pilot

**Frozen:** 2026-08-09, after Stage A and before any GPU call of Stage B.
**Compute:** Stage A CPU only (done). Stage B: one prefill plus two
single-token forwards per video, Qwen3-VL-8B-Instruct, bf16, single RTX 5090.
**Status:** kill test. Runs once under the rule below; no fallback tuning, no
second fit, no bar renegotiation.

## Claim under test

The scalar readout z = logsumexp(Yes) − logsumexp(No) is a one-dimensional
summary of a state that provably holds more (supervised probe 0.837 where z
scores 0.321, `READOUT_BOTTLENECK_NOTE.md`). Every previous attempt to widen it
read the state at the last *prompt* position, where the dominant direction of
variance is the answer the model was asked for, so unsupervised summaries
rediscovered z (corpus-wide PC1 ↔ z at Spearman 0.992) and band-conditioned
summaries do no better (Stage A below).

This pilot manufactures the contrast on the answer side instead of the question
side. The prompt stays byte-identical to the frozen judge. After the shared
prefill, the model is forced through two continuations that differ in one
token: the first token of "Yes" and the first token of "No". The hidden state
at that appended position is recorded for both. The object of study is the
difference

    Δh(v) = h(v, Yes) − h(v, No),

the internal state of *committing to Yes rather than No against this specific
evidence*.

**Phenomenon.** The answer posterior saturates on 82 percent of videos, so the
scalar contrast is clipped; but saturation is a property of the softmax over
two token sets, not of the residual stream. Two videos that both answer "Yes"
with certainty can still commit to that answer against different evidence.

**Mechanism.** Appending the answer token forces the model to condition on a
commitment it did not choose. The resulting state must reconcile the committed
answer with the evidence in context. Where the evidence supports the commitment
the reconciliation is cheap; where it does not, the state must carry the
tension. Δh(v) is therefore evidence-conditioned even where z is not.

**Prediction.** Δh varies across videos in a direction that is not z, and its
leading residual direction orders videos inside the saturation band where z is
nearly flat.

**Disconfirmation.** If Δh is essentially the same vector for every video
(abort A1), or if its leading residual direction correlates with z above 0.90
(abort A2), or if the pairing between the two arms carries no information
(control C4), the story is false and the pilot is dead. These outcomes are
reachable and are checked before any clause is scored.

## Stage A, already run: the free control

Before manufacturing anything, the cheap rescue of the earlier PCA failure was
run and is reported in full in `ANSWER_CONTRAST_PILOT_NOTE.md`. Inside the
frozen saturation band, PCA fitted on band members only cannot rediscover the
corpus-wide z axis. It also does not recover the construct: the best of the
nine frozen candidates reaches sign-free AUC **0.6348** on the HateClipSeg
in-band arena, below the 0.70 floor, so Stage B proceeds.

**Premise correction, recorded here because it changes how clause C1 must be
read.** The idea as proposed asserted that z is flat inside the band and "the
incumbent scalar cannot order at all" there. That is false. The in-band z range
is +13.000 to +19.500 and z reaches AUC **0.6472** on the C1 arena. The C1 bar
of 0.70 was fixed before this was measured and is **not** changed; it now
carries the stronger meaning of beating both the in-band scalar (0.6472) and
band-conditioned PCA (0.6348), rather than beating chance.

**Second recorded fact, same origin, no bar changed.** Stage A also measured
the supervised ceiling of the C1 arena: an L2 logistic probe on the frozen
judge's layer-27 states, leave-one-out over the 183 in-band videos, reaches AUC
**0.6253**, which is *below* the in-band scalar's 0.6472. With labels and 4096
features, the strict versus non-strict distinction is not linearly present in
the state at the last prompt position. C1 therefore asks the answer-side
contrast to carry construct information that the prompt-side state does not
carry. That is exactly the pilot's hypothesis, so the clause stays as written
and at 0.70; but the clause is now known to be a demanding one rather than a
routine one, and a failure at C1 must be read against this ceiling rather than
as a bare shortfall.

## Frozen extraction protocol

1. **Prompt.** Byte-identical to the frozen judge. `SYSTEM_MESSAGE`,
   `DUPLEX_PROMPT`, `READER_BLOCKS["prag"]`, `YOUTUBE_RULES` (`BILIBILI_RULES`
   for MHClip-ZH, not used here), transcript capped at 300 characters, 16 frames
   from `frames_16`, `min_pixels` 65536 and `max_pixels` 100352. All imported
   from `src/duplex/score_duplex_probe.py` and
   `src/duplex/extract_duplex_readout.py`; nothing is redefined.
2. **Prefill.** One forward over the prompt with the generation prompt
   appended, keeping the key-value cache. The final-position logits give the
   frozen z, which is written out and checked (pre-flight below).
3. **Two arms.** The cache is cropped back to the prefill length between arms,
   so both arms see the identical prefix. Arm Yes appends token
   `tokenizer.encode("Yes", add_special_tokens=False)[0]`; arm No appends
   `tokenizer.encode("No", ...)[0]`. Both tokens are asserted to be members of
   the frozen Yes / No id sets. The tokens are corpus-constant: nothing about
   the video selects them.
4. **Output per video.** Δh of shape (37, 4096) in float16 — the embedding row
   plus the 36 transformer layers, at the appended position — together with z
   and the two arm norms. Individual arms are also stored so that no re-run is
   needed for any descriptive check.

**Cost.** One prefill plus two one-token forwards is approximately 1.1 times
the frozen judge's cost per video, since the prefill dominates.

## Frozen readout

Let L be the layer. On the fit set:

- mean Δh over the fit set, per-dimension residual standard deviation over the
  fit set;
- residual r(v) = Δh(v) − mean Δh of **the corpus being evaluated** (each
  corpus is centred by its own unlabeled mean, which is label-free), divided
  per dimension by the **fit set's** residual standard deviation;
- PC1 = the leading right singular vector of the fit set's standardized
  residual matrix;
- score(v) = ⟨standardized r(v), PC1⟩;
- sign fixed once, on the fit set, so that Spearman(score, z) is positive
  there. No eval corpus participates in the sign choice.

**Layers.** 27 primary. 18 and 36 secondary and reported. The full 37-row sweep
is descriptive only and cannot change any verdict.

**Components.** PC1 only. No second component, no component selection.

## Corpora

**Fit (unlabeled, no labels read at any point):** ImpliHateVid `train_clean`
(1283) and HateMM train (744), pooled after per-corpus centring, 2027 videos.
**Frozen fallback if the GPU budget does not permit both:** fit on HateMM train
only. The fallback is declared here so that the choice cannot be made after
seeing a result.

**Eval:** HateMM test (215), HateClipSeg (394), MHClip-EN test (161), and
ImpliHateVid test (400) as the regression guard.

## Pre-flight, before any clause is scored

- Every eval corpus's recomputed z must reproduce the frozen `scores.jsonl` z:
  Spearman ≥ 0.999 and median absolute difference ≤ 0.05. This is what proves
  the prompt is byte-identical.
- Coverage: every video in every split must produce a finite z and a
  (37, 4096) Δh, or the run aborts rather than silently shrinking a stratum.
- Stratum sizes must reproduce exactly: HateClipSeg in-band 183 = 126 + 57;
  MHClip-EN 34 no-target and 15 protected-target; HateMM valley 70 false
  positives and 83 true positives.

## Aborts, evaluated before any label is touched

- **A1, degeneracy.** If the median over fit-set videos of
  cos(Δh(v), mean Δh) is ≥ 0.99, the manufactured contrast is the same vector
  for everyone and the design is **DEAD**.
- **A2, z-renaming.** If |Spearman(score, z)| ≥ 0.90 on any eval corpus, the
  readout is z under a new name and the design is **DEAD**.

## Controls

- **C4, pairing placebo.** The No-arm states are permuted across videos by a
  fixed derangement (seed 20260808), within corpus, and the entire pipeline is
  re-run on the deranged pairs. The real score must beat the placebo score by
  at least 0.05 AUC on the C1 arena. This tests that the *pairing* of the two
  arms carries the information, not the marginal distribution of either arm.
- **Norm control.** ‖r(v)‖ is scored on the C1 arena by the identical rule. If
  ‖r(v)‖ reaches AUC ≥ score AUC − 0.03 there, the readout is commitment
  magnitude in disguise and the design is **DEAD**.

## Frozen decision rule

**SURVIVES** only if no abort fires and all five clauses hold.

| # | Clause | Arena | Bar |
|---|---|---|---|
| C1 | Primary | HateClipSeg saturation band, strict-hate 126 vs non-strict 57 | AUC ≥ 0.70 |
| C2 | Construct | MHClip-EN, no-protected-target 34 vs protected-target 15 | sign-free AUC ≥ 0.72 |
| C3 | Mid-band | HateMM, 70 valley false positives vs 83 valley true positives | AUC ≥ 0.65, true positives ranked above |
| C4 | Pairing placebo | C1 arena | real AUC ≥ placebo AUC + 0.05 |
| C5 | Regression guard | ImpliHateVid test, score substituted for z under the frozen KDE-valley recipe | macro-F1 ≥ 0.852 |

C2 is scored sign-free, max(A, 1 − A), and this is declared in advance because
the story does not predict the polarity of a targetedness axis relative to a
severity axis. The reference values it must beat are the scalar's own sign-free
0.679 and the supervised ceiling 0.837. C1 and C3 are directional under the
train-fixed sign; reversing them after the fact is forbidden.

Reference values on the C1 arena, all measured before this preregistration was
frozen: in-band z 0.6472, best band-conditioned PCA axis 0.6348.

Descriptive and unable to change the verdict: the layer sweep, per-corpus
Spearman tables, the z-matched variant of the C3 arena, HateClipSeg union
collapse, and MHClip-EN and HateMM whole-corpus AUCs.

## Reporting

One note, `docs/duplex/ANSWER_CONTRAST_PILOT_NOTE.md`, reporting every arm and
both stages regardless of outcome, with no video id and no transcript text.
Machine-readable results in `docs/duplex/reports/answer_contrast_pilot.json`.
</content>
