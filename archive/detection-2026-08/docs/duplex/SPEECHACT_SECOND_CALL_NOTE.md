# Result note — Speech-act second call: DEAD (dual-axis death repeats on the attribution construct)

**Prereg:** `PREREG_speechact_second_call.md` (8ffe65a). **Run:** 2026-08-10,
commit 3eefb06 (scripts); raw outputs in `results/speechact_call/`
(gitignored). **Verdict: clauses 1, 3, 4 FAIL (clause 2 passes for the
wrong reason) → the design dies per the frozen rule.**

## What was tested

Owner-proposed: use the unspent second call to ask the frozen judge
directly whether the hostile content is asserted/endorsed by the video's
author or only quoted/reported/criticized. Same inputs as the frozen hate
call, only the judgment question replaced; z_sa = Yes−No logit contrast.
215 HateMM + 149 MHClip-ZH + 3 EN incidental videos; ~2 minutes GPU.

## Numbers

- **Clause 1 (ZH separation): FAIL.** AUROC 0.6966 (CI 0.561–0.821),
  bar 0.75.
- **Clause 2 (HateMM separation): PASS but empty.** AUROC 0.9031 — carried
  entirely by z, as clause 3 shows.
- **Clause 3 (incremental over z): FAIL.** Residualized AUROC ZH 0.3584
  (inverted), HateMM 0.4978 (chance).
- **Clause 4 (not z renamed): FAIL.** In-pool |Spearman(z_sa, z)| ZH
  0.9360, HateMM 0.9182 — the dual-axis death fingerprint (0.935)
  reproduced almost exactly. Full-corpus Spearman: ZH 0.9675, HateMM
  0.9827.
- The model answers "the author endorses it" for quoted/reported hate:
  mentioned-stratum mean z_sa is POSITIVE in both corpora (ZH +1.07,
  HateMM +3.55); the two EN counter-speech false positives the question
  was meant to flip get z_sa +7.00 and +4.50, tracking z.
- Saturation is mild (entropy 0.09–0.14 bits; graded scores) — the answer
  is not degenerate, it just measures the same thing z measures.
- One resolved ambiguity, declared: the frozen question's trailing "Answer
  Yes or No." replaced the skeleton's answer instruction so no
  duplication; everything above the question byte-identical to
  DUPLEX_PROMPT (programmatically asserted).

## Interpretation (bounded by the prereg)

1. Verbal access to speech act on video evidence is dead: asked directly,
   the judge's answer is the hate scalar renamed (correlations 0.92–0.98),
   and it affirms author endorsement for quoted hate.
2. Combined with the illocution transfer audit (same day): the model
   encodes the assert/attribute distinction at ceiling on text evidence,
   yet on video evidence BOTH access channels — a text-fitted linear
   probe and the model's own verbalized judgment — return nothing beyond
   z. The conflation is not a readout artifact; it is how the model
   processes video evidence end-to-end.
3. The speech-act program is now closed from both sides under current
   constraints. Per the prereg's interpretation boundary, the only
   remaining door is a small human speech-act annotation set (non-hate
   labels: asserted vs mentioned, ~50–100 videos) to fit/select a
   video-manifold readout directly — OWNER RULING PENDING. Honest risk,
   recorded in advance: today's ZH inversion and the verbal conflation
   are consistent with the video-state speech-act signal being weak; the
   annotation door may buy a dead probe. Its pilot is cheap (owner
   annotation time ~1–2 hours; CPU probe fit; sealed evaluation reusable
   as-is).
