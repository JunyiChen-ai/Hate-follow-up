# Result note — Temporal attribution pilot: DEAD

**Prereg:** `PREREG_temporal_attribution_pilot.md` (26ccde5). **Run:**
2026-08-12, script committed 403637c; raw outputs in
`results/temporal_attribution_pilot/` (gitignored). **Verdict: both
variants fail both clauses → the attribution-localization family dies
per the frozen rule.**

## Numbers

- Cohort: 316 sparse+timestamped HateClipSeg videos; 159 scored (157
  excluded because the timestamped ASR chunks reconstruct the
  un-collapsed transcript while the frozen prompt uses the
  collapse_repeats gate output — prereg forbids patching; kept vs
  excluded are similar on z, duration, tokens).
- Macro AUC, offensive-union vs normal-only segments: **grad×input
  0.5252**, **late-layer attention 0.4819**, token-density control
  0.5182. Bars were 0.65 and control+0.05. Transcript-only and
  frame-only channels: same picture (0.49–0.53). Hateful-strict:
  0.5439/0.5021 vs control 0.5255.
- Attribution − control = +0.007: the curve is token density in
  disguise (the prereg's named failure branch).
- Attention variant is dominated by the sequence-start sink + trailing
  question (positional quartiles 0.67/0.00/0.02/0.31) — the published
  failure mode reproduced.
- Rigor notes: gradient pass reproduces the frozen judge's z bitwise on
  all 159 videos; prompt built through the frozen imports; two-pass
  execution of the single call (attention pass + checkpointed gradient
  pass) forced by 32 GiB VRAM, documented.

## Interpretation (bounded by the prereg)

1. The evidence-localization premise is false FOR THIS JUDGE: its
   video-level decision mass does not concentrate on the annotated
   offensive segments. Consistent with the corpus facts: these videos
   sit deep in saturation (mean z 11.5), and the judge integrates
   register/topic globally — annotators mark the explicit moments, the
   judge fires on the whole discourse.
2. Kill scope: parameter-free attribution readouts (answer-gradient ×
   input; uniform late-layer attention) from the frozen single call.
   Not covered: mechanisms that never read attribution — e.g., verbal
   segment-indexed readouts (the model names which numbered chunk
   violates), which are a different family and go through the goal
   loop's next idea-discovery round.
3. Goal-loop consequence: re-enter idea discovery for the next
   label-free localization direction, with this death folded into the
   constraint set.
