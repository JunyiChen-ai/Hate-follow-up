# Result note — Masked parallel isolation pilot: SURVIVES (direction 3)

**Prereg:** `PREREG_masked_parallel_isolation_pilot.md` (7928b9b).
**Novelty:** checked NOVEL-with-caveats BEFORE the prereg (IDEA_REPORT.md
f0deb1f; mask primitive conceded to SingGuard 2606.22873 / InvariRank
2604.27599 / T3S 2511.17945; the claim is the contamination finding +
isolation as remedy + localization output). **Run:** 2026-08-18, script
committed aa403cb; raw outputs in `results/masked_parallel_isolation/`
(gitignored). **Verdict: both frozen clauses pass → direction 3
survives; method-stage prereg licensed.**

## Clause 1 — fidelity (PASS, no debugging pass used)

- Spearman(masked packed z, sequential isolated z) = **0.99889** over
  all 2281 chunks (bar ≥ 0.99); max |Δz| 1.25, mean 0.172 —
  bf16 kernel numerics from the changed sequence shape (a freshly
  recomputed sequential score shows the same residual), not packing.
- AUC deltas vs the sequential reference: within-video pooled −0.0015,
  cross-video +0.0002 (bar ±0.01).
- Prompt-identity assertion held for all 2281 chunks
  (prefix 160 tokens; concat(prefix, branch) == sequential prompt ids).
  Fully-causal 4D mask reproduces the no-mask forward at max |Δ logit|
  = 0.0 — the mask plumbing is exact.

So the packed masked forward IS the N isolated calls, at one-pass cost.

## Clause 2 — counterfactual gap (PASS, narrowly)

| Arm | within-video macro | within pooled | cross-video | mean z |
|---|---|---|---|---|
| M (masked) | **0.6197** | 0.6225 | 0.7202 | −7.91 |
| C (full causal, same tokens/positions) | **0.5622** | 0.6281 | 0.8502 | −0.13 |
| sequential reference | 0.6238 | 0.6236 | 0.7202 | −7.92 |

- Gap = 0.0575 ≥ 0.05 bar — cleared by 0.0075. Honest reading: the
  contamination story holds but the measured within-video effect is
  modest; Arm C landed at 0.562, between the predicted ≈0.52 and Arm M.
- Dissociation (not preregistered, descriptive): removing the mask
  HELPS the cross-video contrast (0.850 vs 0.720) while hurting the
  within-video one — full attention smears a global verdict over every
  chunk (mean z jumps −7.91 → −0.13), which aids video-level
  discrimination and destroys moment-level discrimination. This is the
  contamination mechanism showing both of its faces, and it cleanly
  motivates the two-call design: global judge call (full attention) +
  locator call (masked isolation).

## Efficiency (descriptive, 30 videos)

| Regime | s/video | tokens/video |
|---|---|---|
| N sequential isolated calls | 0.442 | 3111 |
| Plain left-padded batch | 0.602 | 7036 |
| Packed masked pass | **0.112** | **1058** |

Prefix computed once; 3.9× over the sequential loop on this stack. The
honest three-regime decomposition per the Codex caveat — the claim is
"one forward, prefix shared", not a fabricated 1/N.

## Deviations

1. Smoke set = the 3 videos with the most chunks (217 chunks) instead
   of first-3-by-id (which have one chunk each — vacuous for a
   mask check). Efficiency set kept first-30-by-id as specified.
2. Arm C uses an explicit 4D causal mask (same code path as Arm M);
   verified bit-identical to the default forward.

## Consequence

Direction 3 is the first localization direction to survive its pilot
(directions 1 and 2 died). Next per the prereg's interpretation
boundary: method-stage prereg — LAVAD-convention pooled frame-level
AUC on HateMM vs LELA's published numbers, extension corpora,
multimodal-prefix ablation, label-free thresholding, and the honesty
section on the within-video boundary (0.62).
