# Stress test round 1 — Codex (xhigh), 2026-07-09

Verdict: all three drafts "needs revision" — wording/traceability only, no evidence gaps. Key findings and dispositions:

ADOPTED:
- ARuf natural-row rounding: 72B 69.7→69.6, 27B 74.9→74.8 (recomputed from E1 summary.csv exact means).
- "accuracy degrades" → "performance degrades" (table is macro-F1).
- Untraceable "more than twice the parameter-weighted cost" → replaced with traceable call-count phrasing (72B pass/video vs 2B pass + ≤0.8 or 0.69 verifier calls/video).
- "on every video" → "every evaluable video under the same evaluation protocol as Table 1" (coverage 214/215 HM, 392/401 IH).
- "under half of the videos" → "0.69 verifier calls per video on average" (EN/ZH band rates exceed half).
- "post-submission verifier re-run" → "post-submission reproducibility audit … re-pinned to exactly reproducible artifacts" (both cpjh and 7rqV footnotes).
- SAGE "strongest recent system on HateMM" → "a strong recent system" (MM-HSD 87.8 under CV exists).
- E2 baseline list reordered to match table column order (HateMM 79.5, EN 76.4, ZH 75.2, IH 80.3); "best of the six orderings" made explicit.
- 7rqV "falsified" → "tested by a counterfactual ablation"; "match or exceed on three of four benchmarks" → competitive-with framing + explicit SAGE extension promise; train sizes hedged to "roughly 550 to 1,300 (appendix)".
- "happy to run further analyses" → "welcome further questions".

DELIBERATE DEVIATIONS from Codex (with reason):
- Ablation deltas (6.3/2.4/2.9/1.3/4.2) and "720 configurations"/"§4.6" KEPT with explicit attribution to the submitted paper's Table 2/§4.6 — the provenance gate allows `paper` as a source and reviewers hold the paper; Codex's stricter reading came from the artifact list in my prompt.

Full Codex output preserved in the session transcript (threadId 019f4395-ee22-7640-992e-c3ba10e6f466).
