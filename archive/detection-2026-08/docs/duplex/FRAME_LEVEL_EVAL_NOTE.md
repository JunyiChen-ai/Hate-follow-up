# Result note — Frame-level evaluation on HateMM: PASS (direction 3, endpoint 1)

**Prereg:** `PREREG_frame_level_evaluation_hatemm.md` (febf147).
**Run:** 2026-08-18, script committed d18199c; outputs in
`results/frame_level_eval/` (gitignored). CPU-only transformation of
the pilot's per-chunk scores; no new model calls.

## Numbers (LAVAD convention: all 212 test videos' frames pooled, 1 fps)

| Row | ROC-AUC | PR-AUC | Within-hate macro |
|---|---|---|---|
| **Primary: z_masked, all frames** | **0.7451** | 0.5601 | 0.5706 (sd 0.212) |
| Sensitivity: sequential reference | 0.7459 | — | 0.5736 |
| Sensitivity: covered frames only | 0.7957 | — | 0.5694 |
| Descriptive: z_causal (unmasked) | 0.7461 | 0.5168 | 0.5316 |

28,751 frames (6,965 positive); uncovered frames (silence/music,
12.5%) floored per prereg. Bar was ≥ 0.65 → **PASS**.

Published reference points (LELA, protocol unstated — comparison to
reported numbers, no equivalence claim): GPT-4o Mini 72.64 at 12–16
calls/frame; Gemini-2.0 Flash 70.28; best open 7B 64.73; Qwen2.5-7B
62.14. Our single text-only locator forward on an open 8B lands at
74.5 pooled (79.6 on covered frames).

## Two honesty findings (mandatory reading before any paper claim)

1. **The pooled metric barely feels the mask.** z_causal reaches
   0.7461 pooled — equal to masked — because its cross-video inflation
   (chunk-level 0.850 vs 0.720) and its within-video collapse (macro
   0.532 vs 0.571) cancel in the pooled population. The mask's value
   is NOT the pooled headline; it is (i) score semantics — masked
   scores ARE isolated judgments (mean z −7.91, calibrated to the
   judge's Yes/No scale, thresholdable label-free; causal scores sit
   at mean −0.13, a smeared global verdict with no per-segment
   meaning), (ii) within-video discrimination, (iii) exact equivalence
   to N isolated calls (fidelity 0.9989). A paper must say this
   plainly and use the decomposition as a finding: the field's
   operative metric (pooled frame AUC) is dominated by video-level
   discrimination — we measure both components separately.
2. **Within-video remains the boundary.** Frame-level within-hate
   macro 0.571 (chunk-level 0.624): moment-vs-moment discrimination
   inside a hate video is real but weak, for us and — by construction
   of the metric — unmeasured for every published system.

## Status

Direction 3 has now passed: novelty check (NOVEL-with-caveats,
f0deb1f), mechanism pilot (both frozen clauses, febf147), and the
benchmark endpoint (this note). Remaining method-stage items, each its
own prereg, owner input welcome on ordering: multimodal-prefix
variant; MHClip_EN/ZH + ImpliHateVid extension; label-free frame
threshold (KDE-valley/Otsu on chunk scores) → interval extraction;
within-video improvement attempts; paper framing per the novelty
verdict (lead with contamination finding + isolation remedy; concede
the mask primitive to SingGuard/InvariRank/T3S).
