# Paper outline v1 (2026-08-19) — for owner review

Working title (plain, claim-first):
**"One Pass, Every Moment: Label-Free Temporal Localization of Hateful
Video Content by Isolating the Judge"**
(alternatives: "Global Verdict Contamination and Its Remedy: ...";
title decision is the owner's.)

## Framing rules (binding, from the novelty check f0deb1f)

Lead with the PHENOMENON and its REMEDY; concede the mask primitive
explicitly to SingGuard (2606.22873) / InvariRank (2604.27599) / T3S
(2511.17945) in related work. Never present block-diagonal masking as
an invention. The method is ONE thing (single locator forward); the
TRIAGE judge is not part of it (owner ruling).

## 1. Introduction

- Hate moderation needs WHERE, not just WHETHER (regulator/reviewer
  workload; temporal label noise line 2508.04900 as motivation).
- Existing localization: either 12–16 closed-model calls per frame
  (LELA) or training on labels (MultiHateLoc, WTAL/VAD lines). Open
  8B models reportedly far behind (LELA's own table: best 7B 64.7).
- Our finding: a frozen 8B judge CAN localize — but only if you stop
  it from reading the whole video. Contribution list:
  1. The contamination measurement (new): every in-context per-segment
     probe of a video-reading judge returns the global verdict smeared
     over the timeline (attribution 0.525, sentinel 0.499, packed
     questions 0.519 — all ≈ chance within-video).
  2. The remedy: one packed forward with per-segment isolation
     (mechanism borrowed, application + evidence new): mathematically
     ≡ N isolated calls (Spearman 0.9989), 3.9× cheaper, and the
     counterfactual (same tokens, full attention) collapses
     within-video 0.620→0.562 while INFLATING cross-video 0.720→0.850
     — the two faces of contamination, measured.
  3. Zero labels at trained-baseline quality: HateMM pooled frame ROC
     0.7451 vs best trained 0.7504; best within-video localization on
     MHC EN/ZH; best PR-AUC and video AUC on HateMM.
  4. The first fully specified frame-evaluation protocol + released
     GT arrays for HateMM/MHC (the field has NONE — survey table),
     plus a supervised-baseline suite re-run under it.

## 2. The contamination phenomenon (measurement section)

- Setup: frozen judge (cite base paper), HateClipSeg + HateMM gold.
- Table: all access modes (capability map) — attribution, sentinel,
  packed questions, isolated chunk/window/frame. The one contrast that
  works: isolation.
- External echo: 2605.20194 "carryover effect" (text, fixed by N
  calls); You Only Judge Once chose full attention without testing
  isolation — the axis was untested in public.

## 3. Method: masked parallel isolation (locator)

- One forward: shared rules prefix + timestamped ASR chunks,
  block-diagonal mask, branch-local position restart, per-branch
  Yes/No margin. Frame scores by span spreading; video score = max.
- Exactness: prefix+branch token identity ⇒ per-branch computation ≡
  isolated call (fidelity table; SDPA kernel caveat measured).
- Cost: one forward, prefix once; honest three-regime decomposition
  (serial / plain batch / packed) — 0.11 s/video on one RTX 5090.
- Deployment: ≤2 calls/video with the (separate) detection judge;
  localization itself is 1 call.

## 4. Evaluation protocol (contribution 4)

- Survey table: 6 localization works, none states grid or span→frame
  rule, none releases gold arrays (LELA/MultiHateLoc protocols
  unstated; MultiHateLoc repo LICENSE-only; EventVAD release
  non-runnable; upstream baseline repos all test-select checkpoints).
  Our protocol: 1 fps, half-open containment, frozen dirty-data rules,
  SHA-pinned arrays, val-carve selection for all trained baselines.

## 5. Results

- Main tables (3 corpora × 8 systems, supervision column). Headlines:
  zero-label ≈ best trained pooled on HateMM; best within-video on
  MHC; Vad-R1 (zero-shot MLLM) predicts whole-clip intervals 100% of
  the time in both prompt arms.
- The decomposition finding: pooled frame AUC is dominated by
  video-level discrimination (our counterfactual + per-table
  within-video rows ≤0.62 for everyone). The field's operative metric
  does not measure localization; we report both components.
- Channel analysis: audio is the strongest trained channel
  (0.767–0.778); our transcript locator reads the same speech signal
  label-free. Non-speech frames honestly at floor (MHC pooled cost).
- Ablations: mask on/off (the counterfactual IS the ablation of the
  story); threshold-family/label-free operating point (if included —
  pending); granularity (chunk vs window curves from diagnostics).

## 6. Honest boundaries (own section, not buried)

- Within-video moment discrimination is weak for ALL systems
  including ours (≤0.62); ZH macro rests on 7 videos; MHC spans are
  near-total (89–92% coverage) so MHC localization is nearly
  degenerate; LELA's published numbers are not reproducible (protocol
  unstated) and are cited as-reported, not compared.

## 7. Related work

- Hate/harm localization (6-paper line + video-level detectors);
  parallel-encoding masks (concede: SingGuard/InvariRank/T3S/
  Hydragen/packing); training-free VAD; speech-act/contamination
  antecedents.

## Open items before writing starts

1. EventVAD numbers land → final table freeze.
2. Owner: venue target + whether the base-paper judge appears as
   "call 1" in a system figure or is cited only.
3. Owner: do we include the label-free thresholding/interval
   extraction step (KDE/Otsu on chunk scores) as a method component
   or leave frame scores as the output? (No current metric needs
   intervals — only HateClipSeg Task 2 would.)
4. Decide whether the HateClipSeg capability-map section (deaths of
   directions 1–2) goes in the main paper or appendix.
