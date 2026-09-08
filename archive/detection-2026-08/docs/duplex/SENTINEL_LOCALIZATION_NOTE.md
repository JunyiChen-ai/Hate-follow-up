# Result note — Sentinel logit-lens localization pilot: DEAD (direction 2)

**Prereg:** `PREREG_sentinel_localization_pilot.md` (99d492c). **Novelty:**
checked NOVEL-with-caveats before running (nearest neighbors: 2607.27667
single-prefill answer-direction projection, labeled calibration, static
images; 2606.10487 streaming moderation probes, trained, own-output).
**Run:** 2026-08-12, script committed 1505911; raw outputs in
`results/sentinel_localization/` (gitignored). **Verdict: both arms at
chance → interpretation-grid branch "both < 0.60": the
elicited-local-judgment family dies; goal loop re-enters idea
discovery.**

## Numbers (326 sparse HateClipSeg videos, one locator forward each)

- Arm A (per-segment sentinel, median of last-4-layer margins): macro
  AUC **0.4994** (final layer 0.4843). Token-density control 0.5282.
  Bars: 0.65 and control+0.05 — both failed.
- Arm B (packed per-segment questions, diagnostic): **0.5186** — also
  below the 0.60 capability floor and below the density control.
- Per-video AUC distributions symmetric around 0.5 — coin flip, not a
  diluted signal.
- Position-drift artifact dominates: Arm A margins drift UP with chunk
  index (mean Spearman +0.384), Arm B drifts DOWN with question index
  (−0.356). Neither reflects content.
- Arm B answers Yes to essentially every segment (margin mean +5.36,
  IQR [5.06, 6.08]) once it has read the whole video.
- The sentinel readout DOES track the video-level verdict (per-video
  mean margin vs frozen z: Pearson 0.452) — right direction, zero
  within-video discrimination.
- Rigor: final-layer sentinel margin verified against the model's own
  logits (max |d| 0.00 after applying the final RMSNorm; the
  hidden-states tuple is pre-norm in transformers 4.57.1 — deviation
  documented); frozen imports for rules/frames/token ids; instruction
  wording SHA-256 recorded; 0.71 s/video.

## Interpretation (bounded by the prereg)

1. Combined with direction 1's death (attribution of the global
   verdict ≈ token density): the two failures share one mechanism —
   **the judge forms a single global verdict, and every within-video
   probe returns that verdict smeared over the timeline**. With causal
   attention over the full video, each "local" judgment position has
   already absorbed the global context; after reading a hateful video,
   the model stamps every segment as violating.
2. What this does NOT yet establish: whether the model can judge an
   ISOLATED segment correctly when the rest of the video is absent
   (LELA's per-frame calls suggest yes for GPT-4o-mini but note open
   7B models are "substantially worse"). This is the decisive
   capability question for any remaining direction (including
   one-forward parallel isolation via block-diagonal attention masks),
   and is answered by a cheap descriptive diagnostic before the next
   idea-discovery round.
