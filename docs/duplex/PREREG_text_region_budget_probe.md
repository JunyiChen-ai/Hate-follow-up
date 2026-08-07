# Pre-registration — Text-region pixel-budget probe (MHClip-EN test)

**Frozen:** 2026-08-08, before any judge scoring under the new visual arms.
The budget-matched rendering scheme (section "Frozen rendering scheme") must be
completed and committed before the first judge call; it may depend on
engineering constraints but never on scores.
**Compute:** one RTX 5090; OCR box detection + 3 judge arms × 161 videos.
**Status:** evidence-side probe on the frozen single-call 8B judge. This is
the spatial information-restoration candidate named in
`PROVENANCE_AND_SUCCESSOR_PROBES_NOTE.md` §5, gated by the completed A2 census.

## Phenomenon and mechanism

MHClip-EN is the one corpus where the judge's ranking itself is weak (test AUC
0.78) and the threshold is blameless. The A2 census found text boxes on 84% of
sampled videos at native resolution, and MHClip-EN is speech-poor (median
fresh transcript 318 chars vs 952 on ImpliHateVid), so on-screen text is
plausibly load-bearing evidence. The judge receives ~91 visual tokens per
frame, at which burned-in text is plausibly illegible. Mechanism: re-routing a
fixed share of the visual token budget to native-resolution crops of detected
text regions restores an evidence channel the current rendering destroys.
Restoration acts on the input; the judge, prompt, call count (1), and readout
(raw z) are frozen.

The judge's behavioral law (question-insensitive, evidence-sensitive) predicts
input-side restoration can move decisions where prompt changes cannot.

## Arms (all budget-matched; same total image-token count ± 2%)

- **BASELINE** — existing uniform-16 rendering (scores already on disk).
- **TEXT** — part of the budget replaced by native-resolution crops of
  OCR-detected text regions (detector and box-selection rule identical to the
  A2 census; recognized strings never enter the prompt — OCR is used only to
  locate boxes).
- **RAND** — identical crop count/sizes at uniformly random locations
  (seed 20260808), same donor frames.
- **ANTI** — identical crop count/sizes placed to avoid all detected text
  boxes (maximal-distance placement), same donor frames.

Videos with no detected text keep BASELINE rendering in all arms and are
analyzed as a separate stratum.

## Frozen rendering scheme

To be filled by the implementing run BEFORE scoring, specifying: number of
replaced frames/crops, crop resolution cap, donor-frame selection rule, and
the token accounting showing budget match across arms.

- Replaced slots: uniform-16 grid retained; the 4 grid frames with the largest
  detected text-box area (native resolution) each contribute one crop; the 4
  grid frames with the smallest text-box area are dropped to pay for them
  (12 full frames + 4 crops = 16 image slots).
- Crop: tight union bounding box of detected text boxes in the donor frame,
  padded 8%, rendered at native resolution capped to the same per-slot token
  count as a full frame.
- RAND/ANTI use the same 4 donor frames and the same crop aspect/area, moved
  per their placement rule.
- Token accounting to be reported in the run log before scoring.

## Frozen readout and metrics

8B judge, single call, raw z. Primary metric: ROC-AUC on the MHClip-EN test
split (the claim is about the judge's ceiling, not the threshold). Secondary:
macro-F1 at the label-free KDE-valley threshold recomputed per arm, and the
speech-poor/speech-rich split at the frozen A2 median (318 fresh-transcript
chars).

## Frozen decision rule

The mechanism **PASSES** only if all clauses hold:

1. AUC(TEXT) − AUC(BASELINE) ≥ +0.03.
2. AUC(TEXT) − AUC(RAND) ≥ +0.02 and AUC(TEXT) − AUC(ANTI) ≥ +0.02
   (targeting, not generic resolution/diversity, is load-bearing).
3. The AUC gain concentrates in the speech-poor half: speech-poor ΔAUC >
   speech-rich ΔAUC (the A2 revised prediction).

Clause 1 failing kills spatial restoration on MHClip-EN. Clause 2 failing
means any gain is a resolution effect, not a text-evidence mechanism — do not
promote. Clause 3 failing keeps the result but retires the speech-scarcity
story.

## Safety and hygiene

- Recognized OCR strings never enter prompts, logs, or committed files.
- No raw hateful-video IDs in committed reports; machine-readable results stay
  in gitignored `results/`.
- Pre-flight integrity scan: decode every frame of every arm before scoring;
  coverage assertion after scoring (n scored = n expected) — the MHClip-ZH
  truncated-frame incident must not recur.
