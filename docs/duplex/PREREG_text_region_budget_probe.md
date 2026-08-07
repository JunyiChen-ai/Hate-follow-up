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

Filled in 2026-08-08 by the implementing run, before the first judge call and
before any arm score existed. Everything below is either a restatement of the
rule frozen above or a measurement of the pixels, never of the outcome.

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

### Implementation, as executed

Frames. The BASELINE judge reads the 16 pre-extracted native-resolution jpgs in
`frames_16/<video_id>`, so the arms read the same files. Full-frame slots point
at those files unchanged; only crop slots are newly rendered. No frame is
re-extracted from the mp4 and no timestamp can drift.

Detector. `scripts/duplex/text_region_budget/detect_boxes.py` imports the box
functions from the A2 census module itself rather than copying them, and
constructs the same `easyocr.Reader(["en"])` at engine-default CRAFT
thresholds. Geometry only; no string is requested or stored.

Text area of a frame is the union OCR-box area as a fraction of frame area, the
A2 census measure. A frame is a candidate donor when that fraction reaches
0.01, the census's own substantive-frame threshold. Donors are the k candidates
with the largest text area, k = min(4, number of candidates), ties broken by
frame index; the k frames with the smallest text area among the non-donors are
dropped. A video with no candidate donor keeps BASELINE rendering in all three
arms and forms the no-text stratum: 24 of 161 videos.

Crop. Union bounding box of the donor's detected boxes, each side expanded by
8% of the corresponding bounding-box side and clamped to the frame.

Token cap. A full frame costs T = (h/32)(w/32) merge cells after Qwen3-VL
`smart_resize(factor 32, min_pixels 65536, max_pixels 100352)`, the settings the
BASELINE judge runs under. T is uniform within every video of this split and
takes the values 81, 84, 88, 91 and 96. Each crop is bicubic-resampled to a
merge grid (gh, gw) whose product lies within two cells of T and whose aspect
ratio is the closest available to the crop's, so the processor's own resize is
a no-op and the slot cost is known exactly. Two cells of slack per crop is the
smallest tolerance the integer factorisations allow; it is 30 times smaller than
the ±2% budget rule. The scorer asserts the observed image-token total against
the predicted one for every video and aborts on any mismatch.

Slot order. Temporal, with each crop placed immediately after its donor frame.

RAND. Uniformly random legal top-left, `numpy.default_rng(20260808)`, drawn in
video order and then slot order. ANTI. The legal top-left maximising the minimum
Euclidean gap to every detected box in that frame, searched on a 65×65 grid of
candidate positions, ties broken toward the smallest (y, x).

### Token accounting, measured before scoring

Corpus image tokens: BASELINE 232,688, each arm 232,468, a deviation of
−0.09%. Per-video deviation ranges from −0.31% to +0.57%, mean absolute 0.13%,
and 0 of 161 videos fall outside ±2%. Pre-flight decode: 7,728 images across
the three arms opened and loaded, 0 failures.

### Geometry the frozen rule turns out to produce, and how to read clause 2

Measured on the 528 donor slots before scoring:

- The union bounding box is large. It covers 36% of the frame at the median,
  and for 39% of donor slots it covers more than half the frame. The crop
  therefore magnifies text by only 1.66× in linear scale.
- Because the crop is large, a displaced rectangle of the same size still lands
  on much of the text. RAND covers 52% of the donor frame's text area on
  average and ANTI covers 39%, against 99.5% for TEXT. ANTI reaches a
  genuinely text-free position in 277 of 528 slots; in the other 251 the frame
  offers no such position and the maximal-distance rule, which is the
  operational half of the ANTI definition, still applies.
- On donor frames, 93.8% of text area already clears the census's 12-pixel
  judge-scale legibility bar under BASELINE rendering; the crop raises that to
  96.7%. Donor selection picks the frames whose text is largest, which are the
  frames whose text was least likely to have been destroyed in the first place.

These are properties of the rule that was frozen, not choices made here, and
the rule is executed unchanged. They do change how the clauses may be read.
Clause 1, TEXT against BASELINE, is unaffected: both arms are exactly what the
pre-registration says and clause 1 is the clause that kills the mechanism.
Clause 2 is diluted, because RAND and ANTI deliver half and two-fifths of the
same text at the same magnification. A clause 2 failure therefore cannot by
itself distinguish "targeting is not load-bearing" from "the control arms were
too close to the treatment to separate". A clause 2 pass under this dilution is
correspondingly strong.

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
