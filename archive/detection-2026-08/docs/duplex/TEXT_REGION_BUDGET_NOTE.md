# Text-region pixel budget: result note

**Date:** 2026-08-08. **Verdict:** **FAIL** under the frozen rule; all three
clauses fail.
**Compute:** one RTX 5090. OCR box detection over 2,576 frames, then three
judge arms of 161 videos each, 483 8B calls, about six GPU-minutes of scoring.
**Preregistration:** `docs/duplex/PREREG_text_region_budget_probe.md`
**Code:** `scripts/duplex/text_region_budget/`
**Machine-readable result:** `results/text_region_budget/results.json`

Re-routing a quarter of the judge's visual token budget to native-resolution
crops of detected on-screen text moves MHClip-EN test AUC from 0.785 to 0.789.
The rule required +0.03. The arm built to *avoid* text moves it further, to
0.793, and the arm that places the same crops at random moves it least. The
gain such as it is sits in the speech-rich half, not the speech-poor half the
story predicted. Spatial text restoration does not raise this judge's ceiling
on MHClip-EN.

## What was run

The BASELINE arm is the existing held-out test measurement, unchanged and not
rescored. The three new arms differ from it in exactly one respect: the pixels
in four of the sixteen image slots. Prompt, transcript overrides, transcript
cap, model, dtype, call count and the raw-z readout are the same objects, not
reimplementations. The scorer imports the prompt builder and the transcript
logic from `src/duplex/extract_duplex_readout.py`, the module that produced the
BASELINE scores, and the analysis imports the AUC estimator, the KDE-valley
recipe and the operating-point arithmetic from
`scripts/duplex/crossbench_analyze.py`, the module that produced the committed
BASELINE report.

On-screen text was located by the A2 census detector, imported from the census
module rather than copied: the same EasyOCR CRAFT engine at the same
engine-default thresholds, detection only, with recognised strings never
requested and never stored. The four grid frames with the most detected text
area each contribute one crop of the padded union bounding box, and the four
frames with the least text area are dropped to pay for them. TEXT places the
crop on the text. RAND places a rectangle of identical size at a uniformly
random legal position under seed 20260808. ANTI places it at the legal position
furthest from every detected box. The 24 videos with no detected text keep
BASELINE rendering in all three arms.

Two integrity checks passed. Every image of every arm was opened and loaded
before the first judge call: 7,728 images, no failures. Every arm reached
161 of 161 videos scored, and the scorer aborts if the image-token total of any
video departs from the value predicted when the arm was built.

A third check was free and worth having. The 24 no-text videos are rendered
identically in all four arms, so their raw z must be bit-identical to BASELINE.
It is, in all three arms, to the last digit. The pipeline is therefore the
BASELINE pipeline whenever the pixels are the BASELINE pixels.

## Token accounting

Reported before scoring and repeated here. Across the corpus the arms cost
232,468 image tokens against the BASELINE's 232,688, a difference of 0.09%.
Per video the deviation runs from -0.31% to +0.57%, mean absolute 0.13%, with
0 of 161 videos outside the ±2% rule. No arm bought its result with pixels.

## Results

ROC-AUC, 49 hateful against 112 normal.

| Arm | Overall | Speech-poor (n=65) | Speech-rich (n=96) | Text detected (n=137) | No text (n=24) |
|---|---:|---:|---:|---:|---:|
| BASELINE | 0.7847 | 0.7768 | 0.7951 | 0.8044 | 0.6797 |
| TEXT | 0.7888 | 0.7730 | 0.8071 | 0.8087 | 0.6797 |
| RAND | 0.7864 | 0.7749 | 0.8025 | 0.8063 | 0.6797 |
| ANTI | 0.7929 | 0.7736 | 0.8102 | 0.8142 | 0.6797 |

The bootstrap interval on the BASELINE AUC alone is [0.714, 0.853], so no
column here separates from any other. A paired bootstrap on the difference is
the sharper statement: TEXT minus BASELINE has a 95% interval of
[-0.008, +0.015], which excludes the +0.03 the rule demanded. The failure is
not a near miss under noise; the required effect lies outside the interval the
data support.

Rank agreement with BASELINE is 0.993 by Spearman in every arm, on a mean
absolute z shift of 0.77 in a score range spanning 39 units. The
intervention changes the pixels the judge sees and barely changes the order it
puts the videos in.

Secondary metric, macro-F1 at each arm's own label-free KDE valley:

| Arm | Valley (raw z) | Macro-F1 |
|---|---:|---:|
| BASELINE | -0.50 | 0.6962 |
| TEXT | +2.51 | 0.6491 |
| RAND | +1.41 | 0.6572 |
| ANTI | +2.81 | 0.6640 |

Every arm loses macro-F1 while its AUC is flat or slightly up. The rendering
change is small enough to leave the ranking intact and large enough to move the
valley by three raw-z units, which costs recall. That is a fact about the
threshold recipe, not about the visual intervention, and it is a reminder that
the label-free threshold is the more fragile half of the method.

## Clause verdicts

| Clause | Rule | Observed | Verdict |
|---|---|---:|---|
| 1 | AUC(TEXT) - AUC(BASELINE) >= +0.03 | +0.0041 | **FAIL** |
| 2 | TEXT - RAND >= +0.02 and TEXT - ANTI >= +0.02 | +0.0025 and -0.0041 | **FAIL** |
| 3 | speech-poor gain > speech-rich gain | -0.0038 against +0.0120 | **FAIL** |

Clause 1 fails by a factor of seven. Clause 2 fails in a direction the story
cannot absorb: ANTI, the arm designed to look everywhere except at the text,
outranks TEXT. Clause 3 fails with the sign reversed, since every arm is
slightly worse than BASELINE on the speech-poor half and slightly better on the
speech-rich half.

The pre-registration flagged before scoring that clause 2 was diluted, because
the union bounding box is large enough that a displaced rectangle of the same
size still lands on 52% of the text under RAND and 39% under ANTI. That caveat
protects a clause-2 failure from being read as a refutation of targeting. It
does not rescue anything here, because clause 1 is undiluted and fails on its
own, and because the ordering that emerged puts ANTI on top rather than leaving
the three arms tied.

## Interpretation

The mechanism assumed an evidence channel that the current rendering destroys.
The geometry measured before scoring says the channel was mostly intact.
On the donor frames, 93.8% of detected text area already clears the census's
12-pixel legibility bar at BASELINE resolution, and the crop raises that only to
96.7%. Donor selection picks the frames with the most text, and those are
exactly the frames whose text was largest and therefore least likely to have
been lost. The A2 census measured illegibility across all frames and all boxes;
the frames this scheme actually spends its budget on are not the illegible ones.

So the probe did not test restoration of a destroyed channel. It tested adding
1.66x linear magnification to text that the judge could already read, at the
price of four frames of temporal coverage. The result is that the judge's
ranking is indifferent to the trade. The small positive drift shared by all
three arms, largest in ANTI, is consistent with the drop of four
low-text frames mattering more than what replaces them, which is a statement
about frame redundancy and not about text.

This also closes off the obvious repair. Cropping tighter, to the single
largest text box, would raise magnification to 4.5x and would let ANTI find
genuinely text-free placements in 97% of slots instead of 52%. But it would
also discard 53% of the detected text area, and the failure above is not a
failure of magnification: the text was legible and the ranking did not move.
A tighter crop makes a sharper experiment about a premise that has already been
measured false on this corpus.

MHClip-EN remains the corpus where the judge's ranking is weak and the
threshold is blameless. That weakness is not explained by on-screen text the
judge cannot read.

## Decision

Spatial text restoration is retired for MHClip-EN. It is not promoted, not
carried to the other three benchmarks, and not tried at a tighter crop.

The A2 census keeps its standing as a measurement, with one correction to how
it should be read. Its headline is that 84% of MHClip-EN videos carry
substantive on-screen text, and that is intact. Its implied inference, that the
text is therefore an evidence channel the judge is losing, does not survive: on
the frames where text is densest the judge could already read it, and giving it
more pixels changes nothing.

The successor-probe list in
`docs/duplex/PROVENANCE_AND_SUCCESSOR_PROBES_NOTE.md` §5 should drop the
spatial information-restoration candidate. The behavioural law that motivated
it, that this judge is question-insensitive and evidence-sensitive, is not
challenged by this result; what is challenged is the claim that on-screen text
is missing evidence. Any successor probe on MHClip-EN needs to name an evidence
channel that is measurably absent from the judge's input, not one that is
present and merely small.
