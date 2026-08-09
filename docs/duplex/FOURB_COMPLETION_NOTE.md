# The 4B completion sweep: the fixed model swap on all five corpora

**Date:** 2026-08-10. **The swap does not survive the third corpus, and it does not survive the fifth at all.**
**Compute:** 164 GPU-seconds for the three new corpora over 704 videos, 194 seconds of wall clock end to end; the weights were already cached and nothing was downloaded. The analysis is CPU only and rescores nothing.
**Preregistration:** `docs/duplex/PREREG_4b_completion.md`, frozen at commit ba00da1 before a single new score was written.
**Scoring:** `scripts/duplex/run_4b_completion.sh` driving `src/duplex/extract_duplex_readout.py` unmodified.
**Analysis:** `scripts/duplex/fourb_completion_analyze.py`, which imports its KDE recipe, AUC routine, macro-F1 definition, oracle search and head loader from `scale_emergence_analyze.py` rather than re-implementing any of them.
**Machine-readable result:** `results/scale_emergence/fourb_completion.json`

This is a **fixed model swap evaluated everywhere**: one `Qwen3-VL-4B-Instruct` checkpoint scored on all five corpora, compared with one `Qwen3-VL-8B-Instruct` checkpoint scored on all five corpora. No table, mean or sentence below selects a model per dataset. It is measurement, not a method claim: D3 fired at d804f51 and killed the graded angular-rotation account, so no mechanism is attached to any scale effect here and none is offered.

## Headline

On four corpora the swap looks like a small win, +0.0114 on the mean valley macro-F1. On five it cannot be scored, because on HateClipSeg the 4B's score distribution has one mode and the label-free recipe returns no threshold at all. The four-corpus margin is also not a scale effect: the 4B loses to the 8B on three of the four corpora and wins on HateMM by more than the three losses combined. A mean assembled from one large gain and three consistent losses is a HateMM result divided by four.

The ceiling moves the wrong way as well. The 4B's labeled-oracle mean is below the 8B's on both aggregates, which means the swap lowers the best score any threshold rule could ever reach on this judge.

## The five-corpus table

Held-out test splits, except HateClipSeg, which has no official split and is the whole annotated corpus surviving media attrition. Every cell reads the same prompt, the same 16 frames at 91 vision tokens per frame, the same gated fresh transcripts uncapped, and one forward pass per video. AUC, valley macro-F1 and oracle macro-F1 use labels; mode count and trough depth do not.

| corpus | collapse | n | arm | AUC | modes | trough | valley macro-F1 | oracle macro-F1 | gap |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| ImpliHateVid | hateful | 400 | 4B | 0.9410 | 2 | 0.3615 | 0.8568 | 0.8624 | 0.0055 |
| ImpliHateVid | hateful | 400 | 8B | 0.9474 | 2 | 0.6041 | **0.8823** | 0.8925 | 0.0101 |
| HateMM | hate | 215 | 4B | 0.9120 | 2 | 0.3499 | **0.7904** | 0.8612 | 0.0708 |
| HateMM | hate | 215 | 8B | 0.9230 | 2 | 0.2835 | 0.6562 | 0.8879 | 0.2317 |
| MHClip-EN | hate + offensive | 161 | 4B | 0.7855 | 2 | 0.0000 | 0.6561 | 0.7127 | 0.0566 |
| MHClip-EN | hate + offensive | 161 | 8B | 0.7848 | 2 | 0.0797 | **0.6962** | 0.7017 | 0.0055 |
| MHClip-ZH | hate + offensive | 149 | 4B | 0.8624 | 2 | 0.0684 | 0.7023 | 0.7706 | 0.0683 |
| MHClip-ZH | hate + offensive | 149 | 8B | 0.8551 | 2 | 0.1679 | **0.7256** | 0.7932 | 0.0676 |
| HateClipSeg | offensive union | 394 | 4B | 0.7620 | 1 | -- | -- | 0.6905 | -- |
| HateClipSeg | offensive union | 394 | 8B | 0.7547 | 2 | 0.2980 | **0.6412** | 0.6725 | 0.0312 |
| HateClipSeg | hateful strict | 394 | 4B | 0.7163 | 1 | -- | -- | 0.6698 | -- |
| HateClipSeg | hateful strict | 394 | 8B | 0.7701 | 2 | 0.2980 | **0.4437** | 0.7214 | 0.2777 |

Trough is the de-quantized relative KDE trough depth on the frozen E7 convention, and the valley and oracle columns are computed on the de-quantized score. A dash means the recipe found fewer than two modes and returned no threshold, which is the substantive result for that cell and not a missing measurement. The HateClipSeg label-free threshold is identical across its two collapses by construction, since no fitted quantity reads a label; only the labels the predictions are scored against change.

The MHClip-EN 4B trough is not rounded to zero, it is 4.1e-06. The recipe found two modes and placed a valley whose density is indistinguishable from the lower mode's. That is a shoulder, not a valley, and the number it returns is a threshold in name only.

**Reproduction.** Recomputing the 8B arm from the stored bf16 scores with this code returns 0.8823, 0.6562, 0.6962, 0.7256, 0.6522 and 0.4479 against the six committed values, every one within 0.00003 of the report it came from. The committed references quoted in the run request were checked cell by cell and none was misquoted. De-quantization moves no score by more than 0.2453, one bf16 grid step at these magnitudes, and it moves a valley macro-F1 by at most 0.011 (HateClipSeg union, 0.6522 stored against 0.6412 de-quantized); both figures are in the JSON and the de-quantized one is primary throughout, matching `SCALE_EMERGENCE_NOTE.md`.

## The means

| aggregate | arm | valley macro-F1 mean | labeled-oracle mean |
|---|---|---:|---:|
| Four corpora (IHV, HateMM, EN, ZH) | 4B | **0.7514** | 0.8017 |
| Four corpora | 8B | 0.7401 | 0.8188 |
| Five corpora (adding HateClipSeg union) | 4B | **undefined** | 0.7795 |
| Five corpora | 8B | 0.7203 | 0.7896 |

The 8B four-corpus valley mean recomputes at 0.74009 against the committed 0.7401 in `ANCHORED_OPERATING_POINT_NOTE.md`. Its oracle mean recomputes at 0.8188 against the committed 0.8175, the 0.0013 difference being the de-quantization of MHClip-EN.

The five-corpus 4B mean is undefined rather than low, per the rule frozen in the pre-registration: an arm with any dashed cell gets no mean, because averaging over the corpora where a rule happened to return something is exactly the selection the dash exists to prevent. Stated plainly, a deployment that swaps the 8B for the 4B and runs the label-free recipe gets no decision at all on one of these five corpora.

## Where the 4B helps and where it hurts

| corpus | valley delta, 4B minus 8B | AUC delta | oracle delta |
|---|---:|---:|---:|
| ImpliHateVid | -0.0255 | -0.0064 | -0.0301 |
| HateMM | **+0.1342** | -0.0110 | -0.0267 |
| MHClip-EN | -0.0401 | +0.0007 | +0.0110 |
| MHClip-ZH | -0.0233 | +0.0073 | -0.0226 |
| HateClipSeg, union | no threshold | +0.0073 | +0.0180 |
| HateClipSeg, strict | no threshold | -0.0538 | -0.0516 |

One corpus carries the entire four-corpus margin. HateMM gains 0.1342; the other three lose 0.0255, 0.0401 and 0.0233, summing to 0.0889. The net is +0.0453 over four corpora, which is the +0.0114 on the mean. Remove HateMM and the fixed 4B swap loses on every remaining corpus in the set.

The HateMM gain is also the case the pre-registration flagged in advance as the one to distrust, and the error counts confirm the flag. The 8B's valley sits at -2.35 and calls 70 of 129 negatives positive; the 4B's sits at +1.81 and calls 40. The 4B does not read HateMM better — its AUC is 0.0110 lower and its oracle 0.0267 lower — its valley simply lands higher in a corpus whose normal class is documented as saturated with hate-adjacent surface features. What improved is where the recipe cut, not what the judge saw. The same structure appears in reverse on MHClip-EN, where the 4B's valley lands at +4.42 and misses 30 of 49 positives while the 8B's lands at -0.50 and misses 16.

HateClipSeg is the clean failure. The corpus that already defeated every threshold rule in the closeout program defeats the 4B one step earlier: the 4B never produces a bimodal score distribution on it, so there is nothing for the recipe to cut. This is the 2B's HateMM failure mode from d804f51 reappearing one scale up, on the hardest corpus. The honest reading of that is not that the 4B is worse — a rule that returns nothing is more honest than a rule that returns a shoulder, as MHClip-EN shows — but it does settle the deployment question. A fixed 4B swap is not a drop-in replacement, because on one corpus in five it declines to decide.

## What the mean says next to TRIAGE

TRIAGE reports 0.808 over the four original corpora at 1.73 MLLM calls per video. Our label-free valley means are 0.7401 at 8B and 0.7514 at 4B, both at one call per video, so the swap closes about a sixth of a gap that was 0.068 and is now 0.057. That is the generous reading and it is not the important one.

The important one is the oracle row. The four-corpus labeled-oracle mean falls from 0.8188 at 8B to **0.8017** at 4B, which puts the 4B's perfect-threshold ceiling *below* TRIAGE's reported number. At 8B the closeout could still say that threshold selection, solved perfectly, would land about one point above the comparator; after the swap it would land two points below it. Whatever the 4B buys on the label-free operating point, it costs on the ranking that any operating point has to work with, and it costs enough to remove the headroom entirely. The swap is not a route to closing the gap with TRIAGE. It is a way of getting slightly luckier with the cut while ranking slightly worse.

## Honest reading

The four-corpus result is real and the interpretation the run request offered for it does not survive the fifth corpus. "The label-free operating point prefers the intermediate scale" would require the preference to show up as a tendency across corpora; instead it shows up once, on HateMM, in the direction and for the reason the pre-registration named as suspect in advance, while the 4B loses on the three corpora where the negative class is not saturated and produces no threshold at all on the fifth. The mechanism-free empirical statement that the five cells license is narrower and less interesting: *the KDE valley recipe happens to cut HateMM's 4B score distribution in a better place than it cuts the 8B's, and nowhere else in this set does the smaller model help.* Ranking is uniformly at or below the 8B outside two marginal cells, the labeled ceiling is below the 8B on both aggregates, and the corpus that has broken every threshold rule in this program breaks the 4B before the threshold stage is even reached. Nothing here recommends the swap, nothing here recommends per-corpus scale selection, and nothing here revives the mechanism D3 killed.

## Limitations

- One model family and one post-training recipe. Qwen3-VL's scale ladder is not separated from its recipe, and the 4B-against-8B contrast is width at fixed depth: both have 36 layers, at 2560 and 4096.
- Five corpora, two arms, twelve corpus-collapse cells. Enough to show that the four-corpus margin rests on one corpus, not enough to characterise when a smaller judge's valley lands better.
- HateClipSeg has no official split, is the whole annotated corpus after media attrition, and is 87.3 percent positive under the union collapse, where a rule that flags almost everything scores well by accident of prevalence. Its union-collapse numbers should not be read as a detection result.
- The unweighted mean over corpora of 149 to 400 videos is a crude summary and is used only because the committed 0.7401 reference is defined that way.
- The label-free threshold is transductive: it reads the whole unlabeled test batch at once. A per-video online decision would need the threshold carried over, and the 4B's HateClipSeg failure would then surface as an absent carried threshold.
- One arm per cell, one forward pass per video, no repeated sampling and no ensembling.
- No causal intervention and no mechanism. The trough depths are reported as observations, as `SCALE_EMERGENCE_NOTE.md` requires after D3.
- The oracle column uses the macro-F1-maximising convention. The committed reports maximise hateful-class F1 instead; the two differ materially only on MHClip-EN, where the hateful-F1-max threshold gives macro-F1 0.6366 at 4B and 0.6831 at 8B. Both conventions are in the JSON, and the 4B's oracle deficit is larger, not smaller, under the committed convention.
