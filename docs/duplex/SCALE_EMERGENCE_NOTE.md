# Scale emergence of the label-free operating point: result note

**Date:** 2026-08-10. **D1 does not fire. D2 does not fire. D3 fires.**
**Compute:** 126 GPU-seconds for the 4B arm over 615 videos, plus about 90 seconds of download; the analysis is CPU only and rescores nothing.
**Preregistration:** `docs/duplex/PREREG_scale_emergence.md`, frozen at commit d0941a0 before a single 4B weight was downloaded.
**Scoring:** `scripts/duplex/run_scale_emergence.sh` driving `src/duplex/extract_duplex_readout.py` unmodified.
**Analysis:** `scripts/duplex/scale_emergence_analyze.py`
**Machine-readable result:** `results/scale_emergence/results.json`

## Headline

The transition is real, it is not smooth capability scaling, and it sits between 2B and 4B rather than anywhere near 8B. Both larger scales place their scores in two modes with a trough between them; the 2B does not, and on HateMM it produces one mode and no threshold at all. The construct is equally present at every scale, so nothing is missing from the 2B's representation. What the 2B lacks is a score distribution a label-free rule can cut.

The mechanism that was proposed to explain this does not survive its own test. Angular commitment does not order the three scales the way trough depth orders them: the 4B rotates its residual stream toward the answer direction *more* than the 8B does on both corpora while cutting a shallower trough on one of them. D3 fires and the angular-rotation account is disconfirmed as a graded explanation. It survives only as the coarse observation that the one scale without a valley is also the one scale whose cosine spread is an order of magnitude narrower than the others, which is a two-group contrast, not a mechanism.

## The three-scale table

Held-out test splits only. ImpliHateVid `test_clean`, 400 scored of 401, 200 hateful against 200 normal. HateMM `test_clean`, 215, 86 hate against 129 non-hate. Every arm reads the same prompt, the same 16 frames at 91 vision tokens per frame, the same gated fresh transcripts uncapped, and one forward pass per video. `sd(z)` and trough depth are label-free; AUC, probe AUC and the two macro-F1 columns use labels and are evaluative only.

| corpus | model | n | sd(z) | range | modes | trough | cos sd | AUC | probe AUC | valley macro-F1 | oracle macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ImpliHateVid | Qwen3-VL-2B | 400 | 1.00 | 4.51 | 2 | 0.0780 | 0.00388 | 0.9214 | 0.9214 | 0.8083 | 0.8425 |
| ImpliHateVid | Qwen3-VL-4B | 400 | 12.39 | 37.57 | 2 | 0.3615 | 0.04835 | 0.9410 | 0.9377 | 0.8568 | 0.8624 |
| ImpliHateVid | Qwen3-VL-8B | 400 | 13.40 | 40.13 | 2 | 0.6041 | 0.03322 | 0.9474 | 0.9480 | 0.8823 | 0.8925 |
| HateMM | Qwen3-VL-2B | 215 | 1.04 | 5.46 | 1 | -- | 0.00433 | 0.8931 | 0.8251 | -- | 0.8208 |
| HateMM | Qwen3-VL-4B | 215 | 12.13 | 38.25 | 2 | 0.3499 | 0.04661 | 0.9120 | 0.8839 | 0.7904 | 0.8612 |
| HateMM | Qwen3-VL-8B | 215 | 11.67 | 40.32 | 2 | 0.2835 | 0.02784 | 0.9230 | 0.8717 | 0.6562 | 0.8879 |

Trough is the de-quantized relative KDE trough depth on the frozen E7 convention. A dash means the recipe found fewer than two modes and returned no threshold, which is the substantive result for that cell and not a missing measurement. `cos sd` is the standard deviation across the corpus of the cosine between the final state, after the model's own RMSNorm, and the mean-Yes-minus-mean-No unembedding direction. The probe is ridge on the hidden state at `round(0.75 x n_layers)`, five stratified folds, penalty fixed at 1.0, which is layer 21 on the 2B and layer 27 on the 4B and the 8B.

Supporting geometry, same runs:

| corpus | model | valley | mean cos | direction norm | mean state norm | bf16 trough | max de-quantization shift |
|---|---|---:|---:|---:|---:|---:|---:|
| ImpliHateVid | 2B | 0.444 | 0.0045 | 1.545 | 143.5 | 0.0450 | 0.2419 |
| ImpliHateVid | 4B | -1.586 | -0.0082 | 1.097 | 218.1 | 0.3637 | 0.2279 |
| ImpliHateVid | 8B | -4.516 | 0.0004 | 1.654 | 226.8 | 0.6043 | 0.2286 |
| HateMM | 2B | -- | 0.0060 | 1.545 | 143.9 | -- | 0.2344 |
| HateMM | 4B | 1.806 | 0.0185 | 1.097 | 219.2 | 0.3476 | 0.2327 |
| HateMM | 8B | -2.346 | 0.0179 | 1.654 | 230.6 | 0.2831 | 0.2453 |

De-quantization moves no score by more than 0.2453, which is one bf16 grid step at these magnitudes, and the bf16 and de-quantized trough columns agree to three decimals everywhere except the ImpliHateVid 2B cell, where the exact-precision trough is the *deeper* of the two. That reproduces the round-2 refutation of the quantization confound on a second split and a second corpus.

The analysis reproduces every previously committed cell it overlaps. The 8B ImpliHateVid trough comes back at 0.6041 against the 0.604 in `TEST_RUNS_NOTE.md`, its valley macro-F1 at 0.8823 against 0.8823, the 8B HateMM trough at 0.2835 against 0.283 and its valley macro-F1 at 0.6562 against 0.6562, and the 2B HateMM arm finds one mode and no valley exactly as before.

## Disconfirmer verdicts

**D1 — smooth capability scaling. Does not fire.** The rule required the 4B trough to reach 0.7 times the 8B's on *both* corpora. It reaches 0.5985 times on ImpliHateVid and 1.2343 times on HateMM. The conjunction fails, so the reading is not collapsed to smooth scaling, but the two corpora fail it in opposite directions and that is the more informative fact. On ImpliHateVid trough depth rises monotonically with scale, 0.078 to 0.362 to 0.604. On HateMM it does not rise at all past 4B: the 4B cuts a deeper trough, 0.350 against 0.283, than the model twice its size. What is monotone across both corpora is not depth but existence. The 2B has one mode on HateMM and a trough of 0.078 on ImpliHateVid, which is a valley in name only; both larger scales have two modes and a trough above 0.28 everywhere.

**D2 — construct absent at 4B. Does not fire.** The 4B probe reaches 0.9377 against the 8B's 0.9480 on ImpliHateVid, a deficit of 0.0103, and 0.8839 against 0.8717 on HateMM, where it is 0.0122 *above* the 8B. Neither approaches the 0.05 margin. The construct-present half stands. It stands at 2B as well, which D2 did not govern: the 2B probe reads 0.9214 on ImpliHateVid, 0.0266 below the 8B, and 0.8251 on HateMM, 0.0466 below. The smallest model carries a linearly decodable hatefulness construct on both corpora and cannot be thresholded on either.

**D3 — angular rotation as the mechanism. Fires.** The rule required the ascending rank order of cosine spread to match the ascending rank order of trough depth on both corpora. On HateMM they agree, 2B then 8B then 4B in both. On ImpliHateVid they disagree: cosine spread orders the scales 2B, 8B, 4B while trough depth orders them 2B, 4B, 8B. The 4B has the widest angular spread of the three on both corpora, 0.0484 and 0.0466, above the 8B's 0.0332 and 0.0278, and yet on ImpliHateVid it cuts a trough only 60 percent as deep. Angular commitment therefore does not explain how deep the trough is. The pre-registration is explicit about what follows: the trough ordering is reported as an observation with no mechanism attached.

One descriptive remark, flagged as post-hoc and carrying no weight: the coarse two-group version of the angular claim is untouched. The 2B's cosine spread is 6.4 to 12.4 times narrower than the two larger models' and it is the only arm that fails to produce a usable valley. What died is the graded version, that more rotation means a deeper trough. Between 4B and 8B more rotation goes with a shallower trough on one corpus of two.

## The dissociation

What transfers down the scale ladder and what does not, stated as the six cells license and no further.

**Transfers.** The construct. A linear probe on the mid-stack residual stream reads hatefulness at 0.92 and 0.83 on the 2B against 0.95 and 0.87 on the 8B. Ranking also transfers: the 2B's raw logit contrast orders hateful against normal at 0.921 and 0.893, within 0.03 of the 8B on both corpora. A 2B judge is a usable ranker and its internals know what the 8B's internals know.

**Does not transfer.** Thresholdability. The 2B's scores occupy a range of 4.5 and 5.5 logits with a standard deviation near 1.0, against 38 to 40 logits and a standard deviation near 12 at the two larger scales. On HateMM that distribution has a single mode and the label-free recipe returns nothing. On ImpliHateVid it has two modes separated by a trough of 0.078, so the recipe returns a number, and the number costs 0.074 macro-F1 against the 8B's operating point.

The two halves are independent measurements of the same six runs, and they point opposite ways. Everything a supervised user of a small judge would want is present; everything a label-free user needs is absent. That is the finding.

A second dissociation runs the other way and is worth recording because it cuts against the obvious reading. On HateMM the 4B is the *better* label-free detector: 0.7904 macro-F1 at its own valley against the 8B's 0.6562, a gap to oracle of 0.071 against 0.232. The 8B ranks HateMM better, 0.9230 against 0.9120, and thresholds it far worse. Ranking quality and threshold quality are not the same axis at any scale, and the corpus where the 8B's valley diverges most from the class boundary is the one already documented as having a normal class saturated with hate-adjacent surface features.

## What this means for small-guardrail deployment

Ranking is cheap and label-free thresholdability is not. A 2B judge on a single forward pass will hand a moderation queue an ordering within 0.03 AUC of a model four times its size, at roughly a quarter of the memory, and that ordering is fit for triage where a human reads down the list. The moment the deployment needs an automatic accept-or-flag decision without labels, the 2B stops being a smaller version of the 8B and becomes a different instrument: on one of the two corpora here there is no distribution feature for a threshold rule to find, and the rule returns nothing rather than returning something wrong. That failure is at least honest, and a deployment should treat "the recipe found no valley" as the operational signal it is.

The corollary for anyone tempted to fix this by rescaling is that no rescaling can work. Relative trough depth is invariant to affine transformations of the score, so temperature, calibration constants and logit scaling all leave it exactly where it is. The 2B's problem is not that its scores are small. It is that they are unimodal.

The corollary for scale selection is narrower than "use a bigger model". Between 4B and 8B the label-free operating point does not improve reliably: it improves on ImpliHateVid and degrades on HateMM. If the reason for reaching past 2B is thresholdability rather than ranking, the evidence here supports 4B and does not support 8B over 4B.

## Limitations

- One model family, three scales, two corpora, six cells. Qwen3-VL is a single pretraining and post-training recipe and nothing here separates its scale ladder from its recipe.
- The pre-registered non-instruct control could not be run: no base Qwen3-VL checkpoint is published, only Instruct, Thinking, FP8 and GGUF variants. The declared substitute, Qwen3-VL-8B-Thinking, was checked and rejected before it was scored: its chat template appends `<think>\n` to the generation prompt, so the final prompt position is the opening of a reasoning block rather than the Yes-or-No answer position, and its `z` would not be the same measurement. The origin of the geometry, pretraining scale against post-training recipe, is therefore untested.
- Depth and width move together across these checkpoints and the two are not separated. The 2B has 28 layers and 2048 dimensions; the 4B and the 8B both have 36 layers, at 2560 and 4096. The 4B against 8B contrast is width at fixed depth; the 2B against the rest is both at once.
- The unembedding geometry is not constant across the ladder. The answer direction has norm 1.545 on the 2B, 1.097 on the 4B and 1.654 on the 8B, so the 4B's is a third shorter than either neighbour. Cosine is scale-invariant and unaffected, but `sd(z)` and the raw range are not, and the 4B's ordinary-looking dynamic range is produced against a shorter direction.
- No causal intervention. Nothing here manipulates angular spread and observes the trough. D3 rests on a rank comparison over three points per corpus, which is why it can disconfirm the graded mechanism and cannot confirm any replacement.
- The probe layer is fixed by rule at three-quarters depth and no layer sweep was run. A different depth could change the probe column, though it would have to change it by more than 0.05 to move D2.
- The label-free threshold is transductive: it reads the whole unlabeled test batch at once. A per-video online decision would need the threshold carried over, and the 2B's failure mode would then surface as an unstable carried threshold rather than as an absent one.
- Two corpora is enough to show that trough depth is not monotone in scale and not enough to characterise when it is.
