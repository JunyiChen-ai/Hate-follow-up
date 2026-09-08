# Pre-registration — Generic-harm nuisance probe

**Frozen:** 2026-08-08 before computing any general-purpose visual-encoder
features.  
**Development data:** HateMM `train_clean` (primary) and ImpliHateVid
`train_clean` (non-inferiority control). Held-out test scores are not read by
this probe.  
**Purpose:** gate a novel nuisance-orthogonalization method aimed at high-score
generic-harm false positives.

## Candidate mechanism

The holistic moderator may use visually salient generic harm (fights, weapons,
riots, injury, threatening scenes) as a shortcut for policy-specific hate.
Generic harm is valid evidence about video danger but is a nuisance for the
narrow hate label. A frozen general-purpose vision-language encoder can measure
that nuisance independently of protected-group vocabulary. Removing only the
*population-level linear component* of raw hate logit explained by generic
visual harm may expose the policy-specific residual without asking the MLLM a
second moderation question.

This differs from modality deletion: every frame and transcript still reaches
the unchanged holistic judge. The auxiliary encoder adds an explicit nuisance
measurement; it does not classify hate.

## Frozen nuisance feature

- Encoder: `google/siglip-base-patch16-224`, frozen.
- Frames: the repository's same 16 uniformly sampled frames per video.
- Positive concept text, exactly:
  `a video showing physical violence, fighting, weapons, riots, injury, or threatening behavior`
- Negative concept text, exactly:
  `a peaceful video with no physical conflict, weapons, injury, or threatening behavior`
- Per frame: two-text softmax probability for the positive concept.
- Per video primary feature `g`: maximum frame probability. Mean and top-four
  mean are reported as secondary diagnostics and cannot replace `g`.

No identity terms, slurs, hate examples, dataset labels, or moderation scores
enter the encoder.

## Frozen label-free residual

Within each unlabeled corpus, robustly standardize `g` by median and MAD. Fit
ordinary least squares using all videos, without labels:

`z = alpha + beta * g_std + epsilon`.

The candidate score is:

`r = z - max(beta, 0) * g_std`.

The nonnegative clamp encodes the mechanism: only a positive generic-harm
shortcut is removable. Apply the repository's frozen KDE-valley recipe to `r`.
No coefficient, threshold, feature, or concept text is selected by labels.

## Frozen signal rule

Labels are revealed only after `g`, `beta`, `r`, and the KDE valley are frozen.
The mechanism **PASSES** only if all hold:

1. On HateMM, AUC of `g` for original-valley FP over TP is at least 0.65.
2. HateMM full-corpus hateful-vs-normal AUC improves by at least 0.03 from `z`
   to `r`.
3. HateMM KDE-valley macro-F1 improves by at least 0.10.
4. At the original HateMM valley, at least 25% of FPs flip down under the
   residual valley while no more than 10% of TPs flip down.
5. ImpliHateVid AUC and KDE-valley macro-F1 each decline by no more than 0.02.
6. A shuffled-`g` placebo (seed 20260808, same formula) achieves less than half
   the HateMM AUC gain of the real feature.

Report 2,000-draw bootstrap intervals for AUC and macro-F1 differences.

## Next-step map tied to the method goal

- **Pass:** implement Generic-Harm Orthogonalized Moderation as the full
  single-MLLM-call candidate and run frozen held-out tests.
- **`g` separates FP/TP but residualization fails:** the signal is real but the
  linear removal is wrong; next method version uses a preregistered monotone 2-D
  density boundary, with the same frozen feature and no feature search.
- **`g` does not separate FP/TP:** close visual generic-harm nuisance and move
  immediately to the next information-adding candidate, non-lexical delivery
  restoration from frozen general-purpose audio/prosody embeddings.
