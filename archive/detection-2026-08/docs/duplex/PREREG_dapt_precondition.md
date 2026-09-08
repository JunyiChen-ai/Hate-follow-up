# Pre-registration — DAPT precondition probe (familiarity account of HateMM mid-band congestion)

**Frozen:** 2026-08-10, before any perplexity is computed.
**Compute:** ~615 teacher-forced forwards (HateMM test 215 + ImpliHateVid
test 400), no training, ≈ minutes on the RTX 5090.
**Status:** diagnostic gating the DAPT (corpus-conditioned continued
pretraining) family. If it fails, DAPT is not run.

## Claim under test

HateMM's mid-band congestion (18% unsaturated videos filling the valley;
label-free threshold 0.6562 vs oracle 0.888) is partly a FAMILIARITY
artifact: the frozen judge parses this domain (BitChute-style content,
noisy speech, in-group slang) less well than familiar domains, and its
unsaturated middle contains disproportionately the videos it predicts
poorly. If the middle is not harder to predict than the poles, the
familiarity story is false and prediction training has no lever on the
congestion.

## Frozen protocol

- Inputs: identical frozen judge prompt (frames + title + c2 override
  transcript, per-corpus rules); one teacher-forced forward per video;
  per-token NLL computed ONLY over the transcript token span.
- Exclusion: videos with < 20 transcript tokens excluded from band
  statistics (count reported).
- Bands (frozen convention from the saturation pilot): saturated poles
  |z| ≥ 13; mid-band |z| < 13. z from the committed test scores.
- Statistics: per-video mean NLL. Link 1 (descriptive): HateMM median
  NLL vs ImpliHateVid median NLL. Link 2 (decisive, HateMM): AUC of
  per-video NLL separating mid-band from poles, with Mann-Whitney test.
- Control: same Link-2 statistic on ImpliHateVid (where the label-free
  operating point already works, 0.8823).

## Frozen decision rule

The DAPT family proceeds to a training preregistration only if:

1. **Link 2 holds on HateMM:** NLL separates mid-band from poles with
   AUC ≥ 0.60 and Mann-Whitney p < 0.05 (mid-band harder to predict).

Interpretation guards: if ImpliHateVid shows an equal or stronger
Link-2 effect (AUC within 0.03 of HateMM's or higher), the effect is
generic and does not specifically explain HateMM's congestion — proceed
only with this caveat recorded and a matched ImpliHateVid regression
guard in the training prereg. If Link 2 fails, the familiarity account
is dead: mid-band uncertainty reflects construct-boundary content, not
domain unfamiliarity, and corpus-conditioned prediction training has no
identified lever — DAPT is not run, recorded as a prevented-in-advance
death.

Reported descriptively either way: NLL distributions per band per
corpus; Spearman(NLL, |z|) per corpus; NLL vs transcript length check;
the 11 code-Q videos' NLL percentiles.
