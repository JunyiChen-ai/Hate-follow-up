# Frozen preregistration: CCA on HateMM validation_clean

Frozen before any CCA/A08/A10/A12/text prediction or temporal-ground-truth
evaluation is produced on this cohort.

## Cohort

- All 107 IDs in `/home/jehc223/data/HateMM/splits/validation_clean.csv`.
- Exclude only missing media, missing ASR duration, duplicates, or overlap with
  the already-developed HateMM test cohort; every exclusion must be reported.
- Prediction dataset alias: `HateMM_val_sealed`.
- Filenames contain upstream video-level class words; neither that string nor
  any video/temporal label is an inference input or selection criterion.

## Frozen method

1. A10/Vid-Group with the generic hateful-content query and CLIP-L/14 128-bin
   features; diverse top-8 export uses temporal NMS tIoU=.5.
2. Connected top-8 support component containing the rank-1 center owns temporal
   boundaries.
3. A08/VideoTGB and A12/TimeLens provide visual consensus/disagreement states.
4. Qwen3-VL-8B transcript Yes-minus-No logit uses one explicit empty transcript
   as its content-free null, batch size 1 and left padding.
5. CCA audits only consensus samples on which text proposes a veto.  With a
   Beta(1,1) prior, text is authorized in disagreement iff
   `P(veto reliability > .5 | consensus audit) > .95`.  If unauthorized,
   disagreements use visual OR.  Text cannot create or move boundaries.

## Frozen comparisons and metrics

- A10 rank-1, top-8 extent, visual OR, VASTA, and CCA on exactly the common IDs.
- Primary: interval F1@.5. Secondary: F1@.3/.7, pooled ROC/PR, within-video ROC.
- Report consensus audit counts, posterior authorization probability, candidate
  veto correct/wrong/net, and selected-arm regret against max(VASTA, visual OR).

## Acceptance

CCA supports the authority-transfer mechanism only if:

1. its authorization decision is made before opening temporal GT;
2. if authorized, candidate disagreement veto net is positive; if denied, net
   is non-positive;
3. CCA F1@.5 is no worse than `max(VASTA, visual OR) - .002`;
4. CCA does not reduce F1@.3 or F1@.7 by more than .01 relative to that best arm;
5. the prediction and GT cohorts are complete and ID-identical.

No threshold, prompt, cohort membership, NMS setting, or fallback action may be
changed after predictions begin.
