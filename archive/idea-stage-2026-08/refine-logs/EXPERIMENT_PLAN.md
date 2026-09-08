# Experiment Plan: Poset T3AL Successor

## E0 — Reproduction Freeze

- Freeze prompt, margin `0.4`, visual budget `0.05`, topology weight and seeds.
- Re-run `evaluate_poset_t3al_successor.py`; require zero constraint violations/failures.

## E1 — Canonical Four-Dataset Coverage

- Produce transcript evidence and visual curves for every official test video.
- Report pooled ROC-AUC/PR, within-video macro ROC-AUC and interval metrics per dataset.
- Keep within-video macro ROC-AUC primary.

## E2 — Core Ablations

- repaired T3AL;
- fixed query + poset;
- ordinal query + poset;
- topology ordinal query + poset;
- end-to-end parity shuffle and reverse;
- direct transcript teacher;
- post-T3AL + poset negative control.

## E3 — Clean Model Selection

- LODO over HateMM, HateClipSeg, MHC and MHC-ZH, or a genuinely non-test validation split.
- Select prompt, margin, topology weight and visual budget without target-test labels.

## E4 — Robustness

- transcript dropout and timestamp jitter;
- confidence corruption and missing-speech fallback;
- chunk-duration sensitivity;
- at least three seeds where applicable.

## E5 — Secondary Boundaries

- frozen visual snapping only as a secondary module;
- dev-selected radius/smoothing and held evaluation;
- per-dataset interval-F1/tIoU with paired bootstrap.

## Promotion Gate

- zero qualified-pair violations;
- held/LODO main > fixed, ordinal, shuffle and reverse with paired CI above zero;
- positive direction on at least three datasets;
- no teacher-superiority claim unless its paired CI is above zero.
