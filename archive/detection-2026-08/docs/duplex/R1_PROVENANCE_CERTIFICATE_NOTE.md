# R1 Provenance Certificate Kill-Test — Result

**Date:** 2026-08-25  
**Preregistration:** `docs/duplex/PREREG_r1_provenance_certificate.md`  
**Model:** frozen Qwen3-VL-8B-Instruct  
**Views:** full, payload-only, no-payload, cyclic frame/source shift, matched nuisance  
**Verdict:** **KILL**

## Results

| Corpus | n (asserted/mention) | Original AUC | Context | Binding | Deletion | Certificate | Residual certificate |
|---|---:|---:|---:|---:|---:|---:|---:|
| HateMM | 28 (17/11) | 0.655 | 0.452 | 0.564 | 0.527 | 0.586 | **0.540** |
| MHC-ZH | 12 (7/5) | 0.029 | 0.357 | 0.400 | 0.371 | 0.386 | **0.600** |

The frozen survival rule required residual certificate AUROC >= 0.70 on both corpora and improvement over deletion-only. The certificate beats deletion-only numerically on both, but misses the primary threshold on both; therefore the idea dies before adapter or boundary experiments.

## Interpretation

The interventions measure payload/context sensitivity and some visual-order sensitivity, but do not recover a stable signed author-commitment variable. In particular, removing context or cyclically shifting frames does not consistently distinguish quoted/reported/countered hostility from asserted/endorsed hostility. This matches the project's prior speech-act and illocution-transfer failures rather than overturning them.

No prompt, threshold, certificate formula, cohort or view was changed after observing results. R2 write-gating and the typed boundary decoder are not run.

## Artifacts

- `results/r1_provenance_certificate/scores.jsonl`
- `results/r1_provenance_certificate/hidden/`
- `results/r1_provenance_certificate/report.json`
- `results/r1_provenance_certificate/logs/full.log`
- `scripts/duplex/r1_provenance_certificate.py`
