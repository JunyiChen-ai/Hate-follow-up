# Experiment Code Review — Paradigm Adaptation

Final deployment verdicts: P1–P7 runners approved after blocker remediation.

Material fixes enforced during review:

- removed label-skewed cohort construction and all inference-time GT access;
- exported ASR through a strict inference-only allowlist;
- aligned visual frames, ASR units, proposal curves, and dense outputs on explicit time grids;
- isolated T3AL and Poset canvas inputs and recorded true model calls;
- added config identity, source hashes, exact coverage, duplicate rejection, and fail-closed resume behavior;
- added AV1 FFmpeg fallback plus batch black-frame detection;
- split English by timed words and CJK by timed characters without copying whole chunks into zoom windows;
- used official `t=i/4` mapping for 4 FPS curves and intervals;
- replaced tie-unsafe ECDF ranking with average-rank ties and neutral constant curves;
- reran TimeLens-8B on the exact clean cohort with auditable model/runner/manifest provenance;
- made cross-video memory prequential, dataset-isolated, paired with a no-memory control, and order-audited.

Known scope limitations:

- these are mechanism pilots, not final benchmark estimates;
- P6 uses an external temporally supervised/RLVR checkpoint and is label-free only with respect to target inference;
- P6 is post-selection exploratory on this split;
- P7 has only two videos per dataset in Stage-A, hence at most two distinct orders;
- MHC_zh subsets used here contain no positive frames.
