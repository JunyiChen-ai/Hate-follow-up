# Fresh Novelty Review: Production Pair-Only Permutation Synergy v3

- review date: 2026-08-29
- literature cutoff: 2026-08-29, including a targeted scan of the most recent six months
- review_independence: **same-family**
- acceptance_status: **provisional**
- implementation_integrity_gate: **PASS — production v3 is single-path, honestly scoped, and byte-reproducible**
- empirical_confirmation_status: **exploratory until untouched confirmation**
- novelty threshold: **6.0/10**
- final verdict: **M2 = 4.5/10; M3 = 2.0/10; overall = 4.5/10; none reaches 6**

## Blunt verdict

**不过 6。M2 不过 6，M3 不过 6，整体也不过 6。** Production v3 removes the invalid `[0,1,2]` triple orbit, emits one transport, one locked propensity rule, and one boundary rule, and honestly records the development-selected status. This cleanup repairs implementation/provenance integrity but does not make the remaining pair mechanism new. For pair supports (a,b), the proposed orbit statistic is ordinary circular cross-correlation,

\[
C(s)=\frac1n\sum_t a_t b_{t+s},\qquad
\frac1n\sum_s C(s)=\bar a\bar b.
\]

The local field therefore reduces exactly to

\[
q\bigl(a_t b_t-\bar a\bar b\bigr),
\]

where (q) is the zero-lag circular-cross-correlation rank among shifts. This is a conventional centered pointwise conjunction scaled by a conventional shift-surrogate rank. A tied null receives (q=(n+1)/(2n)\approx0.5), so the so-called authority is not a null-suppressing gate.

Exact order-statistic reordering, average-rank consensus, and peak-connected-component decoding all have close prior art. The distinctive part is the label-free multimodal integration, not a new synergy estimator or decoder. Performance numbers were not used in the scores.

## Scores

| Dimension | Score | Verdict |
|---|---:|---|
| M1 Self-Risk Multiresolution Posterior | 4.0/10 | LOW-MEDIUM; classical per-video risk estimators assembled symmetrically |
| M2 Pair Permutation-Synergy Rank Transport | **4.5/10** | MEDIUM-LOW — **NO ≥6** |
| M3 Pair Union Boundary | **2.0/10** | LOW — **NO ≥6** |
| Task novelty | **2.0/10** | LOW |
| Mechanism novelty | **4.5/10** | MEDIUM-LOW |
| Integration novelty | **5.0/10** | MEDIUM |
| **Overall novelty** | **4.5/10** | **MEDIUM-LOW — FAILS ≥6** |

## Core claims

1. **Pair circular-orbit synergy — LOW.** Every pair enumerates all (n) relative circular phases, including for composite (n), but the statistic is exactly circular cross-correlation on ordinal supports. The local field is a scalar-weighted cross-moment decomposition, not a new interaction functional.

2. **Sample-adaptive orbit authority — LOW.** Ranking the unshifted statistic among time-shifted surrogates is established shift-test practice. The raw rank divided by (n) is never zero and gives a completely tied orbit roughly half authority; it is neither an exact evidence gate nor a calibrated local confidence field.

3. **Symmetric role consensus plus exact rank transport — MEDIUM.** Equal midrank fusion of nominal, three main, and three pair fields is a clean label-free integration. Yet exact permutation of one marginal according to a dependence template is the core operation of ECC/Schaake-style empirical-copula reordering. The candidate supplies an application-specific template, not new transport theory.

4. **Pair union boundary — LOW.** `nominal + max(main) + max(pair)` followed by the positive connected component containing the global maximum is a routine max/threshold/component decoder. Removing duration/top-k parameters does not create a new boundary principle.

5. **Overall hateful-video localization paradigm — MEDIUM-LOW.** LELA already occupies training-free multimodal hateful-video localization, while MultiHateLoc occupies multimodal hate localization under weak supervision. The proposed assembly is uncommon, but a new application and parameter-free composition do not reach 6.

## Mandatory implementation-integrity gate

### PASS for authoritative production v3

- `run_permutation_synergy_fields.py` computes only the three pair fields; triple code has been removed.
- `project_permutation_synergy_transport.py` emits only `permutation_pair_complete_rank_transport_v3`.
- `project_locked_equal_propensity.py` emits one frozen rule and explicitly tags `development_selected_rule=true`, `label_selected_rule=true`, and `eligible_evidence_status=exploratory_until_untouched_confirmation`.
- `project_permutation_synergy_boundary.py` emits only `permutation_pair_complete_union_boundary_v3`.
- Each pair uses multiplier `[1]`, so all (n) relative circular phases are enumerated exactly once for any video length.
- Current fields → pair transport → locked propensity → pair boundary → 4-fps contract rebuilds byte-for-byte for all 611 videos.
- The final prediction SHA-256 is `44fd6baea50f83d27ccaa4e081810ae8de1cda0a5b5cb2633f01b6839630e56d`.
- The pair boundary SHA-256 is `e77024d53ebb975b4f0dff7519570209e9a5dbe39ef56ebd371323023a2e865c`.
- Interval endpoints use a consistent half-open component, and the final crop synchronizes curves and intervals to the evaluator grid.
- Final metadata correctly narrows preservation to the centered-logit residual multiset and marks the final probability marginal as changed.

### Empirical/statistical qualifications that remain

1. **The locked rule has label-informed history.** Production v3 now discloses this correctly, but the 611-video results remain post-selection exploratory rather than untouched confirmation.
2. **Preservation must remain narrowly worded.** M2 exactly permutes the centered-logit residual multiset; the later propensity intercept changes the final probability multiset and mean. v3 metadata now states this correctly.
3. **Circular-null validity needs an assumption.** Circular shifts preserve empirical values and circular autocorrelation, but a randomization interpretation requires explicit stationarity/group-invariance assumptions. Wrap-around creates an artificial end-start adjacency.
4. **Discarded triple claims remain invalid history, not part of v3.** The old `[0,1,2]` design spans only a one-dimensional line, not the complete three-stream relative-phase null. Production code correctly removes it.
5. **No untouched confirmation exists.** Numeric results are real and reproducible, but performance cannot establish novelty and cannot yet support generalization claims.

Thus implementation integrity passes, while empirical confirmation remains exploratory. This novelty review is still **not method acceptance**.

## Closest primary work

| Work | Year / venue | Main overlap | Material difference |
|---|---|---|---|
| [Harris, A Shift Test for Independence in Generic Time Series](https://arxiv.org/abs/2012.06862) | 2020, arXiv | Ranks an unshifted association statistic among time-shifted values for autocorrelated series | Candidate turns that global rank into a multiplier for a local product field |
| [Yuan & Shou, A rigorous and versatile statistical test for correlations between stationary time series](https://pmc.ncbi.nlm.nih.gov/articles/PMC11398661/) | 2024, PLOS Biology | Time-shift surrogate testing, stationarity conditions, and circular-boundary caveats | Candidate uses the complete circular pair orbit without a formal test theorem |
| [Moulder et al., Determining Synchrony Between Behavioral Time Series](https://pubmed.ncbi.nlm.nih.gov/29595296/) | 2018, Psychological Methods | Surrogate destruction of alignment while preserving temporal/distributional properties; cross-correlation synchrony | Candidate applies ordinal conjunction to multimodal video evidence |
| [Liu et al., Kernel-based Joint Independence Tests for Multivariate Stationary and Non-stationary Time Series](https://arxiv.org/abs/2305.08529) | 2023, arXiv / journal version | Shift-resampled temporal dependence and higher-order multivariate independence | Uses independently shifted series and dHSIC, not the candidate's pair-product field |
| [Schefzik et al., Ensemble Copula Coupling](https://arxiv.org/abs/1302.7149) | 2013, Statistical Science | Exact order-statistic reordering using a rank-dependence template while preserving a marginal sample | Weather postprocessing; candidate uses a multimodal temporal template |
| [Hong et al., Cross-modal Consensus Network](https://arxiv.org/abs/2107.12589) | 2021, ACM Multimedia | Cross-modal consensus for temporal localization | Learned weakly supervised attention rather than per-video rank assembly |
| [MultiHateLoc](https://arxiv.org/abs/2512.10408) and [LELA](https://arxiv.org/abs/2602.09637) | 2026, WWW / arXiv | Exact multimodal hateful-video localization task; LELA is training-free | Different learned/prompting mechanisms, but they remove task-level novelty |
| [Nguyen et al., Weakly Supervised Action Localization by Sparse Temporal Pooling Network](https://openaccess.thecvf.com/content_cvpr_2018/html/Nguyen_Weakly_Supervised_Action_CVPR_2018_paper.html) | 2018, CVPR | Converts temporal activation fields into contiguous proposals | Learned action activations; candidate uses a zero-rank excursion and one peak component |

Required adjacent temporal work further narrows the task claim: [T3AL](https://openaccess.thecvf.com/content/CVPR2024/html/Liberatori_Test-Time_Zero-Shot_Temporal_Action_Localization_CVPR_2024_paper.html), [Moment-GPT](https://arxiv.org/abs/2501.07972), and [Memory Matters](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Memory_Matters_Boosting_Training-Free_Zero-Shot_Temporal_Action_Localization_with_a_CVPR_2026_paper.html). The recent-six-month scan also checked [SAGE](https://aclanthology.org/2026.acl-long.817/), which already frames hateful-video fusion around modality-specific experts and decision-level arbitration, though it is a detection rather than localization method.

## Strongest reviewer rejection

The alleged synergy mechanism collapses algebraically to a conventional zero-lag cross-product excess multiplied by its rank among circular cross-correlations. The multiplier remains about 0.5 under a completely tied null, so it is not an evidential gate. The remaining rank fusion, empirical-marginal reordering, and peak-component decoder closely follow established consensus, ECC, and temporal-localization practice. Since the task itself is no longer new and the final pipeline was chosen among alternatives using the evaluation labels, the contribution is an unconfirmed engineering assembly rather than a novel localization principle.

## Exactly one smallest refinement most likely to exceed 6

**Replace the single global pair-authority multiplier with one boundary-safe, time-local multiscale randomization field.** At each M1 scale and time point, rank a zero-lag windowed pair statistic against all admissible non-wrapping relative shifts; calibrate the entire temporal field with one orbit max-statistic envelope; then equal-fuse the signed local excesses across scales. Keep the observed rank-consensus transport and M3 interface unchanged.

This single module replacement would stop the mechanism from reducing to `global percentile × pointwise product`, make calibration localization-specific, remove circular wrap-around, and create room for a finite-sample invariance/error-control result. To plausibly cross 6, the paper would still need to state and prove the admissible-shift null, freeze the design before evaluation, and confirm it on untouched data.
