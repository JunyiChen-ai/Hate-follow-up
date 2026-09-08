# Fresh Novelty Review: Local Multiscale Max-Statistic Synergy

- review date: 2026-08-29
- literature cutoff: 2026-08-29, including a targeted scan of the most recent six months
- review_independence: **same-family**
- acceptance_status: **provisional**
- performance_blind: **true**
- implementation_integrity_gate: **PASS WITH WARNINGS — the deterministic computation and artifact chain match the description; no calibration theorem is established**
- empirical_confirmation_status: **exploratory until untouched confirmation**
- novelty threshold: **6.0/10**
- final verdict: **M2 = 5.2/10; M3 = 2.0/10; overall = 4.6/10; none reaches 6**

## Blunt verdict

**不过 6。M2 不过 6，M3 不过 6，整体也不过 6。** M2 genuinely implements a nonwrapping, time-local, multiscale lag-max-referenced construction, so it is a meaningful conceptual improvement over the discarded circular global-authority variant. Its exact tuple is uncommon. It still combines familiar windowed scans, time-shift surrogates, max-statistic references, rank fusion, and ECC-like rank rearrangement rather than establishing a new localization principle.

The decisive problem is statistical, not semantic. For scale width `w`, nonzero lags contribute maxima over unequal numbers of windows, ranging from 1 to `n-w`. These shifts are not an exchangeable transformation group on a common sample space, zero lag is omitted from the empirical reference, and the maximum is not joint over time, scales, and modality pairs. Therefore the mid-CDF is a bounded empirical lag-max-reference score, not a finite-sample p-value, simultaneous calibration, or FWER guarantee. The fixed widths `{1, round(sqrt(n)), round(n^(2/3))}` are heuristic design choices. M3 remains a routine max-plus-connected-component decoder and contributes little mechanism novelty.

Performance numbers were excluded from novelty scoring. The disclosed label/development-selected propensity rule makes the present empirical evidence exploratory, which is a separate validity judgment rather than a novelty penalty.

## Scores

| Dimension | Score | Verdict |
|---|---:|---|
| M1 Self-Risk Multiresolution Posterior | 4.0/10 | LOW-MEDIUM; symmetric assembly of classical per-video risk estimators |
| M2 Local Multiscale Max-Statistic Rank Transport | **5.2/10** | MEDIUM — **NO >=6** |
| M3 Local-Maxstat Union Boundary | **2.0/10** | LOW — **NO >=6** |
| Task novelty | **2.8/10** | LOW; training-free and weakly supervised hate localization already exist |
| Mechanism novelty | **4.8/10** | MEDIUM-LOW |
| Integration novelty | **5.4/10** | MEDIUM; exact tuple is uncommon but components are close to prior art |
| **Overall novelty** | **4.6/10** | **MEDIUM-LOW — FAILS >=6** |

## Core claims

1. **Dense nonwrapping local fields referenced to lagwise maxima — MEDIUM.** The exact ordering of positive conjunction, moving-window scans, nonzero-lag maxima, empirical reference mapping, and equal-scale averaging is uncommon in multimodal hate localization. Local scan and shift-surrogate/max-statistic comparison are nevertheless established.

2. **Finite-sample simultaneous temporal calibration — LOW / unsupported.** Unequal overlap and window counts make the lag maxima non-identically distributed. The transformations are not exchangeable on a common support; the reference excludes zero lag and does not jointly maximize over scales and pairs. Safe wording is “lag-max-referenced empirical score,” not “randomization p-value,” “distribution-free,” “FWER-controlled,” or “simultaneously calibrated.”

3. **Adaptive multiscale localization — LOW.** The three widths are deterministic functions of `n`, but their exponents are not derived from M1 risk selection, a duration model, a detection boundary, or a theorem. They are a fixed heuristic temporal pyramid and retain development-selection risk.

4. **Equal-midrank consensus plus exact residual order-statistic transport — LOW-MEDIUM.** The symmetry is clean and label-free at inference, but rank averaging is conventional and exact marginal-preserving rearrangement has close empirical-copula/ECC prior art.

5. **Certificate boundary and overall paradigm — LOW for M3; MEDIUM-LOW overall.** `nominal + max(main) + max(pair)` followed by the positive connected component containing the global maximum is an ordinary peak superlevel-set decoder. Parameter removal does not itself create a new boundary principle.

## Mandatory implementation-integrity gate

### PASS WITH WARNINGS for the implemented heuristic

- `run_local_maxstat_synergy_fields.py` uses nonwrapping moving means at three per-video widths and computes the three modality-pair fields. It does not use circular wrap.
- For each nonzero lag with overlap at least `w`, it computes the temporal maximum of the local conjunction statistic before mapping zero-lag local values through a tie-aware empirical mid-CDF.
- All 611 videos have three distinct scales after the cap.
- The production downstream chain emits one rank-transport method, one honestly tagged locked propensity rule, one boundary rule, and one final fixed-4-fps method.
- Rebuilding the current field -> transport -> propensity -> boundary -> fixed-grid chain reproduces the 611-video outputs; independent metric recomputation matches the stored values up to final floating-point serialization digits.
- The exact preservation claim is correctly limited to the nominal centered-logit residual multiset at M2, not the final probability marginal after propensity replacement.

### Warnings that limit claims

1. **No theorem-level simultaneous calibration.** At width `w`, lag maxima are based on different numbers of windows. The mid-CDF is not an exact randomization distribution.
2. **No joint maxT across scales or pairs.** Averaging three separately referenced fields does not control multiplicity over time x scale x modality pair.
3. **Heuristic widths.** `1`, `sqrt(n)`, and `n^(2/3)` were not justified prospectively or theoretically.
4. **Exact-float tie sensitivity.** Approximately `1e-17` support perturbations can cross empirical-CDF bins; an audit reconstruction observed per-scale field changes as large as about `0.049`. The chain is deterministic on fixed inputs but not numerically robust near ties.
5. **Exploratory evidence.** The locked propensity rule records `development_selected_rule=true` and `label_selected_rule=true`; the 611-video cohort is not untouched confirmation.

This gate verifies faithful implementation of the stated heuristic. It is **not method acceptance** and does not validate calibration language.

## Closest primary work

| Work | Year / venue | Main overlap | Material difference |
|---|---|---|---|
| [Harris, A Shift Test for Independence in Generic Time Series](https://arxiv.org/abs/2012.06862) | 2020, arXiv | Ranks an unshifted association statistic against shifted versions under stationarity | Candidate creates dense local fields but does not inherit Harris's guarantee |
| [Yuan & Shou, A Rigorous and Versatile Statistical Test for Correlations Between Stationary Time Series](https://pubmed.ncbi.nlm.nih.gov/39146390/) | 2024, PLOS Biology | Truncated time-shift inference with false-positive control | Candidate uses unequal-overlap lag statistics rather than a fixed common construction |
| [Walther & Perry, Calibrating the Scan Statistic](https://arxiv.org/abs/2008.06136) | 2022, Electronic Journal of Statistics | Window maxima, scale calibration, and permutation/rank/sign scans | Candidate averages separately referenced heuristic scales without joint calibration |
| [Arias-Castro et al., Distribution-Free Detection of Structured Anomalies](https://arxiv.org/abs/1508.03002) | 2018, Annals of Statistics | Permutation and rank-based scan statistics for structured anomaly detection | Candidate specializes evidence to multimodal ordinal conjunction |
| [Koenig, Munk & Werner, Multiscale Scanning with Nuisance Parameters](https://academic.oup.com/jrsssb/article/87/2/510/7829180) | 2025, JRSSB | Multiscale local scans with explicit multiplicity/FWER treatment | Candidate lacks scale calibration and an invariance theorem |
| [Schefzik et al., Ensemble Copula Coupling](https://arxiv.org/abs/1302.7149) | 2013, Statistical Science | Reorders a target marginal according to a supplied rank dependence template | Candidate applies this rank-rearrangement pattern to temporal multimodal evidence |
| [MultiHateLoc](https://arxiv.org/abs/2512.10408) | 2026, WWW | Multimodal temporal hate localization and cross-modal fusion | Weakly supervised and learned rather than frozen label-free inference |
| [LELA](https://arxiv.org/abs/2602.09637) | 2026, arXiv | Training-free multimodal hateful-video localization with frame scores | Uses LLM prompting/composition matching rather than shift-reference fields |

Adjacent temporal-localization work further narrows task novelty: [T3AL](https://openaccess.thecvf.com/content/CVPR2024/html/Liberatori_Test-Time_Zero-Shot_Temporal_Action_Localization_CVPR_2024_paper.html), [Moment-GPT](https://arxiv.org/abs/2501.07972), [Memory Matters](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Memory_Matters_Boosting_Training-Free_Zero-Shot_Temporal_Action_Localization_with_a_CVPR_2026_paper.html), and recent [OZ-TAL](https://arxiv.org/abs/2605.09976). Earlier windowed shift-surrogate synchrony is explicit in [Moulder et al.](https://pubmed.ncbi.nlm.nih.gov/29595296/).

## Strongest reviewer rejection

The paper repackages established maxT logic—comparing observed local statistics with surrogate maxima—using a time-shift bank that lacks the common support and exchangeability needed for inferential meaning. It then averages uncalibrated heuristic scales and modality pairs, applies familiar rank/ECC transport, and ends with a routine peak-component rule. The exact implementation is task-specific new engineering, but neither a new localization principle nor a valid simultaneous-calibration method.

## Exactly one smallest implementable refinement

**Replace the variable-overlap lag bank with a pre-specified Yuan–Shou-style fixed central core.** Choose one prospective truncation radius and compute zero-lag and every admitted nonzero-lag statistic on identical-length segments with the same number of width-`w` windows. Keep M1, equal-scale fusion, rank transport, and M3 unchanged.

This single replacement removes the immediate lag-count bias and creates a plausible stationarity-based route for the shift reference. By itself it still would not establish fieldwise or multiscale FWER control, so the claim should remain a common-support shift-reference score unless a separate theorem is proved.

