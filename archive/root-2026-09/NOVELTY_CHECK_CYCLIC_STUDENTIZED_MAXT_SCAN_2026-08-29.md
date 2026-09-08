# Fresh novelty check: cyclic studentized maxT + scan (run05)

`review_independence=same-family`; `acceptance_status=provisional`; cutoff 2026-08-29; performance-blind.

## Verdict

The authoritative null-player v2 implementation passes the mandatory implementation-integrity gate **with warnings**. This is not method acceptance. The conceptual method does **not** reach the required novelty threshold.

| Component | Novelty /10 | >=6? |
|---|---:|:---:|
| M2: scale-prepivoted cyclic pair field | 5.5 | No |
| M3: penalized interval scan | 2.4 | No |
| Task | 2.5 | No |
| Mechanism | 4.9 | No |
| Integration | 5.5 | No |
| **Overall** | **4.8** | **No** |

The defensible contribution is a pair-specific, scale-prepivoted cyclic-max-referenced empirical midrank field fused by rank-based residual rearrangement, followed by an uncalibrated placement-penalized interval scan. It is an uncommon composition, not a new statistically referenced localization paradigm.

## Integrity gate

- 611 unique aligned rows across the authoritative chain.
- Null-player audit passes: 546 constant-modality pair/video cases and zero nonzero interaction fields.
- Independent reconstruction of the smallest all-nonconstant video (`n=19`) matches all three pair fields.
- Transport preserves the actual single-L-curve nominal centered-logit residual multiset (maximum absolute discrepancy `1.1e-15`); no three-way Self-Risk barycenter is credited.
- Propensity preserves the residual multiset within `1.4e-11`; M3 leaves dense scores unchanged.
- Thirty independently recomputed M3 endpoints match; final lengths and interval clips are legal.
- Warnings: the cyclic output is descriptive rather than an adjusted p-value; pair-specific calibration does not survive fusion; M3 has no variance/dependence calibration; the propensity rule is development/label selected, so empirical evidence is exploratory.

## Core claims

| Claim | Rating | Reason |
|---|---|---|
| Scale-prepivoted cyclic pair field | MEDIUM | Uncommon in this task, but group randomization, prepivoting and max statistics are established. |
| Equal-rank residual transport | LOW | Exact multiset preservation is useful; external-template rank rearrangement is mature, including ECC. |
| Penalized interval localization | LOW | Classical changed/epidemic scan with an unnormalized placement penalty. |
| Label-free hateful-video localization | LOW | Directly preceded by LELA and adjacent weak/training-free localization work. |
| End-to-end integration | MEDIUM | No exact tuple found, but it composes occupied mechanisms. |

## Strongest rejection

LELA already occupies training-free multimodal hateful-video localization. M2 combines complete-group cyclic randomization, empirical-CDF prepivoting and maximum statistics, but its mid-CDF outputs are not conservative adjusted p-values, the references are pair-specific, and downstream rank fusion discards calibration. Residual transport resembles ECC-style rank rearrangement. M3 uses a Gaussian-looking scan penalty without Gaussianity, variance normalization, long-run-variance control or dependence calibration. The null-player branch repairs implementation sanity but is not novelty. The result is an engineering integration of known operations, not a new paradigm.

## One substantive refinement

Replace the three pair-specific reference banks with one conservative complete `C_n^2` tri-modal relative-phase orbit reference: fix visual phase, independently shift language and audio, maximize jointly across all three pairs, positions and scales, and use inclusive upper-tail orbit counts. This is computationally demanding but directly targets the pairwise-composition objection while preserving the empirically useful residual transport.

## Closest primary work

- Westfall & Young, *Resampling-Based Multiple Testing* (1993): https://www.wiley-vch.de/de/fachgebiete/mathematik-und-statistik/resampling-based-multiple-testing-978-0-471-55761-6
- Hemerik & Goeman, group-invariance permutation tests (TEST 2018): https://link.springer.com/article/10.1007/s11749-017-0571-1
- Beran, “Prepivoting Test Statistics” (JASA 1988): https://doi.org/10.1080/01621459.1988.10478649
- Harris, “A Shift Test for Independence” (2020): https://arxiv.org/abs/2012.06862
- Yuan & Shou, “Truncated Time-Shift Test” (PLOS Biology 2024): https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002758
- Walther & Perry, “Calibrating the Scan Statistic” (JRSS B 2022): https://academic.oup.com/jrsssb/article/84/5/1608/7072948
- König, Munk & Werner, multiscale nuisance calibration (JRSS B 2025): https://academic.oup.com/jrsssb/article/87/2/510/7829180
- Schefzik, Thorarinsdottir & Gneiting, ECC (Statistical Science 2013): https://projecteuclid.org/journals/statistical-science/volume-28/issue-4/Uncertainty-Quantification-in-Complex-Simulation-Models-Using-Ensemble-Copula-Coupling/10.1214/13-STS443.full
- LELA (2026): https://arxiv.org/abs/2602.09637
- MultiHateLoc (2026): https://arxiv.org/abs/2512.10408
- T3AL (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/html/Liberatori_Test-Time_Zero-Shot_Temporal_Action_Localization_CVPR_2024_paper.html
- Memory Matters (CVPR 2026): https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Memory_Matters_Boosting_Training-Free_Zero-Shot_Temporal_Action_Localization_with_a_CVPR_2026_paper.html

The full reviewer response, implementation audit, search log and provenance are stored in `.aris/traces/novelty-check/2026-08-29_run05/`.
