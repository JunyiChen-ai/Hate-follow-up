# Provisional novelty check: joint tri-modal product orbit (run07)

Cutoff 2026-08-29; performance-blind. This self-review follows a mandatory implementation audit but is **not** a fresh external/fresh-agent verdict and is not method acceptance.

## Blunt verdict

| Scope | Score /10 | >=6? |
|---|---:|:---:|
| M1 witness before transport | **6.2** | Yes, provisionally |
| M1 realized after current midrank transport | **4.4** | No |
| M2 hierarchical log-mean-exp propensity | **2.8** | No |
| M3 modality-balanced penalized scan | **2.2** | No |
| Integration | **4.9** | No |
| **Overall current chain** | **4.7** | **No** |

The `C_n × C_n` witness is a real conceptual upgrade: it uses one complete product-group orbit, one maximum family over all three pairs and all temporal positions, identity inclusion, conservative inclusive upper tails and a null-player axiom. This narrowly clears 6 as a standalone task-specific mechanism because no exact hateful-video-localization precedent was found.

It does not make the current end-to-end chain reach 6. The downstream code independently midranks each pair field. Since `0.5 - p(x)` is monotone in the pointwise conjunction `x`, this operation removes p-value magnitude and the common-reference cross-pair scale; only within-pair ordering and reference-induced tie partitions remain. The explicit novelty penalty is **−1.8 points** (`6.2 → 4.4`).

## Implementation gate

Status: **PASS WITH WARNINGS**.

- 611 unique aligned keys through witness, transport, free energy, balanced scan and fixed-4fps final.
- Stored total product-orbit states `214,899,052`; maximum video orbit `11,009,124`; both equal the independently summed/maximized `n²` values.
- Independent brute-force product-orbit reconstruction for `n=3,4,5,7` exactly matches the optimized reference.
- All 627,172 non-null field samples lie on an inclusive integer-count `/n²` grid (maximum numerical grid error `4.66e-10`).
- All 546 null-player pair/video cases are exactly zero.
- Exact transport preserves the nominal logit multiset within `1.11e-15` and its mean within `4.44e-16`.
- Recomputed metrics and strict-common comparison agree within `2.22e-16`.
- The downstream M2/M3 warnings from run06 remain: 14 dense-only chunk fallbacks; M2 is algebraically nested log-mean-exp; M3 ignores M1 interactions and forces one interval even for 582/611 negative best scan scores.

## Calibration-erasure audit

- There are 1,287 non-null pair/video witness fields, but only 268 vary after the joint maximum reference; 1,019 collapse to a constant.
- Across every frame/pair, the smallest stored p-value is `0.233918`; no p-value is at most `.10` or `.05`.
- The median per-video minimum p-value is `0.752688`.
- Current midranking therefore turns descriptive within-video differences among generally weak certificates into rank coordinates. It must not be described as preserving adjusted evidence strength or downstream maxT validity.

## Strongest rejection

The paper's most distinctive statistical object is dead-ended by its own fusion: pairwise midranking converts the joint orbit p-fields into essentially raw-conjunction rank templates and discards their calibrated magnitudes and shared family. The next module is ordinary two-level log-mean-exp pooling, and the interval decoder does not consume M1 at all; it is a classical, uncalibrated, forced-nonempty epidemic scan over direct modality ranks. Thus the actual localization pipeline is still a composition of recognizable operators, even though its unused/intermediately stored witness is interesting.

## Is a certificate-preserving transport sufficient?

**Necessary, but not sufficient for overall >=6.** Replacing the transport alone would likely raise the realized M1 contribution to roughly 6, but the full chain would remain around `5.4` because M2 is standard pooling and M3 is both classical and statistically disconnected from M1. The minimum credible paradigm upgrade is one certificate-preserving operator that uses the same joint product-orbit evidence for both dense projection and interval selection; alternatively, claim only M1 as the contribution and demote M2/M3 to implementation details.

## Closest primary foundations

- Westfall & Young, *Resampling-Based Multiple Testing* (1993): https://www.wiley-vch.de/de/fachgebiete/mathematik-und-statistik/resampling-based-multiple-testing-978-0-471-55761-6
- Hemerik & Goeman, group-invariance randomization tests (TEST 2018): https://link.springer.com/article/10.1007/s11749-017-0571-1
- Harris, shift tests for dependent time series (2020): https://arxiv.org/abs/2012.06862
- Yuan & Shou, truncated time-shift testing (PLOS Biology 2024): https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002758
- Schefzik et al., ECC rank rearrangement (Statistical Science 2013): https://projecteuclid.org/journals/statistical-science/volume-28/issue-4/Uncertainty-Quantification-in-Complex-Simulation-Models-Using-Ensemble-Copula-Coupling/10.1214/13-STS443.full
- Walther & Perry, scan calibration (JRSS B 2022): https://academic.oup.com/jrsssb/article/84/5/1608/7072948
- LELA, exact training-free hate-localization task (2026): https://arxiv.org/abs/2602.09637
- MultiHateLoc (2026): https://arxiv.org/abs/2512.10408
- Memory Matters (CVPR 2026): https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Memory_Matters_Boosting_Training-Free_Zero-Shot_Temporal_Action_Localization_with_a_CVPR_2026_paper.html

`review_independence=self-audit`; `acceptance_status=provisional_pending_fresh_reviewer`.
