# Final Fresh Novelty Review: Shared Shapley Candidate

- review date: 2026-08-29
- review mode: fresh, adversarial, primary-source search through 2026-08-29
- review_independence: **same-family**
- acceptance_status: **provisional**
- implementation_integrity_gate: **FAIL**
- novelty threshold: **6.0/10**
- final verdict: **M3 = 3.0/10; overall = 2.5/10; neither reaches 6**

## Blunt verdict

**不过 6。M3 自身不过 6，整体也不过 6。** M2 的旧 pair-fit inversion 是一个足以单独阻止整体过 6 的核心风险，但它不是唯一阻断项：新 M3 的所谓完整 8-coalition game 在当前实现中是严格可加的，因而 exact Shapley 退化为三个 singleton modality log-likelihood-ratio traces，所有跨模态 Möbius interactions 都是数值零。更严重的是，当前 boundary artifact 无法由当前脚本和当前输入重建，违反项目要求的实现完整性前置门。

性能数字没有参与新颖性评分。611 个视频上的提升只能说明该组合可能有经验价值，不能使经典组件、不可识别的 inversion 或退化的 Shapley game 变新。

## Scores

| Dimension | Score | Verdict |
|---|---:|---|
| M1 Self-Risk Multiresolution Posterior | 4.0/10 | LOW |
| M2 Coalition-Inverted Rank Transport | 1.0/10 | LOW |
| M3 Shared Null-Complete Shapley Boundary | 3.0/10 | LOW |
| Task novelty | 2.0/10 | LOW |
| Mechanism novelty | 2.5/10 | LOW |
| Integration novelty | 2.5/10 | LOW |
| **Overall novelty** | **2.5/10** | **LOW — FAILS ≥6** |

## Core claims

1. **Label-free multimodal hateful-video temporal localization — LOW.** LELA already claims the exact training-free multimodal hate-localization task and produces per-frame modality scores, max-modality fusion, and thresholded frame decisions. MultiHateLoc already formalizes tri-modal temporal hate localization under weak supervision.

2. **Per-video L-curve/run-GCV/spectral-noise equal barycenter — LOW.** Per-instance parameter choice is a reasonable adaptation, but L-curve, GCV, empirical-noise spectral filtering, and fixed/equal model averaging are classical. The precise trio was not found in hateful-video localization, but an unreported application-level conjunction is not a new inference principle.

3. **Coalition inversion plus midrank transport — LOW.** The algebra identifies roles only if VA/VL/LA are values of one common additive game with a common null and scale. Here the pair coalitions were separately fit with different priors, reliabilities, and scales. The actual implementation uses quantile interpolation, not an exact nominal-order permutation; therefore the claimed exact marginal preservation is false. Rank-template reordering and marginal-preserving order-statistic assignment already exist in ECC/reordering literature.

4. **Shared null-complete exact Shapley boundary — LOW.** Shared prior/parameters and exact Shapley efficiency are implemented correctly, but the coalition utility is `prior logit + sum of per-view LLRs`. Shapley therefore collapses algebraically to the singleton LLR for each modality, with zero pair/triple interactions. The boundary then becomes max/union rank fusion plus a fixed zero excursion and peak-containing connected component, close to common localization decoding and LELA's max-modality aggregation.

5. **Three-module integration — LOW.** The assembly is unusual, but its strongest dense output still depends on the unidentified M2, while M3 does not introduce genuine coalition interaction and is not reproducible from the checked-in current implementation. A unique assembly of known or invalid components is not a ≥6 methodological contribution.

## Required implementation-integrity audit

### What passes

- `run_shared_coalition_game.py` fits one full 3-view model per opposite hash fold and reuses its prior and reliability parameters for all eight subsets.
- Exact three-player Shapley weights are correct. The stored maximum efficiency error is `1.1102230246251565e-16` over 611 videos.
- `project_shapley_boundary.py` uses average midranks and derives a new interval from a certificate; it does not inherit T3AL intervals.
- The fixed-grid common-cohort evaluator checks exact score/GT length equality before comparison.

### What fails

1. **The Shapley game is structurally interaction-free.** `infer_static` computes a prior plus an additive sum of view log-likelihoods. Audited maximum absolute centered Möbius interactions are:

   - visual-language: `6.1853e-16`
   - visual-audio: `6.1574e-16`
   - language-audio: `9.6143e-16`
   - triple: `1.4337e-15`

   These are floating-point cancellation, not semantic interaction.

2. **Exact floating-point ranks manufacture temporal evidence.** At `1e-12` tie tolerance, the median numbers of unique visual/language/audio role levels per video are only `2/1/2`; language is constant in `375/611` videos. The code nevertheless applies `rankdata(..., method="average")` to unrounded `1e-16` cancellation residues. Thus mathematically tied roles can receive different ranks.

3. **The final interval artifact is not reproducible.** Re-running the current boundary script with the current final dense scores and current shared-game file changes the predicted interval on `480/611` videos. The rebuilt macro interval F1 is `.2027/.1065/.0330`, whereas the stored report is `.2453/.1532/.0405`.

4. **The reported IoU labels are wrong in the request.** The evaluator hard-codes `(0.3, 0.5, 0.7)`. Therefore `.2453/.1532/.0405` are F1@`.3/.5/.7`, not F1@`.1/.3/.5`.

5. **M2 is not the claimed exact transport.** `project_coalition_rank_transport.py` calls `np.interp` on average-rank quantiles. The sorted nominal residual marginal differs in 376/611 videos in the prior audit; the maximum discrepancy was `.264`.

6. **Method choice is development-label informed.** Union, mean, bottleneck, and nominal decoder variants were all evaluated on the repeatedly used 611-video cohort, and union was retained. The formula has no learned numerical weights, but the final method selection is not untouched.

Because this integrity gate fails, no positive novelty result could be described as method acceptance under the repository's `AGENTS.md` rule.

## Closest primary work

| Work | Year / venue | Main overlap | Material difference |
|---|---|---|---|
| [LELA: Towards Training-free Multimodal Hate Localisation with Large Language Models](https://arxiv.org/abs/2602.09637) | 2026, arXiv | Exact task; training-free multimodal hate localization; per-frame modality scores; max fusion and thresholding | Uses five caption modalities, prompting, and composition matching rather than statistical smoothing/Shapley |
| [MultiHateLoc](https://arxiv.org/abs/2512.10408) | 2026, WWW | Tri-modal temporal hate localization; modality-aware temporal encoding/fusion and interval prediction | Weakly supervised MIL rather than label-free inference |
| [Rethinking Weakly-Supervised Video Temporal Grounding From a Game Perspective](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6164_ECCV_2024_paper.php) | 2024, ECCV | Cooperative-game cross-modal interaction used to produce frame-wise scores and localize moments without proposals | Frame-word players and weak training; not hate or modality-mask Shapley |
| [MM-SHAP](https://aclanthology.org/2023.acl-long.223/) | 2023, ACL | Shapley values quantify whole-modality contributions and diagnose unimodal dominance | Explanatory metric, not temporal localization or boundary decoding |
| [Which Modality Decides? Counterfactual Modality Attribution](https://arxiv.org/abs/2608.00076) | 2026, arXiv | Whole modalities as cooperative-game players; counterfactual coalition values and Shapley contributions | MLLM attribution/auditing, not per-video temporal hate localization |
| [Play Fair: Frame Attributions in Video Models](https://arxiv.org/abs/2011.12372) | 2020, ACCV | Shapley-style attribution over a video sequence at frame granularity | Frames, not modalities, are players; explanation rather than prediction boundary |
| [Test-Time Zero-Shot Temporal Action Localization (T3AL)](https://openaccess.thecvf.com/content/CVPR2024/html/Liberatori_Test-Time_Zero-Shot_Temporal_Action_Localization_CVPR_2024_paper.html) | 2024, CVPR | Training-free per-video localization, video-level aggregation, dense region proposal/refinement | Action localization with VLM adaptation; not multimodal hate-role attribution |
| [Ensemble Copula Coupling](https://arxiv.org/abs/1302.7149) and [Schefzik reordering](https://doi.org/10.1002/qj.2839) | 2013 / 2016, Statistical Science / QJRMS | Reorder calibrated marginals according to a rank template; multivariate ranks and order statistics | Probabilistic weather postprocessing, not video; nevertheless directly precedes the claimed rank-transport mechanics |

Additional required comparators do not rescue novelty: [Memory Matters](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Memory_Matters_Boosting_Training-Free_Zero-Shot_Temporal_Action_Localization_with_a_CVPR_2026_paper.html) accumulates test-time memory for training-free ZS-TAL; [Moment-GPT](https://arxiv.org/abs/2501.07972) is tuning-free frozen-MLLM moment retrieval; [Modality-Collaborative TTA](https://openaccess.thecvf.com/content/CVPR2024/html/Xiong_Modality-Collaborative_Test-Time_Adaptation_for_Action_Recognition_CVPR_2024_paper.html) already studies unlabeled multimodal video adaptation and cross-modal consistency. Classical parameter selection is directly covered by [Craven and Wahba's GCV smoothing](https://doi.org/10.1007/BF01404567) and [Hansen's L-curve](https://doi.org/10.1137/1034115).

## Direct answers

1. **Does M3 reach 6? No: 3.0/10.** Exact computation is not enough when the characteristic function structurally makes all coalitional interactions zero. The checked artifact is additionally unstable to machine-level tie noise and is not reproducible.

2. **Does the overall method reach 6? No: 2.5/10.** Task novelty is already occupied, M1 is classical, M2 is unidentified and misdescribed, and M3 is an additive Shapley wrapper around simple rank/max/connected-component decoding.

3. **Does old M2 block overall above 6? Yes.** It supplies the final strong dense score, so a separately identifiable boundary module cannot launder its role semantics. However, deleting M2 would still not make the current M3 reach 6.

## Strongest reviewer rejection

The central contribution is presented as an exact eight-coalition temporal Shapley game, but its audited characteristic function is strictly additive. Every Shapley role therefore equals a singleton modality LLR, and every pair/triple interaction is zero. Worse, exact floating-point ranking turns cancellation noise into temporal ordering, and the saved intervals differ on 480/611 videos when rebuilt from the current code and inputs. M2 algebraically inverts outputs from incompatible fitted games and calls interpolated quantiles exact marginal-preserving transport. M1 packages classical parameter selectors. LELA already covers the exact task and max-modality frame scoring; game-theoretic frame localization, modality Shapley, and rank-template reordering all have direct prior art. The paper's Shapley narrative is contradicted by its implementation, and the performance gains cannot supply the missing novelty.

## Exactly one minimal redesign path

Make **one role-source substitution**: replace both the separately fitted M2 pair inversion and the additive M3 game with a single **shared, null-complete, genuinely non-additive temporal coalition game**, while keeping the empirically useful midrank consensus, rank transport, and connected-component decoder unchanged.

Concretely, fit one opposite-fold, label-free latent evidence model with a shared prior/logit scale and explicit pair/triple cross-modal synchrony factors (for example fixed-form log-linear or copula potentials with predeclared smoothing). Evaluate that same model under all eight masks; require nonzero, stable Möbius fields; use its exact Shapley roles as the sole input to both dense transport and boundary; and use exact order-statistic assignment if exact marginal preservation is claimed. The acceptance gate must include tolerance-aware tie handling, bitwise/reasonably deterministic interval reproduction, a frozen decoder, and untouched-video confirmation. This one substitution removes the fatal M2 identifiability flaw and gives the eight-coalition computation actual content; without it, ≥6 is not plausible.

