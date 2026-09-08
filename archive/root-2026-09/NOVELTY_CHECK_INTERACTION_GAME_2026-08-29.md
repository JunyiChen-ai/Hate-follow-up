# Fresh Novelty Review: Interaction Coalition Game Candidate

- review date: 2026-08-29
- literature cutoff: 2026-08-29, including a targeted scan of the most recent six months
- review_independence: **same-family**
- acceptance_status: **provisional**
- implementation_integrity_gate: **PARTIAL — core logic passes; two metadata/ranking discrepancies and one environment-limited reproduction remain**
- novelty threshold: **6.0/10**
- final verdict: **M2 = 5.4/10; M3 = 3.2/10; overall = 5.2/10; none reaches 6**

## Blunt verdict

**不过 6。M2 不过 6，M3 不过 6，整体也不过 6。** 新版本相对 run01 是实质性修正：它不再是 additive game。一个共享的 2×27 joint categorical table 确实产生非零 pair/triple Möbius fields，八个 coalition 也确实来自同一模型的 exact marginalization。但 `interaction-identifiable` 只能在非常窄的、teacher-relative 意义下成立。

给定冻结的 LESS teacher posterior \(q(y\mid x)\) 和视频均衡经验分布 \(p(x)\)，忽略平滑时该一步构造就是

\[
p_q(x,y)=p(x)q(y\mid x),\qquad
p_q(x\mid y)=\frac{p(x)q(y\mid x)}{\pi_y},
\]

因此 full-input Bayes posterior 精确返回原 teacher：\(p_q(y\mid x)=q(y\mid x)\)。所以非零 masked-coalition interaction 是一个合法的 **teacher-induced observational attribution game**，却不是从无标签数据中独立识别出的 latent hateful-content synergy。Cross-fit 阻止目标视频参与自己的拟合，但不消除这个语义循环。

性能数字完全没有进入新颖性评分。611 个视频上的 interval 提升证明该归因/排序组合可能有经验价值，但不会把 teacher self-reconstruction、Shapley/Möbius、rank shuffle 或 connected-component decoding 变成新机制。

## Scores

| Dimension | Score | Verdict |
|---|---:|---|
| M1 Self-Risk Multiresolution Posterior | 4.0/10 | LOW |
| M2 Coalition-Inverted/Joint-Game Rank Transport | **5.4/10** | MEDIUM — **NO ≥6** |
| M3 Complete Union Boundary | **3.2/10** | LOW — **NO ≥6** |
| Task novelty | **2.5/10** | LOW |
| Mechanism novelty | **5.3/10** | MEDIUM |
| Integration novelty | **5.8/10** | MEDIUM-HIGH |
| **Overall novelty** | **5.2/10** | **MEDIUM — FAILS ≥6** |

A second fresh same-family advisory reviewer gave M2 `6.1/10`, because the exact pseudo-game + Möbius consensus + order-statistic transport combination is uncommon. The final skeptical adjudication remains `5.4/10`: the joint table adds no independently identified outcome information, and the attribution and rank-reordering ingredients have close prior art. Both reviewers agree that M3 and the overall method are below 6 and that latent semantic interaction identification fails.

## Core claims

1. **Shared complete teacher-relative interaction game — MEDIUM (5.3/10).** The complete joint table, shared prior/scale, exact masked marginalization, and nonzero Möbius fields are real improvements over run01. Yet dependent weak-supervision models, conditional SHAP under dependent features, and exact multimodal Shapley interaction are established. The defensible novelty is the task-specific teacher-relative construction, not latent hate-interaction identification.

2. **M2 symmetric midrank rank transport — MEDIUM (5.4/10).** Using exact Shapley/Möbius temporal fields to choose the permutation of a nominal hate residual is an uncommon integration. However, Shapley/Möbius attribution is known, average ranks are classical, and exact order-statistic reordering that preserves a marginal is the core operation of ECC/Schaake-style reshuffling. Current M2 also merges only exactly equal values, not the floating-equivalence classes claimed in the final description.

3. **M3 nominal + max-role + max-interaction union certificate — LOW (3.2/10).** Max/union aggregation, the fixed zero excursion, and selecting the connected component containing the global peak are simple, established localization operations. The new game provides better input fields, but M3 itself does not introduce a new decoding principle. Five M3 variants were compared on the repeatedly used cohort, so “zero learned numeric parameters” does not mean “no label-informed method selection.”

4. **Overall label-free multimodal hate-localization integration — MEDIUM (5.2/10).** The pipeline is coherent and apparently unmatched as a complete assembly. But LELA occupies the exact task, MultiHateLoc occupies dense multimodal hate localization, and the mechanism is explainability/reordering integration rather than a new statistically identified localization model. A new application and uncommon assembly are insufficient for ≥6.

## Required implementation-integrity audit

### What passes

- `run_interaction_coalition_game.py` fits on the opposite hash fold and gives equal total fractional mass per source video.
- One `2×27` class-conditional categorical table is reused for all eight coalitions; each masked value is an exact marginal of that table.
- The game is genuinely non-additive: stored maxima are pair Möbius `.0806816`, triple Möbius `.0218143`; Shapley efficiency error is `2.22e-16`.
- Exact M2 transport assigns sorted nominal residual order statistics to a deterministic consensus order, preserving the residual multiset and video mean exactly.
- M3 uses tolerance-aware midranks, derives new intervals rather than inheriting T3AL intervals, and decodes the `>0` component around the global maximum.
- Current M2 and M3 artifacts were rebuilt byte-for-byte from their checked inputs. Key SHA-256 values are recorded in the trace.
- The evaluator computes pooled ROC/PR, within-video ROC, and greedy one-to-one interval F1 at IoU `.3/.5/.7` as reported.

### What does not cleanly pass

1. **The semantic identifiability label is too broad.** The table is uniquely identified conditional on the teacher soft labels, but latent hateful \(Y\) is not independently identified.
2. **M2's tie claim exceeds its code.** `project_shared_shapley_rank_transport.py` uses ordinary exact-value `rankdata(method="average")`; only M3 merges floating-equivalent values with `max(1e-12, 1e-10·scale)` tolerance.
3. **One fixed prior is under-disclosed.** Besides Jeffreys `0.5` in each of the 54 joint cells, `class_mass = np.ones(2)` imposes a Beta(1,1) class-prior pseudocount.
4. **Whole-pipeline `label_selected_parameters: 0` is misleading.** No numeric weight/threshold was fit, but complete-union was selected after evaluating five decoder formulas on the development cohort.
5. **The full game artifact could not be byte-rebuilt in the available `.venv-r1` because its audio dependency chain lacks `torchaudio`/`iopath`.** The code and stored audit were inspected, and downstream M2/M3 artifacts are reproducible, but this prevents an unqualified end-to-end integrity pass.

This review is therefore **not method acceptance**, even apart from the novelty score.

## Closest primary work

| Work | Year / venue | Main overlap | Material difference |
|---|---|---|---|
| [LELA](https://arxiv.org/abs/2602.09637) | 2026, arXiv | Exact training-free multimodal hateful-video localization task; per-frame scores and cross-modal composition | Five caption modalities and prompting; no teacher-relative joint game or rank transport |
| [MultiHateLoc](https://arxiv.org/abs/2512.10408) | 2026, WWW | Dense tri-modal hate localization and modality-aware temporal fusion | Uses video labels, learned encoders, contrastive alignment and MIL |
| [InterSHAP](https://ojs.aaai.org/index.php/AAAI/article/view/35452) | 2025, AAAI | Exact modality-level Shapley interaction for multiple modalities without performance labels | Model explanation, not temporal rank transport/localization |
| [Learning the Structure of Generative Models without Labeled Data](https://proceedings.mlr.press/v70/bach17a.html) | 2017, ICML | Unlabeled generative weak supervision with higher-order dependencies among noisy sources and latent \(Y\) | Learns dependency structure under identifiability conditions; not teacher-distilled temporal attribution |
| [Learning Dependency Structures for Weak Supervision Models](https://proceedings.mlr.press/v97/varma19a.html) | 2019, ICML | Identifying dependencies among weak sources without ground truth | Robust dependency recovery, not a one-step teacher-conditioned table or video decoder |
| [Rethinking Weakly-Supervised Video Temporal Grounding From a Game Perspective](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6164_ECCV_2024_paper.php) | 2024, ECCV | Cooperative-game interactions generate framewise temporal-localization scores | Learned frame-word game under weak supervision, not three-modality teacher-relative marginalization |
| [MM-SHAP](https://aclanthology.org/2023.acl-long.223/) and [CMA](https://arxiv.org/abs/2608.00076) | 2023 ACL / 2026 arXiv | Whole modalities as coalition players; Shapley contribution/counterfactual attribution | Diagnostic attribution, not marginal-preserving temporal reranking |
| [Ensemble Copula Coupling](https://arxiv.org/abs/1302.7149) and [Schefzik reordering](https://doi.org/10.1002/qj.2839) | 2013 / 2016 | Exact order-statistic reshuffling using a rank/dependence template while preserving marginals | Weather postprocessing, but directly anticipates M2's rank-transport mechanics |

Required temporal comparators reinforce the low task score: [T3AL](https://openaccess.thecvf.com/content/CVPR2024/html/Liberatori_Test-Time_Zero-Shot_Temporal_Action_Localization_CVPR_2024_paper.html), [Memory Matters](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Memory_Matters_Boosting_Training-Free_Zero-Shot_Temporal_Action_Localization_with_a_CVPR_2026_paper.html), [Moment-GPT](https://arxiv.org/abs/2501.07972), and [Modality-Collaborative TTA](https://openaccess.thecvf.com/content/CVPR2024/html/Xiong_Modality-Collaborative_Test-Time_Adaptation_for_Action_Recognition_CVPR_2024_paper.html). Recent-six-month checks also included [Dr. SHAP-AV](https://arxiv.org/abs/2603.12046), [mllm-shap](https://aclanthology.org/2026.acl-demo.38/), [SAGE](https://aclanthology.org/2026.acl-long.817/), [CLARA](https://arxiv.org/abs/2608.15905), and [Approximating Shapley Interactions](https://doi.org/10.1007/s10994-026-07062-6); none contains the complete candidate, but they further crowd modality attribution and hateful-video multimodal reasoning.

Classical mechanism priors include [Aas et al. on dependent-feature SHAP](https://arxiv.org/abs/1903.10464), [Craven–Wahba GCV](https://doi.org/10.1007/BF01404567), and [Hansen's L-curve](https://doi.org/10.1137/1034115).

## Strongest reviewer rejection

LELA already establishes training-free multimodal hateful-video localization, while prior work supplies dependent weak-supervision games, conditional and multimodal Shapley interaction, game-based temporal grounding, rank aggregation, exact order-statistic reshuffling, and connected-component decoding. The claimed semantic delta is unsupported: one weighted joint-table step from \(q(Y\mid X)\) constructs a Bayes model that asymptotically returns that same \(q\). Its exact Shapley and Möbius values therefore decompose a teacher-induced observational game; nonzero terms need not be hateful-outcome interactions. Cross-fitting prevents target reuse but does not break this circularity. M3 is a modest max-and-component assembly, current M2 does not implement the claimed tolerance-aware ranks, and complete-union was chosen from five formulas on the reused development cohort. The defensible contribution is an intricate attribution-and-reranking integration, not interaction-identified latent hate localization.

## Exactly one change most likely to exceed 6

**Replace only the LESS-derived fractional state \(q_{\text{teacher}}(Y\mid V,L,A)\) with one frozen, independently generated fourth hate-semantic soft anchor that is never a coalition player and never enters M1; keep the current 2×27 table, all-eight-coalition exact marginalization, M2 transport, and M3 decoder unchanged.**

For example, obtain the anchor from a frozen training-free multimodal hate judge whose representation/prompt is independent of the three ternary evidence views. The three game players remain V/L/A; the anchor supplies only the soft outcome used to fit their joint relationship. This breaks the exact self-reconstruction of the same LESS posterior while preserving the empirically working ranking/transport architecture. The claim must remain “interaction relative to an independent semantic anchor,” not human-truth causal synergy, and the decoder must be frozen before untouched confirmation. This single source substitution is the smallest plausible path to an overall score above 6.
