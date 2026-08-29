# OMSL-v6：方法与开发阶段结果

## 当前结论

OMSL-v6 是一个只输出 4-fps frame score 的 label-free-at-inference hateful-video localizer。它在当前 643 个可解码视频、四数据集统一 evaluator 上，以预先约定的核心定位指标 `within-video macro ROC-AUC` 同时超过 label-free T3AL reconstruction 和 weak-video-label MultiHateLoc reproduction；fresh novelty reviewer 对完整方法组合评分为 6.1/10。

这是一项 **development-selected exploratory result**，不是 untouched confirmatory SOTA。v1--v5 的设计迭代查看过同一 test cohort；LELA 和官方 MultiHateLoc 也没有发布可复现的 frame grid/span conversion。因此可成立的 SOTA 表述仅限下文明确列出的本地统一协议与 published-reference 对照。

媒体解码、抽帧、ASR 时间戳和音频 embedding extraction 是预处理，不算方法 module。

## Module 1：Frozen Multimodal Coalition Evidence

输入只有一个 visual detector（A10/VidGroup），再加 timestamped Qwen language logits 和 ImageBind audio margins。三条 frozen timeline 对齐到同一个 4-fps 网格并转换为样本内 average ranks。

同一个 symmetric equal-mass log-mean-exp scorer 依次计算 V/L/A 的全部八种 subset。这样后续所有 main effect 和 interaction effect 都来自同一个可比较的 analytic game。这里的 interaction 只表示这个 marginal-stream game 内的非加性项，不宣称是 joint MLLM 的 causal interaction。

该 module 解决的是：不同模态分数没有共同标尺、直接加权融合会让某个模态凭尺度支配结果的问题。

## Module 2：Jointly Calibrated Möbius Localization

先对八个 coalition worth 做精确 Möbius inversion，得到 V、L、A、VL、VA、LA、VLA 七个 effect timeline。然后针对每个视频执行以下操作：

1. 从 lag=1 开始扫描，找到所有非恒定模态首次共同低于固定 ACF reference 的 lag；643/643 个视频均找到真实 crossing，没有 capped fallback。
2. 以该 lag 为 block length，对 language/audio 做 31 次 deterministic、nonwrapping complete-block permutations，visual 保持不动。
3. 每次 permutation 只产生一个 null maximum，这个 maximum 同时覆盖所有时间点以及 VL/VA/LA/VLA 四个 interaction fields。
4. 用 inclusive upper-tail Monte Carlo p-value 将 interaction 转成 signed `1-p` adjusted-evidence display。它不是 interaction posterior，也没有二值显著性 threshold。
5. A10 visual score作为 primary lexicographic key；language/audio main evidence 与 calibrated interaction evidence只细分 A10 的相同分数 plateau，不能颠倒 A10 已区分开的两帧。剩余完全相同的 key 使用 average rank，不加随机 jitter，也不用时间索引破坏 tie。

该 module 解决的是：弱模态反转强视觉证据、逐 interaction/逐时间检验带来的 multiple-comparison 假证据，以及全帧并列导致的定位退化。

## Module 3：Orthogonal Semantic Propensity Readout

Module 2 的 temporal rank 被转换成零均值 residual。视频级 semantic intercept 则由两个 scalar 做 equal-mass log-mean-exp：Jeffreys-smoothed A10 positive occupancy logit，以及 frozen whole-video Qwen logit。

最终每帧 score 等于该视频的 semantic intercept 加上 centered local residual。因为 intercept 对同一视频的所有帧完全相同，它可以调节跨视频 hate propensity，但数学上不能改变任何 within-video ordering。

该 module 解决的是：pooled frame AUC 需要跨视频语义判断，但如果把 whole-video MLLM signal直接混入局部排序，它会把整段视频一起抬高并掩盖真实定位能力。

## 四数据集结果

| Dataset | OMSL-v6 ROC | OMSL-v6 PR | OMSL-v6 within-video ROC | MultiHateLoc within-video ROC | T3AL within-video ROC |
|---|---:|---:|---:|---:|---:|
| HateMM | 0.8507 | 0.5781 | 0.6497 | 0.6108 | 0.5068 |
| HateClipSeg | 0.6692 | 0.6622 | 0.5473 | 0.4996 | 0.5003 |
| MHC | 0.7458 | 0.4970 | 0.7005 | 0.5013 | 0.5605 |
| MHC-zh | 0.7522 | 0.5354 | 0.6835 | 0.4962 | 0.4996 |
| **4-dataset macro** | **0.7545** | **0.5682** | **0.6452** | **0.5270** | **0.5168** |

统一宏平均总表：

| Method | Supervision | Macro ROC | Macro PR | Macro within-video ROC |
|---|---|---:|---:|---:|
| **OMSL-v6** | Label-free inference | **0.7545** | **0.5682** | **0.6452** |
| MultiHateLoc-DMS reproduction | Weak video labels | 0.7316 | 0.5357 | 0.5270 |
| T3AL reconstruction | Label-free | 0.6319 | 0.3790 | 0.5168 |

相对 MultiHateLoc，核心 within-video metric 的 paired 20,000-sample video bootstrap difference 为 +0.11825，95% CI [0.04430, 0.19408]，empirical one-sided p=0.00065。四数据集宏平均 pooled ROC/PR 点估计也更高，但 bootstrap CI 均跨 0，因此不能写成显著提升。

LELA 的论文只报告 pooled frame ROC/AP，且不公开 frame rate、split、span-to-frame rule 或代码，也不报告 within-video metric。OMSL 在 HateMM/MHC 的当前 4-fps pooled ROC 为 0.8507/0.7458，高于 LELA 论文 prose 所述的 0.7264/0.7227；这是 published-reference comparison，不是 matched reproduction，也不能外推到 LELA 的 AP。

## 输出与不主张的内容

- v6 正式输出是 frame-score timeline；643 条 `intervals` 全部为空。
- v5 及更早 artifact 中出现的 intervals 是继承的 A10 proposals，已从 authoritative v6 删除，不能算 OMSL interval performance。
- 不声称 first MLLM hate localization、first multimodal localization、causal interaction discovery或 universal published-method SOTA。
- 可成立的 novelty claim 是：在已检查工作中，没有发现把统一八 coalition analytic game、joint temporal max calibration、visual-primary weak-order guarantee 与 scalar-semantic/local-residual noninterference 同时用于 label-free hateful-video frame ranking 的方法。

## Artifact

- Code SHA-256: `197f52ad7724738623f38b4f2f51ff2a2cea4327ecf6ea03b2feb2edc318368c`
- Prediction SHA-256: `53c32de1e7b5c2fa533d84b8c7bdd26a7bc789410353610406edd3e497dd8bb5`
- Metrics SHA-256: `84d29d15a23056f19dce3d183f095e9396b50e023dddabc1bee43a3fb5f5fb76`
- Coverage: HateMM 215、HateClipSeg 118、MHC 161、MHC-zh 149，共 643 个可解码视频。
