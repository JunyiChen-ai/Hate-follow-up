# M1 Latents candidate19：独立 proposal review

日期：2026-10-04。结论：**PASS（same-family provisional）**。

这是 RESEARCH_ITERATION_RULES 第4条的一次独立方案审查，不是代码审查、性能结论或晋级裁定。未运行 GPU、未读取 test GT/预测、未修改算法常数、共享代码或评测器。按本次实际检索范围，没有发现来源完整方法已经用于 hateful video detection / localization；其余三种 STOP 类别也不成立。允许实施，然后按既有流程做独立代码审查、五视频 smoke 和完整双语料运行。

## 阅读范围与证据

本地完整阅读 `CLAUDE.md`、`RESEARCH_ITERATION_RULES.md`、`experiments/20261004_m1_latents/README.md`、`experiments/20261002_m1_iteration/README.md`、`src/mllm_judge.py`；阅读 `research-wiki/STATUS.md` 当前状态和失败候选汇总。不存在名为 `research-wiki/iterationfailedfamilies.md` 的文件，失败族核对实际以 iteration README 与 STATUS 为准。未重新核验历史 metrics 数字，亦未逐个重审全部归档源码。

实际浏览来源论文 [§3.2–3.3、Algorithm1、实现设置](https://arxiv.org/html/2605.02735v1)，以及 [官方仓库](https://github.com/zhangxin-xd/Unsilencing-Latent-Reasoning)。另读官方 `main.py` 参数、`ulr/inference.py` 初始化/对齐/reward/NES/最佳状态代码段和 `script/run_mmvp.sh`。只下载文本证据，未执行来源代码。证据目录：`runs/20261004_m1_latents/proposal_review/`。

检索及近邻核验如下；“未找到”是检索结论，不是绝对不存在的证明。

| 实际检索词/来源 | 结果与区别 |
|---|---|
| `"Unsilencing" "hateful" latent`；`"Unsilencing Latent Reasoning" hate video`；`"visual latents" "hate" video` | 未定位到完整来源算法的目标任务使用；包括来源论文自身及无关结果。 |
| `"hateful video" "latent reasoning"`；`"hateful video" "latent" optimization`；`"HateClipSeg" "latent"` | 定位到推理/对比学习近邻，未发现相同两阶段连续输入状态优化。 |
| `"hateful video" "test-time" optimization` | 找到 SCANNER，需明确区分而非宣称目标任务从未有 TTA。 |
| [SCANNER](https://arxiv.org/abs/2602.00132)，另核对 [AAAI 方法与实现](https://ojs.aaai.org/index.php/AAAI/article/download/39459/43420) | 基于源模型的质心对齐、多样性约束和目标域参数适配；不是冻结 Qwen 的逐实例四槽对比预热加 NES 熵进展搜索。 |
| [IARE §4.2–4.3](https://arxiv.org/html/2606.11953v1) | 标注解释驱动的 SFT/LoRA 与正确/错误推理偏好 DPO；不是无答案标签的冻结模型连续状态优化。 |
| [CLARA §3.3、§3.6](https://arxiv.org/html/2608.15905v1) | clip 编码、跨视频 local/global 对比损失和监督分类；不是当前单视频 patch–latent 对齐与状态搜索。 |

## 四项 STOP 判定

1. **已用于目标任务：不成立。** 近邻的“对比”“推理”“TTA”词汇重叠不等于来源算法已迁移。当前只支持“尚未找到既有目标任务使用”，后续有效性仍由双语料实测决定。
2. **纯 ensemble：不成立。** 单一冻结 Qwen3，四个状态是同一计算的优化变量；只选一个状态读出，不平均候选路径或原生/优化预测。原生全局和语音分支按现有规则保留。
3. **纯 calibration / 后处理：不成立。** 干预发生在最终视觉 margin 生成之前，优化连续输入；r6、标签、阈值、4 fps 网格和统一评测保持既有协议。
4. **纯工程：不成立。** 完整方法包含问题导向的内容支持、对比目标、连续状态搜索及明确的最终解选择。它不是只改提示词、attention 权重、分辨率或常数，也不是 Explorer 第五轮：没有新帧采集/帧候选选择。历史 attention、残差 steering、删媒体或 latent common-offset 分解失败，不能替代对此算法的实测。

## 源方法与适配完整性

论文规定 query attention 选择独立正负 patch 组、四槽五步对比预热、十五步 reward 搜索和最终单状态解码。方案保留这些算法部分；top20 内重新归一化、Qwen3 接口、初始化与优化常数是事前声明的适配。论文 Algorithm1 比较扰动候选却保存更新中心；方案改为保留实际评价的最佳候选，解决了其与正文最佳状态定义的不一致。[来源](https://arxiv.org/html/2605.02735v1)

**必须纠正来源事实，但无需改变已声明算法：** 初读候选 README 写“官方目前没有实现”，这不符合本次实际访问。官方已有 `main.py`、`ulr/`、启动脚本。其 [inference.py](https://github.com/zhangxin-xd/Unsilencing-Latent-Reasoning/blob/main/ulr/inference.py) 默认 endoftext 初始化，Stage I 对 contextual hidden states 对齐并反传冻结 backbone；Stage II 主 reward 是全 softmax 的 top-k 概率负对数均值，另有可选进展项，且仍保存更新中心。官方 [MMVP 脚本](https://github.com/zhangxin-xd/Unsilencing-Latent-Reasoning/blob/main/script/run_mmvp.sh) 的参数也不同。当前 detached merger embedding、纸面 Eq5 reward、最佳已评价 candidate 是**论文公式为主的独立适配**，不能称官方实现复现。实施时在实验 README 记录这些差异，勿据此追改本轮常数或把官方 reward 悄悄混入。

方案定义足以实施。两个细节在写代码时明确固定并记录即可，不是新一轮选参：NES 指数的循环索引起点（README 的 i 与论文合并两阶段计数方式不同，代码/配置须写出实际首个 sigma）；wrong_support 非等长 patch 的整数映射和半视频 frame 偏移取整方式。禁止运行后按效果选择这些约定。

## 代码审查必须核对的观察有效性

- 相关度是真实 post-RoPE、包含全部可见 keys 分母的 causal attention；仅问题文本 token 求平均。不得用 image-only softmax 或 pre-RoPE QK 代替。top8 与 bottom16 分组互斥，支持 embedding 与 image token/帧位置对齐。
- 前缀/问题缓存、视觉 merger 支持均 detach；Adam 只更新四个 FP32 输入向量。共享 `Judge.prefix_cache` 使用 no_grad，但实验代码还要确认捕获的张量不意外保留图。原生 global/speech 的缓存不能被视觉候选污染；每次评价恢复相同长度与 Qwen rotary delta。
- 当前冻结因果前缀位于状态之前，所以重复 Stage I 不会改变问题到视觉的相关度；一次获取相关度、重用支持并不删除该适配定义内的优化步骤。
- 四个 latent 位置分别计算全词表 top20 分布，reward 用相邻槽熵下降；Yes/No 仅由选中状态最后 hidden 经共享 `margins_fp32` 读出。最佳记录必须含实际评价过的 tensor/reward，未评分 NES center 不得冒充最佳。
- K=0 原生路径、原生六指标、SDPA 与 attention 提取的一致性、状态确实进入最终 margin，都属于实现验证；失败应修实现，不据此宣判方法有效或无效。
- 打分路径无 GT、无第二模型、无候选 score averaging、无额外数据集；缺失语音仍保留原生 None。统一 r6 与评测器不复制、不改动。

## 成本与可证伪机制

每视频 W 个视觉窗的新增模型成本为至多 17W 次四槽后缀 forward，另需实际 attention 获取；前缀可复用但新视频的原生视觉编码仍需付费。五步 Stage I 在声明的 embedding 空间中只做小张量优化，不能把这说成官方 contextual Stage I 的等价加速。60–120 GPU 分钟和 <32GiB 目前只是预算，须用 smoke 的实际窗口数、后缀耗时、attention 获取成本及峰值显存更新；不要先削减 15 步或改变 reward 以满足预算。原生配对诊断成本单报。新增成本有明确算法用途，是否值得由最终提升与消融判断。

novelty 单位为完整状态优化，相对同样四槽的 `initial`。必须明确比较方向是 **full 指标减 initial 指标 ≥ .01，在两个语料同一主指标成立**；README“相对 initial 降低”的语句应理解为去掉优化后的下降，避免实施报告反向计算。`warmup`、`search_only` 分别检验 Stage II/Stage I，未达到各自双语料 .01 门就不能单独主张必要性。`wrong_support` 真正更换支持向量的视觉内容，不能只改索引名称；保留输入上下文和优化预算，只能支持内容绑定解释，不能直接称时间因果证据。单帧/相同内容导致不可干预的样本要如实标记。

完整主门通过之后再做既定完整对照；报告全部六项主指标、eligible 视频数、原始 visual/max 排序及配对区间。只有 r6 后改善、原始视觉排序不改善时，不能宣称“读到更多正确视觉证据”。目前不以潜在 confidence shortcut 或预期精度不足 STOP；这些由已声明的可证伪对照和完整结果裁定。
