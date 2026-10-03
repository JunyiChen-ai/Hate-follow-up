# Preserver 独立 proposal review

日期：2026-10-03。结论：**PASS**。按 `RESEARCH_ITERATION_RULES.md` 第 4 条，未发现四类 STOP 的成立证据；放行实现不代表涨点或视觉保持机制已获验证。

审查实例：`/root/m1_grounder_proposal_review`，沿用与主会话相同模型。本次为 candidate17 的一次独立提案审查。读取 `experiments/20261003_m1_preserver/README.md`、仓库规则、主论文及作者代码；未修改方法或实现，未启动 GPU、训练或推理，未读取 GT、Highlighter 最终预测或性能，未计算内容哈希。

## 四项 STOP 判定

| 条件 | 判断 |
| --- | --- |
| 来源已用于 hateful video detection / localization | 本次实际检索和 primary 核查未找到 MAD-RAG 或同一同模型参考问题 attention-output mixing 在目标任务中的使用。结论受检索范围限制。 |
| 纯 ensemble | 否。参考读取和上下文读取来自同一冻结参数模型；逐层修改内部计算，不是多个独立模型的预测组合。 |
| 纯 calibration / 后处理 / 平滑 | 否。混合对象是 self-attention 输出，进入后续 residual、MLP 和语言层，最终原始 margin 随之改变。 |
| 纯工程技巧 | 否。完整迁移逐层参考表示混合机制；不是只改措辞、替换编码器或调整标量分数。两个缓存是明确的计算图适配，不能单独包装为理论贡献。 |

没有依据预期效果、可能退化或此前候选负结果否决本提案。

## 主来源与官方实现核对

实际读取 [PMLR 官方条目](https://proceedings.mlr.press/v306/zhao26g.html) 链接的 [完整论文 PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/zhao26g/zhao26g.pdf)，并核对本地 `third_party/mad_rag_source_read/paper.txt` 的 §5.1–5.2、§6.1 与参数说明。作者为 Beidi Zhao 等，发表于 ICML 2026。原任务是 OK-VQA、E-VQA、InfoSeek 等知识型单图 VQA，不是仇恨视频定位。

源输入为 `[I, Q_I, C, Q_C]`，两个问题内容相同，前一个问题在 retrieved context 之前。Eq.3 只注入第一问题指向图像的 attention 分量；Eq.5 写完整第一问题输出的凸组合。第一问题还能读取先前 instruction/question tokens，因此两式一般不等价。提案正确选择了完整输出机制，没有继承“只含视觉信息”的说法。

本轮实际从 [官方 Qwen25vl/run_mad_rag.py](https://github.com/ubc-tea/MAD-RAG/blob/main/Qwen25vl/run_mad_rag.py) 读取：

- 第 135–195 行的 wrapper 先运行原 self-attention，再将返回的 Q2 attention output 与 Q1 output 混合；不重新提取 image-only attention。
- 第 177–189 行通过定位的 question token span 逐行对应混合，执行条件是 prefill 序列长度大于 1。
- 第 950 行 CLI 默认 alpha 为 `.5`；辅助函数和 controller 中还存在 `.7` 默认值，但正式入口传入 CLI 参数。提案借用 `.5` 与论文 §6.1 和入口设置一致。
- 第 1062–1070 行为所有 language self-attention layers 安装 wrapper。原 self-attention 返回输出投影之后的表示，提案的 post-`o_proj` 对象与该接口相符。

原论文展示过 alpha 敏感性；固定借用 `.5` 不需要本地扫描，但不能称来源没有数值选择或所有模型都验证过同一最优值。论文报告的近似低开销不适用于两次图像编码及逐窗两次读取的当前实现。

## 适配与解释边界

**两个缓存并非 literal dual-question 的等价实现。** 源方法的 C 和 Q_C 可以直接读取 Q_I；本候选的 full-context prefix 不含参考问题 tokens，C 的 states 也没有参考问题参与。参考前缀省掉 ASR 和 global Q/A，政策和问题仍在。逐层注入保留了来源的核心运算，但改变了上下文路径，须作为适配独立测试。

**混合整段 suffix 是额外范围变化。** 源 wrapper 只混合匹配出来的裸 question span；提案混合相同 token IDs 的全部 query suffix，包含 chat header 和后续模板位置。token 对齐可验证，但绝对/mRoPE 位置及上文不同本来就是两个视图的区别。不能把“token IDs 一样”解释成两次读取的全部计算相同，也不能将全部收益归于问题正文的视觉表示。

**Qwen3 的层间传播须准确表述。** full-context 路径第 l 层的 O 已经受到更早层混合的影响；不是每层都在两个完全独立 native 轨迹之间取平均。参考路径保持原生、只读。原 full-prefix KV、native global/answer/speech 可以保持不变，但被混合的 suffix 隐状态和其后层 K/V 应当变化。后续代码检查要验证机制实际到达最终 margin，同时防止跨窗污染。

该候选与 Reinforcer 的 residual-direction 加法、Projector 的同层 masked-reference 正交修正不同；此处是参考前缀上的逐层完整 attention-output 凸组合。不能仅因为都使用参考视图，就认定为此前同一实验，也不能把换名称当作新机制。

## 成本与流程核对

以 V 为 visual windows、B 为 visual 加可用 speech questions：原方法 `3+B`，Preserver `4+B+V`，配对 `4+B+2V`，声明自洽。多出的一个 reference prefix 和每窗一个 reference query 是新视频真实成本。两个 prefix KV 同时存在；还需一个窗口的 36 层输出数组及计算图所需临时张量。两次图像编码和 FP32 混合成本不能被“缓存复用”隐去。

双 cache 必须各自恢复 native rotary state、各自记录前缀长度并裁剪 query KV。native visual 可以为配对多跑，部署不能把配对多跑算入主方法要求，也不能把 reference 预计算当作免费。只读原始 frame/ASR 资产，无需引入外部数据。实际 frame 数为输入记录值；18 帧病例不能被硬编码的 20 帧映射遗漏。

alpha0、identity reference、post-o_proj oracle、完整 suffix token 对齐、36 层访问、跨视频恢复、无 speech 和 canonical evaluator 接线均已列入独立 code review，范围合适。五视频 smoke 要测两个缓存的真实峰值和耗时，替换估算后再完整评分。本次只检查声明，没有执行这些模型测试。

## 可证伪控制与必要限定

1. **alpha0** 是移除机制与计算恢复检查，必须直接返回原输出以满足精确 native 目标；不能因仍运行参考分支就改掉 native context、mask 或 rotary 状态。
2. **policy-only reference** 能检验 image-conditioned reference 整包是否必要，但同时改变视觉输入、前缀长度、位置和参考表示范数。即使变差，也不能单靠这一结果证明差异只来自图像语义。若后续要作更强归因，应预声明保持布局/尺寸的图像内容控制；本轮不额外要求在主结果前运行它。
3. **循环移位 reference images** 保留全局图像集合，改变顺序及图像和 timestamp slot 的对应，不等于删除视觉证据，也不能单独分离时间配对与顺序作用。需记录实际使用的 frame 索引映射和哪些窗口对应图像改变；重复画面或单窗样本不能充当充分的时间定位反事实证据。采用实际 N，包括 18 帧情形。
4. **reference-only diagnostic** 可检验保留 contextual read 是否有价值；保留 native global/speech 的定义已经明确。不得事后按视频选择 reference-only 与混合中的较好者。

这些控制能够证伪“收益需要参考图像、适当的时间对应和上下文整合”的具体组合假设，但不能仅凭 attention magnitude、激活改变或少量案例证明原方法受到了 ASR/global verdict 压制。后一个原因仍需病例和直接证据支持。

现有 main gate、组件移除门、两个语料三指标及 raw visual/speech/max 排序报告覆盖主要判断。global 不变也不保证 pooled 的改变代表定位改善；max 与 r6 之后的效果仍应与 within 和原始排序共同解释。

## 实际目标任务查新

本轮实际查询包含来源完整题名、`MAD-RAG`、`MAD RAG`、`2602.00344`，分别配合 hate、hateful video、HateMM、HateClipSeg；另外检索 hateful video detection/localization 与 attention output mixing、attention injection、attention distraction、dual-question、image-conditioned 和 context-free attention。未找到目标任务使用来源机制的 primary 证据。搜索出现的通用 RAG、文字 debate、多模态融合或静态 meme 工作不直接等于此机制。

为排除仅按名称判断，本轮读取以下 primary 原文，检查来源编号/标题及上述特征，并结合方法内容区分：

- [LELA](https://arxiv.org/html/2602.09637v1)、[MultiHateLoc](https://arxiv.org/html/2512.10408v1)：多阶段描述/提示、组合匹配或学习式时序融合，未见 MAD-RAG。
- [RAMF](https://arxiv.org/html/2512.02743v1)、[MARS](https://arxiv.org/html/2601.15115v1)：多立场或多阶段理由并不等于同一问题两种 context 的 attention-output 注入。不能因为都有两次问答便认定来源重复。
- [CLARA](https://arxiv.org/html/2608.15905v1)、[IARE](https://arxiv.org/html/2606.11953v1)、[HVGuard](https://aclanthology.org/2025.emnlp-main.456.pdf)、[SAGE](https://aclanthology.org/2026.acl-long.817.pdf)：rationale、专家或 clip 表示融合，未见来源方法采用。
- [MM-HSD](https://arxiv.org/html/2508.20546v1)：CMA 以不同模态作为 query/key 进行早期融合，再与模态特征组合；不是冻结 LVLM 内两次相同问题读取的逐层 output mixing。

结论严格限定为：本轮查新没有触发目标任务采用 STOP，提案的内部混合机制具备可实施、可证伪的定义，可以进入独立代码检查和完整实验。没有对 Highlighter 或其他运行候选的结果作任何推断。
