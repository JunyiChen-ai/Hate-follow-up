# M1 ideation 独立 jury：2026-10-04

结论：优先实施 **per_window_visual_latent_optimization**，其次 **acoustic_alignment_posterior_conditioned_reading**，第三 **complete_semantic_cluster_tree_with_leaf_regrounding**。这是 same-family provisional 的选题判断，不是机制成立或晋级批准。9 个生成项、9 个完全不同 dedup_key、9 个保留排序；未按可实现性或想象中的 shortcut 删除任何项。本轮未跑 GPU、未读新 GT、未改代码或评测器。

## 全量排序

| 排名 | dedup_key | 选择理由与 provisional novelty | 不确定性及最强反驳 |
|---|---|---|---|
| 1 | per_window_visual_latent_optimization | 完整迁移“对比预热+逐位置置信进展搜索”，直接改变连续测量状态；本次未找到该完整方法已用于 hateful video。推荐。 | 源码未公开、常数缺失、Qwen3 适配；提高自信可能不改善排序。不能将 confidence 上升当机制证据。 |
| 2 | acoustic_alignment_posterior_conditioned_reading | 同一 Whisper 的字符对齐，加单调路径分布及 Qwen 支持条件化；不是仅替换时间戳。推荐。 | attention 归一化不是校准的声学后验；ASR 错词不可修复；8 秒窗口可能使改善过小。 |
| 3 | complete_semantic_cluster_tree_with_leaf_regrounding | 完整 VideoTree 聚类、代表观测、适应宽度、相关性深度扩展；新增实际叶观测并全局/局部重读，具有独立完整机制。推荐备用。 | 必须与已归档 Explorer 的加帧区别；MAESTRO 已有 global/local 循环；增加全局重读可能损害 pooled。 |
| 4 | single_reader_temporal_word_confusion_lattice | 单模型声学备选以互斥分支进入一次语义读取，区别于单时间戳对齐。暂放行、备选。 | 序列 beam 分数不等于词后验；分支位置和隔离实现重；若逐假设打分再平均便不再是所提方法。 |
| 5 | executable_temporal_evidence_program | 可执行时间/实体/引用绑定区别于自然语言解释。暂放行。 | 若执行器仅检查 JSON 或来源 ID 存在，没有执行语义约束，就退化成纯工程；稀疏媒体不能证明“不存在”。 |
| 6 | temporal_quotation_coreference_scope_graph | 局部话语所有权、引用和否定作用域的可执行包编译及结构 attention，区别于广播 stance。暂放行。 | 相邻错误经验 Grounder/Factorizer 已说明屏蔽本身不够；ASR 无标点/说话人，文学生成来源不能直接转移为可靠 speaker ground truth。 |
| 7 | provenance_bound_temporal_entity_and_discourse_graph | 持续实体/话语边控制真实证据检索，区别于仅生成 ledger。暂放行。 | MATCH 已检索并验证时空证据；本项需证明跨时所有权图而非额外像素/字幕带来提升；成本最高区间之一。 |
| 8 | interval_witness_reconciliation_before_independent_window_reading | 有限区间见证组合及覆盖矛盾驱动重读可作为自研完整方法。暂放行。 | “出现过”不充分决定 stance；父子合法并不意味着见证真实；与 MAESTRO 的循环区别必须由所有权重排对照验证。 |
| 9 | independent_factual_verification_before_window_read | answer-blind 事实重测加实际来源验证尚有窄差异，暂放行但不优先。 | MATCH/LEAF/IARE 已覆盖核查、grounding 和上下文解释；没有实际来源字段纠正时容易只是更长 prompt/CoT。 |

以上“暂放行”表示当前证据没有触发四类 STOP，不是断言全球首次。仅当正式方案真的缩减成纯 prompt、ensemble、后处理，或查到完整来源方法已在目标任务使用时才能 STOP。文献重叠、预计增益较小、实现难、运行超过 2 小时均不是本仓库的 novelty STOP。

## 首选：每窗口优化连续视觉潜变量

1. 对原生视觉窗口问题，保留原媒体和冻结 Qwen3-VL-8B，在回答前插入 4 个连续 latent slots；保留原全局和语音分支及 r6。
2. 按窗口 query 到图像 token 的相关性分配互不重叠的正/负 patch 集，以来源对比目标预热 5 步；每 slot 正 2、负 4。
3. 从预热状态出发做 15 步来源 NES 与逐 latent 位置的预测分布置信进展奖励；只保留一个选定 latent 状态，用正常 Yes/No margin 进入固定下游，不平均多次预测。

来源是 [Unsilencing Visual Latents](https://arxiv.org/html/2605.02735v1)，实际读摘要/引言、3.1–3.3、式 2–6、Algorithm 1。该方法优化每个实例的 latent，不更新 backbone；与已归档 Reinforcer 的固定 residual 方向叠加、Grounder 的 attention 限制不同。目标领域 [SCANNER](https://arxiv.org/abs/2602.00132) 已做 TTA；“首次 TTA”不能写，其 centroid-guided adaptation 不能据摘要等同于该完整 latent 优化。

源论文存在关键复现缺口：初始化、温度、学习率、搜索噪声初值/衰减及 top-delta 必须补成预先声明的适配常数；不通过 test 指标扫描选择。Algorithm 1 比较扰动候选的奖励，却在成功时保存更新后的状态，和正文“保留最佳奖励状态”不完全一致。正式方案应选定并写明语义一致的规则，不能悄悄实现另一算法后称精确复现。不应以一个未训练的新 special token 冒充已有可用 latent delimiter。来源 full-vocabulary/top-delta reward 不能无声明替换成二元 Yes/No 熵。

首个实验：实现可靠性检查后，全 215 HateMM + 118 HCS 做配对 native/full；完整语料各自一台机器。先评 full 主门，通过/有信号后再按漏斗做 unoptimized-4slots、Stage-I-only、无 Stage-I 的 Stage-II，以及保持预算的错误窗口 patch 绑定。前两项分别排除额外 slot 和仅预热解释；若要把两阶段分别写成 novelty，各自移除都需双语料同项指标下降至少 .01。错误绑定若不降，不得说模型利用了正确窗口视觉证据。记录原始 visual/max 排序、奖励变化和最终 r6 三指标，不能用 r6 后提升替代原始证据检查。

成本：复用媒体/全局/语音与可分离 prefix；每窗口另有相关性采集、5 个 latent 优化更新、15 个搜索迭代和最后读出。若为候选与更新状态各算奖励，实际 suffix forward 次数高于 15，必须计数。生成者估 60–150 分钟，未经测量；先用固定五视频检查显存/耗时再外推，不用五视频选方法参数。超过 ideation 单 pilot 2h 时标 needs full/manual pilot，不能当新颖性失败或终止整体任务。

## 第二选择：声学路径分布条件化语义读取

1. 对既有 transcript 用同一个 Whisper-large-v3 字符教师强制与无监督 head filter，取得 attention emission。
2. 用显式单调动态规划构造路径分布，前向后向求字符/词对窗口支持；不生成每路径 hateful prediction。
3. 把支持作为局部 speech-query attention 的条件项；全转录仍作为解释语境但必须声明其可见性、重复词映射和零支持处理。只产生一次 speech margin，原 visual/global/r6 固定。

[Whisper Has an Internal Word Aligner](https://arxiv.org/html/2509.09987v1)实际读引言、II 方法：原论文是字符重编码、head filtering、hard DTW；这里的 forward-backward 与 Qwen 支持读取是自研扩展，不声称来源已经提出。不能使用 oracle GT head。已有 [MultiHateLoc](https://arxiv.org/html/2512.10408v3) 3.1 把带时间戳句子嵌入重复到帧，[CLARA](https://arxiv.org/html/2608.15905v1) 3.1 按 Whisper 句子时间分段。因此“首次对齐文本”或“词时间戳本身是贡献”均不成立。

首个实验：完整 native/full 对照；随后同 attention 实现的 hard MAP、比例支持、保持词及支持熵的固定循环错时支持。若 full 不优于 hard MAP，不能主张传播不确定性；若准确时间优于比例而 posterior 不增益，只能保留输入质量诊断，不能冒充本候选机制成功。路径归一化、单调支持、字符→词→Qwen token 映射和空语音需在无标签合成例上验证。Whisper HF large-v3 本地存在，但原生 30s ASR 接口不自动等同源码接口。30–90 分钟仅计划值，含重放音频及 speech 重读；新视频同样支付这些调用。

## 第三选择：完整 VideoTree 迁移并实际叶重读

1. 用 Qwen 唯一视觉编码器提取 1fps 特征、确定性聚类，真实代表帧生成描述。
2. 完整执行聚类→caption→相关性判断→宽度扩展及相关簇深度扩展，保存所有叶与时间窗口的实际映射。
3. 用时间排序的树观测重新读全局，并用本地媒体与祖先观测独立读取窗口；不混入 Explorer 历史分数。

[VideoTree](https://arxiv.org/html/2405.19209v3) 实际读 3.1–3.3、实现细节 4：原文最后把节点 captions 按时间排序，不意味着必须把显式树字符串送进回答模型。这里“完整层级获取机制”与“回答时祖先联系”要分开做消融。Qwen 代替原 EVA-CLIP/captioner/LLM，是为固定模型约束的明确适配。不是 Explorer 第五版：采集规则、表示构建和全局重读均不同；若落地只剩加两帧或置信停止，就不能借 VideoTree 名义重开 Explorer。

首个实验：native/full 全语料；随后同一节点/像素/caption 的 flat、同 caption 数的均匀时间分区、祖先叶对应关系置换。最关键对照是保持观测内容不变，仅破坏结构；否则无法把加帧收益当层级证据。预设宽度 4/8/16、相关簇阈值 2、深度 2 是本项目适配，所有其余预算/截断规则也必须声明，不能按语料分支。60–180 分钟未测量，且每新视频增加 1fps 编码、K 个 caption、树决策、新全局和 2W 窗口读取。超过 2h 标完整实验待跑，不做预算性 idea 删除。

## 目标领域近邻：实际阅读范围与差异

| 一手来源 | 本轮实际阅读范围 | 对九候选的约束 |
|---|---|---|
| [MARS](https://arxiv.org/html/2601.15115v1) | 方法 2.2、四阶段与实验设置段 | 客观描述、正反假设及 meta synthesis 已用，不能重命名为新机制。 |
| [RAMF](https://arxiv.org/html/2512.02743v1) | 摘要/引言、方法目录与机制概述；没有逐式独立复核 3.4–3.5 | 生成多立场解释和 local/global fusion 不新；不据此否定具体 latent 或声学路径机制。 |
| [MATCH](https://jianlang.org/papers/MATCH.pdf) | 摘要/引言、III 总览、III-A/B 和 III-C 起始/图2 | 双立场 clue、时空检索、verifier 已有；程序/图必须多出可执行绑定而非同义解释。尚未逐式复核其全部 predictor。 |
| [MAESTRO](https://www.dsta.gov.sg/staticfile/ydsp/projects/files/reports/Report%20-%20Hear%20Me%20Out%20%28%26%20Think%29_%20MAESTRO%2C%20A%20Multimodal%20Agentic%20Model%20with%20Efficient%2C%20Synergistic%20Text-Reasoning%20Optimisation%20Framework.pdf) | 官方 PDF 搜索索引摘要及 state update/tool/decision 段；web 全文失败、直接下载 403 | 已有动态 global/local 重读、身份跟踪；本轮未确认有 VideoTree 式语义聚类层级，因此保留但标不确定，不能声称排除全部重叠。 |
| [LEAF](https://aclanthology.org/2026.findings-acl.604.pdf) | 摘要、3.1、3.2 Reason/Explain 及 Ground 图示 | 已有解释与标签/视频 grounding；不能笼统说 verification 是首次，不能照搬其标签条件。 |
| [IARE](https://arxiv.org/html/2606.11953v1) | 数据说明 3.3–3.4、方法 4.2–4.3 | 上下文注释、LoRA SFT、偏好路径 DPO 已有；与无标签连续实例状态优化不同。 |
| [CLARA](https://arxiv.org/html/2608.15905v1) | 3.1–3.3 的分段、MoE、local/global contrastive；摘要 | 句子分段和视频 rationale 已有；本轮未完整逐式核查 rationale gate，使用父任务阅读补充时应区分。 |
| [SAGE](https://aclanthology.org/2026.acl-long.817/) | 官方摘要 | 模态专家、deliberation 和判决仲裁已用；未据摘要断言全部内部操作。 |

引用范围刻意受实际打开内容约束；生成者在 CANDIDATES.json 记载的更深阅读并不冒充 jury 自己读过。引用相同缩写不等于同一机制。

## 检索记录、约束及后续

三条推荐机制各至少三种实际查询：

- latent：“hateful video latent optimization”；“Unsilencing Visual Latents inference”；“hate video continuous latent test time reasoning”；补充 2026/arXiv 限域查询。
- acoustic：“hateful video speech alignment posterior uncertainty”；“Whisper forward backward alignment attention”；“MultiHateLoc temporal alignment speech”；补充 arXiv/ACL 限域。
- tree：“hateful video VideoTree”；“MAESTRO hateful framework”；“hateful video semantic hierarchy cluster tree grounding”；补充 MAESTRO 官方站点 confidence/threshold/clustering。
- 另查 hateful-video quotation graph attribution，并查近期目标任务图方法。搜索跨 arXiv、官方 ACL、作者 PDF、官方报告；没有声称已穷尽 Google Scholar/Semantic Scholar 数据库。

检索与原始阅读响应保存在 runs/20261004_m1_ideation/jury_evidence/search_read_responses.json 和 supplementary_search_read.json。部分中途 web ref 的索引与预期页面不同；以响应标题/URL 为准，并已用直接 URL 重新读取 LEAF/CLARA，报告只采用核对后的对应关系。

已读本地 CLAUDE、RESEARCH_ITERATION_RULES、STATUS、M1 iteration/ideation README、完整九项 CANDIDATES、Explorer README 与 Grounder/Factorizer/Reinforcer 机制部分。未重新抄历史指标：以 STATUS 及原 runs 指针为准。旧规则中 OMSL 的静态基线不是这轮参考，必须使用 native Qwen3-VL-8B 配对复现+当前固定 r6。

所有首轮都报告两语料三项指标、within 有效视频 84/99；主门要求相同主指标双语料各 +.01，任何 pooled 损失不得超过 .005、within 不得超过 .01。每个最终 claimed novelty 部件也必须通过双语料同项 .01 消融和可证伪错误绑定对照。无标签打分，不扫描标签阈值、不按语料选开关、无预测 ensemble。结果标 development-selected。正式提案由父任务写成确定算法，再按仓库规则独立审查；本 jury 没有给未定义的常数发实现通行证。

技能阶段预算只管理至多三项 ideation pilot（每项目标 ≤2h，阶段至多8 GPUh、超时3h的 pilot 规则），不作为用户持续自主目标的总预算。没有运行，因此本轮 GPU 开销为零。这里的成本区间是生成者规划，不是基于当前十分钟 native 的可靠倍数。

