# M1 第二个完整九项池：一次独立 rule 4 裁定

日期：2026-10-05。审稿实例：`/root/ideation_jury_2`，fresh、同模型独立实例；**same-family provisional**。这是方案放行与实施优先顺序，不是性能、机制成功、晋级或论文首次性认证。

输入为 `experiments/20261005_m1_ideation/CANDIDATES.json` 的全部九项。按原数组顺序赋 C1–C9；九项全部读完后才排名，没有预删或合并。规则来自 `RESEARCH_ITERATION_RULES.md` 第4条与 `CLAUDE.md`；STATUS 仅用于现状/归档边界，不把其中转录数字当作本轮评测。没有打开 GT、预测或 metrics 文件，没有 GPU、模型调用或生产代码修改。

结论：**九项全部 PASS，STOP 0项**。四种 STOP 的任一项都没有得到本次检索与提案内容的充分支持。“未发现来源方法已用于目标任务”仅指下述访问范围，绝不等于证明不存在。训练到冻结模型的适配风险、可能 shortcut、可能无效、成本高或实现规格尚未齐全均不是额外 STOP 条件。

| 实施优先序 | 原序号 | dedup_key | 四项 STOP：目标任务已有 / 纯ensemble / 纯后处理 / 纯工程 | 裁定 |
|---:|---|---|---|---|
| 1 | C4 | anchor_identity_temporal_token_fusion | 未发现 / 否 / 否 / 否 | PASS |
| 2 | C6 | duration_normalized_interval_rotary_source_binding | 未发现 / 否 / 否 / 否 | PASS |
| 3 | C5 | importance_mass_temporal_transport_representation | 未发现 / 否 / 否 / 否 | PASS |
| 4 | C2 | target_conditioned_spatial_search_with_temporal_crop_memory | 未发现 / 否 / 否 / 否 | PASS |
| 5 | C3 | pixel_tracked_text_occurrence_memory_with_support_bound_reading | 未发现 / 否 / 否 / 否 | PASS |
| 6 | C7 | multikey_episodic_retrieval_with_neighbor_filtering | 未发现 / 否 / 否 / 否 | PASS |
| 7 | C8 | ordered_event_slot_retrieval_with_monotone_source_assignment | 未发现 / 否 / 否 / 否 | PASS |
| 8 | C1 | interactive_visual_timeline_retrieval_and_cut_memory | 未发现 / 否 / 否 / 否 | PASS |
| 9 | C9 | query_relevant_event_segmentation_with_background_acquisition | 未发现 / 否 / 否 / 否 | PASS |

优先序依据可明确隔离的机制、已有输入复用程度和新增调用成本，不依据“必然有效”的预测。前三项不需要逐窗生成事实链，适合先具体化；源 TTF 直接涉及 Qwen3-VL，完整算子和单次阈值也更明确。RoTE 直接检验来源区间表示；OTT-Vid 的质量、预算与多帧合并复杂度更多。后六项保持可实施资格；排后不是失败或取消。

## 实际检索与访问范围

原始工具响应保存在 `runs/20261005_m1_ideation/jury_evidence/`，包括 target_detail、source_reads、source_detail1–4、source_target_search_a–c 与 web_raw_initial。查询清单和逐项结构裁定见该目录 JSON。第一次广义目标任务检索当时没有单独持久化完整响应，故不伪称所有最初响应均已保存；随后实际重查与 primary 阅读响应已保存。

- [MATCH 作者全文](https://jianlang.org/papers/MATCH.pdf)：本轮成功打开14页 PDF，重点实际阅读 III-C/D（返回行467–529），确认线索与时空单元的视觉/文本相似度检索、核验和训练预测器。它排除了“首次时空证据检索/核验”的主张，但这些段落不等于完整视觉播放器循环或空间优先队列。
- [IARE 官方资源](https://github.com/DUT-lujunyu/IARE)及[作者预印本](https://arxiv.org/abs/2606.11953)：读到信息增强、解释训练、SFT/DPO 的作者描述；尝试 HTML。没有审查训练实现或数据标注，也不声称穷尽所有隐含视觉工具。
- [MARS](https://arxiv.org/html/2601.15115v1) §2、[RAMF](https://arxiv.org/html/2512.02743v1) §3.1–3.3：实际读取中性描述、正反假设推理、合成/融合。不能主张首次中性描述或多阶段推理。
- [LELA](https://arxiv.org/html/2602.09637v1) §3.1–3.2及方法图说明：实际确认 OCR、逐模态文字和帧分数；[CLARA](https://arxiv.org/html/2608.15905v1) §3.1–3.2：确认语句时间分段、OCR/转录与 clip MoE。它们否定泛化的“OCR/clip 语义分段首次性”，未在这些已读段落发现 C3 的像素失配触发发生支持机制。
- MAESTRO：[作者页](https://adrielkuek.github.io/publications/)和 DSTA 官方报告的搜索索引可访问，描述语义分段、模态对齐与动态 global/local reasoning。**DSTA PDF 直接打开返回 Internal Error；没有完整方法阅读**。因此不把“未见进度条/邻域检索/背景获取”当作全文排除证据。
- [WWW2026 evidence attribution](https://doi.org/10.1145/3774905.3796488)：搜索返回出版者摘要，含对象/OCR、共享证据结构、确定性编排与可选 LLM。只据此限定泛化主张，没有读全文。
- 九个来源均实际打开 primary 页面；细读 TTF §3.1、OTT-Vid §3.1–3.3、TIE §3.1–3.3、VTimeCoT §3.1–3.3、V* §3.2–3.3、VideoAgent §2.2–2.4、MERIT §3.1–3.4、Q2E §3.1及其整体融合描述、VideoEvent §3.1–3.3。源实现均未在本轮核读；官方代码的数值/接口不能以生成镜头的描述代替核验。

源名与 hateful/hate video 联合检索包含 VTimeCoT、V*/Guided Visual Search、VideoAgent、TTF/Temporal Token Fusion、OTT-Vid、RoTE/Time Interval Encoding、MERIT、Q2E、VideoEvent；也检索 token compression、pixel tracking/OCR。返回有不少无关结果，未用作正面证据。此类空缺不能构成“全球首次”的证明。

## 逐项理由与实现边界

### C4，优先1：TTF 锚身份压缩

完整研究机制是语义锚选择、局部空间匹配、阈值身份替代及来源坐标一致的稀疏序列。源方法虽然研究效率，其算法仍是完整科研方法；迁移它来检验局部排序并非只改分辨率或常数。没有多个模型预测融合，也不改输出分数。[TTF §3.1](https://arxiv.org/html/2605.07355v1)。

输入、阈值0.70、3×3邻域及 native G/S 保留已具体。新增约3次前缀阶段 forward/video、333视频12–25 GPU分钟是未测估计。实现前冻结多图边界、锚前置顺序、原始 Qwen position/cache 与 DeepStack gather；读取官方实现或明确逐条 paper-defined adaptation。**身份替代按源公式就是保留锚并删除冗余源行；“相同保留集合但仅删除”的控制可能完全等价，不能伪称独立消融。**真正机制控制是同数量非语义删减、锚选择和来源错绑。源QA保持不证明定位提升。

归档边界：不同于 Selector 的语言层 query 条件 mask、Recycler 的 sink attention 重分配；没有复用它们的选择规则。本轮允许独立提出，不重置旧族预算。与 C5 同属预语言压缩大类，但完整选择/聚合算子不同。

### C6，优先2：RoTE 区间来源表示

中心旋转、sinc频率衰减及由频率集合定义的时长归一化一起改变 query–source 时间点积，是完整表示方法；归一化发生在内部区间算子中，**不是输出 calibration**。原来源做事件条件视频生成；迁移到冻结理解模型属于尚未验证的适配，不因此 STOP。[TIE §3.1–3.3](https://arxiv.org/html/2605.10543v1)。

ASR segment、γ=4、全层全头、原 V/G 保留已具体。外层调用不增、333视频10–18分钟仅估计；丢失 fused kernel 的实际成本未测。实现前冻结实际 Qwen interleaved 时间频率子集、split-half配对、物理时间原点/单位、复制 body 来源映射和无时间语句规则；明确归一化在所用频率子集上怎样定义，并做 noGT 数值域检查。不能拿零半径控制宣称完整 native 等价：物理时间映射本身也变了。

归档边界：已读 Stabilizer 是图像 query 的头相位常量；Acoustic20 是 Whisper 字符路径支持进入语言 attention log-prior。C6 不改成这些算子的参数扫描，作为完整区间编码可独立放行；“同为时间绑定”不自动等于同族，也不自动给予新预算，预算归属仅限本明确机制。

### C5，优先3：OTT-Vid 非均匀质量传输

空间 coverage选择、leave-one-out质量、语义/空间传输和难度预算控制最终合并/删除，是完整表示算法。单一编码器内部不同帧的 token 聚合不构成禁止的多模型 ensemble。[OTT-Vid §3](https://arxiv.org/html/2605.11803v1)。

输入与0.50保留率、四个0.3参数已给；新增约3次前缀 forward/video、333视频15–35分钟仅估计。实现前还需核官方 Sinkhorn正则/迭代/停止值、预算取整/饱和、零重要性规则、链合并权重及根来源坐标、vision saliency 与 DeepStack适配。调用次数不是完整成本，coverage/Sinkhorn与矩阵内存都计入。保存根的多帧成员仅能溯源，不等于模型仍看到了全部发生时长。

它不同于 C4 的单锚身份删除，也不同于 Recycler；没有以更换压缩率复活归档族。实际机制必须由均匀质量、均匀预算及错误传输控制支持。

### C2，优先4：目标条件空间搜索

缺失目标提出→真实区域优先队列→观察后更新→坐标回映→视觉工作记忆形成完整获取方法。只有固定裁剪、升分辨率或换 OCR 才会落到纯工程边界。当前提案包含执行循环，PASS。[V* §3.2–3.3](https://arxiv.org/html/2312.14135v2)。

单 Qwen 取代源 LOC/heatmap 训练模块的功能角色写明，不能称原模型复现。W次目标检测+≤2A次搜索、333视频45–140分钟是未测估计；全触发最坏成本尚未具体。实施前冻结每窗帧选择、box/schema、优先队列分值/平局、两节点如何计数、最小区域与 UNKNOWN、生成 token 上限。MATCH 是检索核验近邻；所读方法没有本空间反馈优先队列。Explorer 搜索时间帧而非空间目标，独立机制边界成立。

### C3，优先5：像素跟踪文字发生记忆

贡献只能是像素对应支撑、失配触发文字重观察及按发生区间查询真实来源；不是 OCR、本体文字记忆或一般证据结构。源 VideoAgent 的 detector/re-ID ensemble 被明确排除，当前 CPU 跟踪+同Qwen读取不是纯ensemble。[VideoAgent §2.3](https://arxiv.org/html/2403.11481v2)；目标近邻为上述 LELA/CLARA/WWW 摘要。

W+≤2W次新增读取、333视频45–170分钟以及4fps CPU解码范围已给。**跟踪器仍写成“光流/模板”，尚非冻结算法**：实现前选定算法，定义前后向阈值、外观残差、cut检测、框初始化、文字变化规则、缺口/重现、查询支持与两次核验上限后的 UNKNOWN。局部文字初始读取 token 上限也需补齐。不是依据这些不确定性 STOP；必须保留原裁剪来区分可见性与 OCR正确性。与 Provenance25 的语义身份/话语边驱动远程检索不同，不得退化成它的 ledger 加密采帧版本。

### C7，优先6：MERIT 多键邻域检索

四键索引、最大键相似度、查询条件邻域过滤、一次不足续查和真实媒体回读保留了完整检索链。相似度选择输入而不融合仇恨判断；同一模型生成多种键不等于 ensemble。MATCH 已有语义线索检索，故不能主张首次检索；差异限定于多键+查询邻域扩展。[MERIT §3](https://arxiv.org/html/2608.07663v1)。

键/邻域数、两轮及生成上限基本具体；同Qwen hidden-state embedding替代专用 E 是功能适配而非源性能保证。W caption + W至2W过滤、5W至6W短文本prefill以及最终2W读取已计；20–60 GPU分钟/100窗不能宣传成整个333视频的低成本。实施前冻结 embedding wrapper、pooling token集合、短/空文本、四远程窗截取、两轮 union 与过滤输出长度、无源/UNKNOWN。输入接近 Provenance25，但不建立实体边或沿图遍历；不能换名复用图输出。

### C8，优先7：有序事件槽的实际来源选择

事件槽生成后被时间可行域约束、选择真实前后来源并重新读取，超出了只改 prompt。Q2E 的多模型、多通道熵融合未迁移；当前是自研有序来源选择，而非完整Q2E复现。[Q2E §3](https://arxiv.org/html/2506.10202v1)。

**目前三槽目标可分离**：current固定，prequel只能向前，sequel只能向后，且没有槽间转移/共享资源项；最优解等于两侧分别取最大值或NONE。允许实现为DP，但不得以“全局路径优化”或DP复杂性作贡献，也不能临时添加未审的新耦合来保住这一说法。可检验的机制仍是条件槽及有序真实来源绑定。

W caption+ceil(W/8)槽生成、4W embedding、2W最终读取，新增10–40分钟/100窗；需要计串行caption和总窗数。NONE分数、相似度归一化、空源、长度和literal模板实施前补齐。与 Program23 的 typed program、Provenance25 的图遍历不同，但“多一次文字重述”不能替代这里真实来源选择。

### C1，优先8：完整视觉时间工具循环

同步时间条、查询检索、高亮、真实cut和基于更新媒体的下一步动作一起构成完整科研方法。单独 overlay 会成为工程技巧，当前完整提案不属于该情形。检索分数不进入最终分数，没有 ensemble。[VTimeCoT §3](https://arxiv.org/html/2510.14672v1)。

单Qwen相关性读取替代源VideoCLIP-XL是明确适配。Q≤2、K≤3、1fps/8秒已具体；QW+≤K+1次生成和333视频65–195分钟是粗估。实施前明确检索top-k、最大记忆帧/token、实际cut预算、动作schema及每步生成上限，尤其说明循环是video级共享还是window级；若逐窗执行，就不能继续沿用只有K+1/video的调用式。Program23已有可执行来源工具，候选独立性必须来自实际视觉状态反馈与时间图形，而不是换工具名；MAESTRO全文访问缺口保留。

### C9，优先9：相关事件分段与背景获取

相关度平滑作用于证据选择，随后实际获取事件背景并做新的模型读取；完整机制不是对最终仇恨曲线做纯平滑，故不触发第4条。移动平均须作为显式方法部件进入消融，不能隐藏。[VideoEvent §3](https://aclanthology.org/2026.lrec-1.395.pdf)。

单中性事实查询取代源MCQA候选答案相关度求和，属于实质功能适配；不能称原算法数值复现。前后4窗、平滑宽3、均值阈值、≤4窗包、2W+B新增生成与上限已具体；新增15–50分钟/100窗不是已测整体成本。实施前冻结边界平滑/无效行、并列事件/代表帧、截取后的事件定义、文字/图像预算及SOURCE缺失规则。MAESTRO已有语义分段与global/local获取，所以只保留查询相关事件+实际代表背景的窄机制；全文未访问造成的不确定性如实保留。不是 Tree21 聚类树或 Explorer18 熵采帧的预算重开。

## 交给主 agent 的执行结论

无需再发起第二轮泛化 proposal review 来重审同一九项。选择后按上述缺口写成可执行、同语料统一规格，再按项目规则做独立 code review。补齐 literal prompts、schema、数值与总成本不是追加 STOP；它是实现必须具体的内容。任何实质改机制的版本需按原家族记录，不能偷偷借此改名或重置修订预算。

完整333两语料的真实主指标与原始局部排序、随后部件消融/错误来源控制仍然未发生。所有提案只有假设，没有证据称超过r6或满足共同≥.01门；未来 test选择结果须标 development-selected。
