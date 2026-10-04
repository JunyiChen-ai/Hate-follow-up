# 候选25 provenance-bound temporal entity and discourse graph：独立提案审查

2026-10-05；独立实例 /root/latents_proposal，gpt-6-astra，与主agent相同模型、不同实例；**same-family provisional**。**规则4结论：PASS。** 这是一次提案审查及具体定义确认，不是有效性、晋级或最终novelty认证。

## 四项实际判定

| 规则4 STOP类别 | 判断与依据 |
|---|---|
| 来源完整方法已用于hateful video detection/localization | 本轮实际检索及可访问一手材料未确认，故不STOP。检索范围有限；WWW2026、MAESTRO全文未获取，不能宣称完整排除所有近邻。 |
| 纯ensemble | 不成立。同一冻结Qwen产生中立ledger、来源关系和最终局部margin；不混合独立模型预测，不投票，不平均native/new曲线。既定native全局/stance及r6保留。 |
| 纯calibration/后处理/平滑 | 不成立。图关系实际决定哪些远端PNG/ASR进入新局部测量，没有从旧分数出发的图校准。 |
| 纯工程而非完整科研方法 | 不成立。方案包含来源绑定实体/话语发现、跨窗关系、持久组件、两跳检索及实际端点媒体重读。新帧、ledger、prompt或来源ID本身均不作为贡献；完整机制超过输入工程。 |

不以“可能变成重复话题”“实体可能识别错”“成本高”“可能无收益”等实施前推理否决。来源只提供启发而非忠实复现，候选属于规则4允许的自研完整机制。

## 实际来源和目标检索

已读候选README全文及窄修后的step4/5/7、source_reads/source_scope.json、九项CANDIDATES对应项和jury rank7；重新核对规则4。CLAUDE.md沿用本同会话已完整读取的权威版本。未读GT、当前预测或main summary，未修改生产、未GPU、未计算内容哈希。

本轮新查询包含：
- `"hateful video" "entity graph"`
- `"hateful video" "persistent" graph`
- `"Semantic Event Graphs" "hate"`
- `"hate video" "discourse" graph evidence`
- `"HateMM" "scene graph"`
- `"hateful video" "knowledge graph"`
- `"hateful video" "temporal graph" entity`
- `"HateMM" "entity" "retrieval" graph`
- `"2601.06097" "hate"`

另查询WWW2026证据结构和MAESTRO工具近邻。原始请求/返回在 runs/20261005_m1_provenance/proposal_review/web_request_01.json至03及对应response。未命中不是全球absence证明；不把无关社会学/游戏/一般图论文命中当目标方法。一次无关opaque gist URL已剔除并留retention_note；未计算或依赖摘要值。

[Semantic Event Graphs](https://arxiv.org/html/2601.06097v1) 本轮独立直接抓取并阅读2.1–2.4、Algorithm1、5.2–5.3，正文保存semantic_event_graphs.txt。来源用YOLOv11持久track、接近START/END事件、MultiDiGraph、实体锚或词匹配检索，并将事件文字交给Gemini回答。其自身说明缺乏外观重识别、离镜事件和复杂共指能力；五视频/120问自动生成与自动评判仅是proof of concept。来源没有本候选的中立多模态话语ledger和端点图片重读；原Algorithm1的event-hash去重**不迁移**，只用可读坐标tuple。此区别已准确写入README。

| 目标一手近邻 | 实际阅读及有限差异 |
|---|---|
| [MATCH](https://jianlang.org/papers/MATCH.pdf) | 本轮刷新III-C访问，复用本独立实例此前已实际完整读取III-A–D/式3–7的内容。局部frame+ASR/OCR按CLIP线索检索、verifier及训练predictor已有。不能称首次真实媒体检索/grounding；所读方法未定义本候选持久实体/话语图路径控制的端点重读。 |
| [RAMF](https://arxiv.org/html/2512.02743v1) | 本轮直接抓取3.2/3.4/3.5，保存ramf_methods.txt；审机制，不是完整复现核验。客观/相反假设、local/global卷积池化门控及语义attention融合已有，不等同来源实例图检索。 |
| [Yadav/Singh WWW2026](https://doi.org/10.1145/3774905.3796488) | 本轮出版方索引摘要实际返回，DOI页面/PDF打开失败。已知共享证据结构、实体、确定性排序、跨模态一致/冲突、null及可选LLM reasoner。不能声称其全部方法被排除，也不能声称首次证据结构。可读摘要未建立完整本方法已使用。 |
| [MAESTRO官方报告](https://www.dsta.gov.sg/staticfile/ydsp/projects/files/reports/Report%20-%20Hear%20Me%20Out%20%28%26%20Think%29_%20MAESTRO%2C%20A%20Multimodal%20Agentic%20Model%20with%20Efficient%2C%20Synergistic%20Text-Reasoning%20Optimisation%20Framework.pdf) | 本轮官方索引附录的CONTINUE/chunk/tool、迭代图及工具分布可读；全文open失败。已有按chunk调用DeepFace/YOLO并更新内容，故工具编排/实体观察不是首创。未完整排除全部身份追踪细节，不据有限摘录宣布完整本方案重复。 |

可保留的窄研究单位是**来源绑定的跨时实体/话语关系实际控制媒体检索和局部测量**。文本引语图24主要干预跨度包及非对称attention；候选25的多模态发生实例、PTS图片、跨窗实际媒体读取和普通完整multimodal forward不同。该区别支持独立候选审查，不预判哪种更有效，也不授权借名称重置失败家族预算。

## 已确认的具体澄清

本次提出三项会影响输入或路径含义的定义，主agent修改后已实际核读：
1. local actor/target为action→entity，speaker为utterance→entity，quote_owner为quote→entity，quotes为本地utterance→copied quote。
2. 跨窗link已冻结中立system；quotes为引用端→被引端，responds_to为响应端→被响应端，retracts指向更早被撤回端；speaker_change严格earlier utterance→later utterance，same_entity对称规范化。类型/时间有计算检验，语义真伪仍待实证。
3. 最终source packet header、坐标格式、两分支字段已冻结，结尾为完整原yesno_question。Visual新增文本只含frame-supported entities/actions及允许关系，不加新ASR/quote/speaker文字；speech使用实际文本与speaker/quote/entity来源及图片。原native全ASR仍可见，无硬隔离主张。

未遗留本轮提案定义阻塞。之后规则6应查实际PTS选帧及坐标、唯一字符串映射、invalid整ledger/block UNKNOWN、组件/路径执行与实际端点媒体对应、两个full forward输入/native等价、无label路径。这些是观察完整性检查，不是新增理论审稿。

## 成本和结果归因边界

新视频支付完整PTS索引、最多2W图片、W次512-token ledger、ceil(W/8)次2048-token link以及每可用分支一次完整多模态重编码。120–300GPUmin/full333未测量；“冻结/缓存”不免除这些成本。完整表和source-only size preflight不得悄悄截断，实际时间/encoder calls需报。没有以预计耗时作STOP。

主门后已声明no_cross_edges、nearest_time、wrong_ownership、wrong_cited_media、no_entity_components。no_cross_edges把检索和更多像素一起去掉，单独不能证明图关系；nearest_time需在任何control分数前冻结实际像素count-pattern匹配，ASR长度差异仍限制语言预算归因。wrong_cited_media应仅扰动非本地证据、保留真实供体坐标及扰动覆盖；来源合法不能代替身份/所有权正确。完整及单部件贡献按README/规则14g的同一指标双语料.01要求验证，全部三指标、within视频数及development-selected保留。

**PASS，same-family provisional。** 一次规则4审查结束；不替代独立code review、完整333结果及最终机制/novelty确认。

